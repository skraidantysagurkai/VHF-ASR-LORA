"""
Fine-tune per-condition LoRA adapters on top of a frozen Whisper model.

One adapter is trained per quality condition (perfect / good / okay / bad).
Each adapter is saved independently so they can be loaded and swapped at
inference time.

Usage
-----
    python train_lora.py \
        --manifest     data/merged/manifest.jsonl \
        --val-manifest data/dev/manifest.jsonl \
        --output-dir   adapters \
        [--model-id  openai/whisper-large-v3] \
        [--conditions perfect good okay bad] \
        [--rank 16] [--alpha 32] [--dropout 0.05] \
        [--epochs 5] [--batch 8] [--grad-accum 4] \
        [--lr 1e-4] [--warmup-ratio 0.1] \
        [--workers 4] [--seed 42] \
        [--resume]

"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from jiwer import wer as compute_wer
from peft import LoraConfig, get_peft_model, PeftModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_CONDITIONS = ["perfect", "good", "okay", "bad"]
LORA_TARGET_MODULES = ["q_proj", "v_proj", "out_proj", "fc1", "fc2"]
ITER_LOG_EVERY = 20  # log train loss every N iterations
EVAL_EVERY_FRACTION = 0.5  # eval 2 times per epoch
DEFAULT_EVAL_SAMPLES = 750  # samples used for mid-epoch eval (0 = full val set)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    audio_path: Path
    transcription: str


class ConditionDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        processor: WhisperProcessor,
        condition: str | None = None,
    ) -> None:
        self.processor = processor
        self.samples: list[Sample] = []
        root = manifest_path.parent

        with manifest_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if condition is not None and row.get("condition") != condition:
                    continue

                text = row.get("text")
                if text is None:
                    text = row.get("transcription")
                text = (
                    (text or "").strip().lower()
                )  # empty string valid for ESC50 noise
                self.samples.append(
                    Sample(
                        audio_path=root / row["audio_filepath"],
                        transcription=text,
                    )
                )

        if not self.samples:
            label = condition or "any"
            raise ValueError(
                f"No samples found for condition '{label}' in {manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        wav, sr = torchaudio.load(str(sample.audio_path))
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav, sr, 16_000)
        wav = wav.mean(0).numpy()

        inputs = self.processor(
            wav,
            sampling_rate=16_000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.squeeze(0)  # (80, 3000)

        if sample.transcription:
            labels = self.processor.tokenizer(
                text=sample.transcription,
                return_tensors="pt",
                padding=False,
            ).input_ids.squeeze(0)
        else:
            # Empty = no speech (ESC50 noise) — EOT token only
            labels = torch.tensor(
                [self.processor.tokenizer.eos_token_id], dtype=torch.long
            )

        return {"input_features": input_features, "labels": labels}


def collate_fn(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    input_features = torch.stack([b["input_features"] for b in batch])

    max_len = max(b["labels"].shape[0] for b in batch)
    labels = torch.full((len(batch), max_len), fill_value=-100, dtype=torch.long)
    for i, b in enumerate(batch):
        seq = b["labels"]
        labels[i, : seq.shape[0]] = seq

    return {"input_features": input_features, "labels": labels}


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_lora_model(
    model_id: str,
    rank: int,
    alpha: int,
    dropout: float,
    device: torch.device,
    resume_from: Path | None = None,
) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.gradient_checkpointing_enable()

    if resume_from is not None and (resume_from / "adapter_config.json").exists():
        print(f"  Loading LoRA weights from checkpoint: {resume_from}")
        model = PeftModel.from_pretrained(model, str(resume_from), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()
    return model.to(device)

CKPT_FILE = "train_checkpoint.pkl"


def save_checkpoint(
    ckpt_dir: Path, epoch: int, step: int, history: dict, optimizer: AdamW, scheduler
) -> None:
    ckpt = {
        "epoch": epoch,
        "step": step,
        "history": history,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    tmp = ckpt_dir / (CKPT_FILE + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ckpt, f)
    tmp.rename(ckpt_dir / CKPT_FILE)


def load_checkpoint(ckpt_dir: Path) -> dict | None:
    path = ckpt_dir / CKPT_FILE
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# EMA smoothing
# ---------------------------------------------------------------------------


def ema_smooth(values: list[float], weight: float = 0.9) -> list[float]:
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1.0 - weight)
        smoothed.append(last)
    return smoothed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def save_plots(
    condition: str,
    iter_steps: list[int],
    iter_losses: list[float],
    eval_steps: list[int],
    eval_losses: list[float],
    eval_wers: list[float],
    plot_dir: Path,
) -> None:
    fig, (ax_loss, ax_wer) = plt.subplots(1, 2, figsize=(16, 4))

    smoothed = ema_smooth(iter_losses, weight=0.9)
    ax_loss.plot(
        iter_steps,
        iter_losses,
        alpha=0.2,
        color="steelblue",
        linewidth=0.8,
        label="train (raw)",
    )
    ax_loss.plot(
        iter_steps,
        smoothed,
        color="steelblue",
        linewidth=1.5,
        label="train (EMA α=0.9)",
    )
    ax_loss.plot(
        eval_steps,
        eval_losses,
        color="tomato",
        linewidth=1.5,
        marker="o",
        markersize=4,
        label="val loss",
    )
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"Loss — {condition}")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_wer.plot(
        eval_steps,
        [w * 100 for w in eval_wers],
        color="darkorange",
        linewidth=1.5,
        marker="s",
        markersize=4,
        label="val WER",
    )
    ax_wer.set_xlabel("Iteration")
    ax_wer.set_ylabel("WER (%)")
    ax_wer.set_title(f"WER — {condition}")
    ax_wer.legend()
    ax_wer.grid(alpha=0.3)

    fig.suptitle(f"LoRA training — condition: {condition}")
    fig.tight_layout()
    path = plot_dir / f"{condition}_metrics.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved → {path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_condition(
    condition: str,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    warmup_ratio: float,
    grad_accum: int,
    output_dir: Path,
    plot_dir: Path,
    resume: bool = False,
    eval_samples: int = DEFAULT_EVAL_SAMPLES,
) -> None:
    adapter_dir = output_dir / condition
    adapter_dir.mkdir(parents=True, exist_ok=True)

    total_steps = (len(train_loader) * epochs) // grad_accum
    warmup_steps = int(total_steps * warmup_ratio)
    eval_every = max(1, int(len(train_loader) * EVAL_EVERY_FRACTION))  # 2× per epoch

    # Build a subsampled loader for fast mid-epoch evals
    # Full val_loader is used only at end of each epoch
    if eval_samples > 0 and eval_samples < len(val_loader.dataset):
        sub_indices = random.sample(range(len(val_loader.dataset)), eval_samples)
        sub_dataset = torch.utils.data.Subset(val_loader.dataset, sub_indices)
        fast_val_loader = DataLoader(
            sub_dataset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
            collate_fn=val_loader.collate_fn,
        )
        print(
            f"  Mid-epoch eval: {eval_samples} samples  |  Full epoch eval: {len(val_loader.dataset)} samples"
        )
    else:
        fast_val_loader = val_loader
        print(f"  Eval: full val set ({len(val_loader.dataset)} samples)")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-2,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # history
    iter_steps: list[int] = []
    iter_losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []
    eval_wers: list[float] = []

    best_val_wer = float("inf")
    best_val_loss = float("inf")
    global_iter = 0
    start_epoch = 1

    # --- resume from checkpoint ---
    if resume:
        ckpt = load_checkpoint(adapter_dir)
        if ckpt is not None:
            start_epoch = ckpt["epoch"]
            global_iter = ckpt["step"]
            hist = ckpt["history"]
            iter_steps = hist["iter_steps"]
            iter_losses = hist["iter_losses"]
            eval_steps = hist["eval_steps"]
            eval_losses = hist["eval_losses"]
            eval_wers = hist["eval_wers"]
            best_val_wer = min(eval_wers) if eval_wers else float("inf")
            best_val_loss = min(eval_losses) if eval_losses else float("inf")
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            print(f"  Resumed from epoch {start_epoch}, global_iter {global_iter}")

    print(f"\n{'=' * 60}")
    print(
        f"  Condition: {condition}  |  train={len(train_loader.dataset):,}  val={len(val_loader.dataset):,}"
    )
    print(
        f"  Steps: {total_steps}  |  warmup: {warmup_steps}  |  eval every: {eval_every} iters"
    )
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        optimizer.zero_grad()

        pbar = tqdm(
            train_loader,
            desc=f"[{condition}] Epoch {epoch}/{epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        for step, batch in enumerate(pbar, 1):
            input_features = batch["input_features"].to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(
                    input_features=input_features,
                    labels=labels,
                )

            loss = outputs.loss / grad_accum
            loss.backward()

            running_loss += outputs.loss.item()
            running_n += 1

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % ITER_LOG_EVERY == 0:
                avg = running_loss / running_n
                iter_steps.append(global_iter + step)
                iter_losses.append(avg)
                running_loss = 0.0
                running_n = 0
                pbar.set_postfix(
                    loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )

            if step % eval_every == 0:
                val_loss, val_wer = evaluate(model, fast_val_loader, device, processor)
                eval_steps.append(global_iter + step)
                eval_losses.append(val_loss)
                eval_wers.append(val_wer)
                model.train()

                marker = ""
                if val_wer < best_val_wer:
                    best_val_loss = val_loss
                    best_val_wer = val_wer
                    model.save_pretrained(adapter_dir)
                    processor.save_pretrained(adapter_dir)
                    marker = "  ← best"

                tqdm.write(
                    f"  [{condition}] epoch {epoch}  iter {global_iter + step:>6}"
                    f"  val_loss={val_loss:.4f}  val_wer={val_wer:.4f}{marker}"
                )

                # save checkpoint after every eval
                history = {
                    "iter_steps": iter_steps,
                    "iter_losses": iter_losses,
                    "eval_steps": eval_steps,
                    "eval_losses": eval_losses,
                    "eval_wers": eval_wers,
                }
                save_checkpoint(
                    adapter_dir,
                    epoch + 1,
                    global_iter + len(train_loader),
                    history,
                    optimizer,
                    scheduler,
                )

        global_iter += len(train_loader)

        # full val eval at end of every epoch
        full_loss, full_wer = evaluate(model, val_loader, device, processor)
        tqdm.write(
            f"  [{condition}] epoch {epoch} FULL eval  "
            f"val_loss={full_loss:.4f}  val_wer={full_wer:.4f}"
        )

    # final eval
    val_loss, val_wer = evaluate(model, val_loader, device, processor)
    eval_steps.append(global_iter)
    eval_losses.append(val_loss)
    eval_wers.append(val_wer)
    if val_wer < best_val_wer:
        best_val_loss = val_loss
        best_val_wer = val_wer
        model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)

    print(
        f"  [{condition}] best val loss: {best_val_loss:.4f}  best val WER: {best_val_wer:.4f}  →  {adapter_dir}"
    )

    save_plots(
        condition, iter_steps, iter_losses, eval_steps, eval_losses, eval_wers, plot_dir
    )

    hist = {
        "iter_steps": iter_steps,
        "iter_losses": iter_losses,
        "eval_steps": eval_steps,
        "eval_losses": eval_losses,
        "eval_wers": eval_wers,
    }
    with open(plot_dir / f"{condition}_history.pkl", "wb") as f:
        pickle.dump(hist, f)

    # remove checkpoint once fully done
    ckpt_path = adapter_dir / CKPT_FILE
    if ckpt_path.exists():
        ckpt_path.unlink()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: WhisperForConditionalGeneration,
    loader: DataLoader,
    device: torch.device,
    processor: WhisperProcessor,
) -> tuple[float, float]:
    """Returns (val_loss, val_wer)."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_refs: list[str] = []
    all_hyps: list[str] = []

    # suppress repetitive generate() warnings
    import transformers

    prev_verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()

    for batch in tqdm(loader, desc="  eval", leave=False, dynamic_ncols=True):
        input_features = batch["input_features"].to(
            device, dtype=torch.bfloat16, non_blocking=True
        )
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_features=input_features, labels=labels)
        total_loss += outputs.loss.item() * len(labels)
        total_n += len(labels)

        label_ids = labels.clone()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # greedy decode — free VRAM immediately after
        predicted_ids = model.generate(
            input_features=input_features,
            num_beams=1,
            max_new_tokens=128,
            language="en",
            task="transcribe",
        )

        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        hyps = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        all_refs.extend(refs)
        all_hyps.extend(hyps)

        # free generated tensor immediately to avoid VRAM creep
        del predicted_ids, labels, input_features, outputs

    transformers.logging.set_verbosity(prev_verbosity)

    val_loss = total_loss / total_n
    val_wer = compute_wer(all_refs, all_hyps)
    return val_loss, val_wer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--val-manifest", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("adapters"))
    p.add_argument("--model-id", default="openai/whisper-large-v3")
    p.add_argument(
        "--conditions", nargs="+", default=ALL_CONDITIONS, choices=ALL_CONDITIONS
    )
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-dir", type=Path, default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training: skip completed conditions, "
        "continue interrupted ones from last checkpoint.",
    )
    p.add_argument(
        "--eval-samples",
        type=int,
        default=DEFAULT_EVAL_SAMPLES,
        help=f"Number of val samples for mid-epoch evals (default: {DEFAULT_EVAL_SAMPLES}, 0=full)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model_id}")
    print(f"LoRA:   rank={args.rank}  alpha={args.alpha}  dropout={args.dropout}")
    print(f"Resume: {args.resume}")

    plot_dir = args.plot_dir or (args.output_dir / "lora_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    processor = WhisperProcessor.from_pretrained(args.model_id)
    processor.tokenizer.clean_up_tokenization_spaces = False
    pad_id = processor.tokenizer.pad_token_id

    _collate = lambda b: collate_fn(b, pad_id)  # noqa: E731

    for condition in args.conditions:
        adapter_dir = args.output_dir / condition

        # Skip fully completed conditions when resuming
        if (
            args.resume
            and (adapter_dir / "adapter_config.json").exists()
            and not (adapter_dir / CKPT_FILE).exists()
        ):
            print(f"\n  [{condition}] adapter complete — skipping")
            continue

        print(f"\n{'#' * 60}")
        print(f"  Building dataset for condition: {condition}")
        print(f"{'#' * 60}")

        train_ds = ConditionDataset(args.manifest, processor, condition=condition)

        if args.val_manifest is not None:
            val_ds = ConditionDataset(args.val_manifest, processor, condition=condition)
            train_split = train_ds
            val_split = val_ds
            print(f"  Val manifest: {args.val_manifest}  ({len(val_ds):,} samples)")
        else:
            n_val = max(1, int(0.1 * len(train_ds)))
            n_train = len(train_ds) - n_val
            train_split, val_split = torch.utils.data.random_split(
                train_ds,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(args.seed),
            )
            print(
                f"  No val manifest — 90/10 split  (train={n_train:,}  val={n_val:,})"
            )

        train_loader = DataLoader(
            train_split,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            collate_fn=_collate,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_split,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            collate_fn=_collate,
        )

        resume_from = adapter_dir if args.resume else None
        model = build_lora_model(
            model_id=args.model_id,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            device=device,
            resume_from=resume_from,
        )

        train_condition(
            condition=condition,
            model=model,
            processor=processor,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            grad_accum=args.grad_accum,
            output_dir=args.output_dir,
            plot_dir=plot_dir,
            resume=args.resume,
            eval_samples=args.eval_samples,
        )

        del model
        torch.cuda.empty_cache()

    print("\nAll adapters trained. Done.")


if __name__ == "__main__":
    main()
