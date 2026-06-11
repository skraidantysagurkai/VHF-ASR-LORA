"""
Quick sanity-check: train ONE LoRA adapter on 1/8 of all data (stratified
across all 4 quality conditions) for 5 epochs, then evaluate on 1/8 of the
val set using the same stratified strategy.

Audio-length statistics are printed before training begins.

Usage
-----
    python train_lora_quick.py \
        --manifest     data/merged/manifest.jsonl \
        --val-manifest data/dev/manifest.jsonl \
        --output-dir   adapters_quick \
        [--model-id  openai/whisper-large-v3] \
        [--rank 16] [--alpha 32] [--dropout 0.05] \
        [--epochs 5] [--batch 8] [--grad-accum 4] \
        [--lr 1e-4] [--warmup-ratio 0.1] \
        [--workers 4] [--seed 42]

    If --val-manifest is omitted a 90/10 split of the train manifest is used
    (split is applied after the 1/4 stratified sub-sample).

Output layout
-------------
    adapters_quick/
        all/            <- PEFT adapter_config.json + adapter_model.safetensors
        lora_plots/
            all_metrics.png
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
ITER_LOG_EVERY = 20
EVAL_EVERY_FRACTION = 0.5
DEFAULT_EVAL_SAMPLES = 500
ADAPTER_NAME = "all"


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
# Manifest parsing & stratified sub-sampling
# ---------------------------------------------------------------------------


@dataclass
class ManifestRow:
    audio_path: Path
    transcription: str
    condition: str


def load_manifest(manifest_path: Path) -> dict[str, list[ManifestRow]]:
    """Returns rows grouped by condition."""
    root = manifest_path.parent
    groups: dict[str, list[ManifestRow]] = {c: [] for c in ALL_CONDITIONS}

    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            condition = row.get("condition", "")
            if condition not in ALL_CONDITIONS:
                continue
            text = row.get("text") or row.get("transcription") or ""
            groups[condition].append(
                ManifestRow(
                    audio_path=root / row["audio_filepath"],
                    transcription=text.strip().lower(),
                    condition=condition,
                )
            )

    return groups


def stratified_eighth(
    groups: dict[str, list[ManifestRow]],
    seed: int,
) -> list[ManifestRow]:
    """Take floor(n/8) rows per condition, shuffle within each group first."""
    rng = random.Random(seed)
    subset: list[ManifestRow] = []
    for condition in ALL_CONDITIONS:
        rows = list(groups[condition])
        rng.shuffle(rows)
        n = max(1, len(rows) // 4)
        subset.extend(rows[:n])
        print(f"  {condition:8s}: {len(rows):>6,} total  →  {n:>5,} sampled (1/8)")
    return subset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MixedDataset(Dataset):
    def __init__(self, rows: list[ManifestRow], processor: WhisperProcessor) -> None:
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        wav, sr = torchaudio.load(str(row.audio_path))
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav, sr, 16_000)
        wav = wav.mean(0).numpy()

        inputs = self.processor(wav, sampling_rate=16_000, return_tensors="pt")
        input_features = inputs.input_features.squeeze(0)

        if row.transcription:
            labels = self.processor.tokenizer(
                text=row.transcription, return_tensors="pt", padding=False
            ).input_ids.squeeze(0)
        else:
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
# Audio length statistics
# ---------------------------------------------------------------------------


def print_audio_stats(rows: list[ManifestRow], label: str) -> None:
    print(f"\n--- Audio length statistics: {label} ({len(rows):,} samples) ---")
    durations: list[float] = []
    for row in tqdm(
        rows, desc=f"  reading audio info ({label})", leave=False, dynamic_ncols=True
    ):
        try:
            info = torchaudio.info(str(row.audio_path))
            durations.append(info.num_frames / info.sample_rate)
        except Exception:
            pass

    if not durations:
        print("  (no audio files could be read)")
        return

    arr = np.array(durations)
    total_h = arr.sum() / 3600
    print(f"  count   : {len(arr):,}")
    print(f"  min     : {arr.min():.2f} s")
    print(f"  p25     : {np.percentile(arr, 25):.2f} s")
    print(f"  median  : {np.median(arr):.2f} s")
    print(f"  mean    : {arr.mean():.2f} s")
    print(f"  p75     : {np.percentile(arr, 75):.2f} s")
    print(f"  p95     : {np.percentile(arr, 95):.2f} s")
    print(f"  max     : {arr.max():.2f} s")
    print(f"  std     : {arr.std():.2f} s")
    print(f"  total   : {arr.sum():.1f} s  ({total_h:.2f} h)")

    # per-condition breakdown
    by_cond: dict[str, list[float]] = {c: [] for c in ALL_CONDITIONS}
    for row, dur in zip(rows, durations):
        by_cond[row.condition].append(dur)
    print("  per condition:")
    for c in ALL_CONDITIONS:
        ds = by_cond[c]
        if ds:
            a = np.array(ds)
            print(
                f"    {c:8s}: n={len(a):>5,}  mean={a.mean():.2f}s  median={np.median(a):.2f}s  total={a.sum() / 60:.1f}min"
            )
    print()


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


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

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
# EMA smoothing & plotting
# ---------------------------------------------------------------------------


def ema_smooth(values: list[float], weight: float = 0.9) -> list[float]:
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1.0 - weight)
        smoothed.append(last)
    return smoothed


def save_plots(
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
    ax_loss.set_title("Loss — all conditions (1/8 sample)")
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
    ax_wer.set_title("WER — all conditions (1/8 sample)")
    ax_wer.legend()
    ax_wer.grid(alpha=0.3)

    fig.suptitle("LoRA quick-train — all conditions, 1/8 data")
    fig.tight_layout()
    path = plot_dir / f"{ADAPTER_NAME}_metrics.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved → {path}")


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
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_refs: list[str] = []
    all_hyps: list[str] = []

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

        del predicted_ids, labels, input_features, outputs

    transformers.logging.set_verbosity(prev_verbosity)

    val_loss = total_loss / total_n
    val_wer = compute_wer(all_refs, all_hyps)
    return val_loss, val_wer


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_all(
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
    adapter_dir = output_dir / ADAPTER_NAME
    adapter_dir.mkdir(parents=True, exist_ok=True)

    total_steps = (len(train_loader) * epochs) // grad_accum
    warmup_steps = int(total_steps * warmup_ratio)
    eval_every = max(1, int(len(train_loader) * EVAL_EVERY_FRACTION))

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

    iter_steps: list[int] = []
    iter_losses: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []
    eval_wers: list[float] = []

    best_val_wer = float("inf")
    best_val_loss = float("inf")
    global_iter = 0
    start_epoch = 1

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
        f"  Adapter: {ADAPTER_NAME}  |  train={len(train_loader.dataset):,}  val={len(val_loader.dataset):,}"
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
            desc=f"Epoch {epoch}/{epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        for step, batch in enumerate(pbar, 1):
            input_features = batch["input_features"].to(
                device, dtype=torch.bfloat16, non_blocking=True
            )
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(input_features=input_features, labels=labels)

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
                    f"  epoch {epoch}  iter {global_iter + step:>6}"
                    f"  val_loss={val_loss:.4f}  val_wer={val_wer:.4f}{marker}"
                )

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

        full_loss, full_wer = evaluate(model, val_loader, device, processor)
        tqdm.write(
            f"  epoch {epoch} FULL eval  "
            f"val_loss={full_loss:.4f}  val_wer={full_wer:.4f}"
        )

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
        f"  Best val loss: {best_val_loss:.4f}  best val WER: {best_val_wer:.4f}  →  {adapter_dir}"
    )

    save_plots(iter_steps, iter_losses, eval_steps, eval_losses, eval_wers, plot_dir)

    hist = {
        "iter_steps": iter_steps,
        "iter_losses": iter_losses,
        "eval_steps": eval_steps,
        "eval_losses": eval_losses,
        "eval_wers": eval_wers,
    }
    with open(plot_dir / f"{ADAPTER_NAME}_history.pkl", "wb") as f:
        pickle.dump(hist, f)

    ckpt_path = adapter_dir / CKPT_FILE
    if ckpt_path.exists():
        ckpt_path.unlink()


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
    p.add_argument("--output-dir", type=Path, default=Path("adapters_quick"))
    p.add_argument("--model-id", default="openai/whisper-small")
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
        "--eval-samples",
        type=int,
        default=DEFAULT_EVAL_SAMPLES,
        help=f"Val samples for mid-epoch eval (default: {DEFAULT_EVAL_SAMPLES}, 0=full)",
    )
    p.add_argument("--resume", action="store_true")
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

    # ------------------------------------------------------------------
    # Stratified 1/4 sub-sampling
    # ------------------------------------------------------------------
    print(f"\n{'#' * 60}")
    print("  Stratified 1/4 sub-sampling — TRAIN")
    print(f"{'#' * 60}")
    train_groups = load_manifest(args.manifest)
    train_rows = stratified_eighth(train_groups, seed=args.seed)
    print(f"  Total train rows after sampling: {len(train_rows):,}")

    if args.val_manifest is not None:
        if not args.val_manifest.exists():
            raise FileNotFoundError(f"Val manifest not found: {args.val_manifest}")
        print(f"\n{'#' * 60}")
        print("  Stratified 1/4 sub-sampling — VAL")
        print(f"{'#' * 60}")
        val_groups = load_manifest(args.val_manifest)
        val_rows = stratified_eighth(val_groups, seed=args.seed)
        print(f"  Total val rows after sampling: {len(val_rows):,}")
    else:
        val_rows = None  # will split from train below

    # ------------------------------------------------------------------
    # Audio length statistics (before training)
    # ------------------------------------------------------------------
    print_audio_stats(train_rows, "train")
    if val_rows is not None:
        print_audio_stats(val_rows, "val")

    # ------------------------------------------------------------------
    # Build datasets & loaders
    # ------------------------------------------------------------------
    processor = WhisperProcessor.from_pretrained(args.model_id)
    processor.tokenizer.clean_up_tokenization_spaces = False
    pad_id = processor.tokenizer.pad_token_id
    _collate = lambda b: collate_fn(b, pad_id)  # noqa: E731

    random.shuffle(train_rows)  # ensure mixed order regardless of condition

    train_ds = MixedDataset(train_rows, processor)

    if val_rows is not None:
        val_ds = MixedDataset(val_rows, processor)
    else:
        n_val = max(1, int(0.1 * len(train_ds)))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"  No val manifest — 90/10 split  (train={n_train:,}  val={n_val:,})")
        # recompute stats on the split val subset
        split_val_rows = [train_rows[i] for i in val_ds.indices]
        print_audio_stats(split_val_rows, "val (split)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    adapter_dir = args.output_dir / ADAPTER_NAME
    resume_from = adapter_dir if args.resume else None
    model = build_lora_model(
        model_id=args.model_id,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        device=device,
        resume_from=resume_from,
    )

    train_all(
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

    print("\nQuick-train done.")


if __name__ == "__main__":
    main()
