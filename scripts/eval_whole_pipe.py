from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torchaudio
from jiwer import wer as compute_wer
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from class_models import ECAPAClassifier, SAMPLE_RATE  # noqa: local import

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSES = ["perfect", "good", "okay", "bad"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

_FRAME_SIZE = 160  # 10 ms at 16 kHz


# ---------------------------------------------------------------------------
# Active-audio extraction
# ---------------------------------------------------------------------------


def active_clip(wav: torch.Tensor, threshold_db: float, clip_len: int) -> torch.Tensor:
    """Concatenate frames above threshold_db dBFS; clip/pad to clip_len samples."""
    thresh_amp = 10.0 ** (threshold_db / 20.0)
    n_frames = wav.shape[0] // _FRAME_SIZE
    trimmed = wav[: n_frames * _FRAME_SIZE]
    frames = trimmed.unfold(0, _FRAME_SIZE, _FRAME_SIZE)  # (N, FRAME_SIZE)
    rms = frames.pow(2).mean(-1).sqrt()
    mask = rms > thresh_amp
    active = frames[mask].reshape(-1) if mask.any() else trimmed
    if active.shape[0] < clip_len:
        active = torch.nn.functional.pad(active, (0, clip_len - active.shape[0]))
    else:
        active = active[:clip_len]
    return active


# ---------------------------------------------------------------------------
# Dataset — loads audio + optional reference transcription + optional GT label
# ---------------------------------------------------------------------------


class EvalDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
    ) -> None:
        self.root = manifest_path.parent
        self.samples: list[dict[str, Any]] = []

        with manifest_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("text") or row.get("transcription") or ""
                self.samples.append(
                    {
                        "path": row["audio_filepath"],
                        "text": text.strip().lower(),
                        "condition": row.get("condition"),
                    }
                )

        if not self.samples:
            raise ValueError(f"No samples found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.samples[idx]
        wav, sr = torchaudio.load(str(self.root / item["path"]))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)  # mono

        return {
            "waveform": wav,
            "text": item["text"],
            "condition": item["condition"],
            "idx": idx,
        }


def make_collate(threshold_db: float, clip_seconds: float):
    clip_len = int(clip_seconds * SAMPLE_RATE)

    def collate_eval(batch: list[dict]) -> dict:
        return {
            "waveforms": torch.stack(
                [active_clip(b["waveform"], threshold_db, clip_len) for b in batch]
            ),
            "texts": [b["text"] for b in batch],
            "conditions": [b["condition"] for b in batch],
            "idxs": [b["idx"] for b in batch],
        }

    return collate_eval


# ---------------------------------------------------------------------------
# Step 1: ECAPA-TDNN routing
# ---------------------------------------------------------------------------


@torch.no_grad()
def classify_all(
    dataset: EvalDataset,
    ecapa_ckpt: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    threshold_db: float,
    clip_seconds: float,
) -> list[str]:
    """Return a list of predicted condition strings, one per sample."""
    num_classes = len(CLASSES)
    model = ECAPAClassifier(num_classes=num_classes).to(device)

    ckpt = torch.load(ecapa_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=make_collate(threshold_db, clip_seconds),
    )

    predictions: dict[int, str] = {}
    for batch in tqdm(loader, desc="ECAPA routing", dynamic_ncols=True):
        wavs = batch["waveforms"].to(device, non_blocking=True)
        logits, _, _ = model(wavs)
        pred_idxs = logits.argmax(-1).cpu().tolist()
        for sample_idx, pred_idx in zip(batch["idxs"], pred_idxs):
            predictions[sample_idx] = CLASSES[pred_idx]

    # return in original order
    return [predictions[i] for i in range(len(dataset))]


# ---------------------------------------------------------------------------
# Step 2: Whisper feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    waveform: torch.Tensor,  # (N_samples,) float32 numpy-able
    processor: WhisperProcessor,
) -> torch.Tensor:
    wav_np = waveform.numpy()
    feats = processor(wav_np, sampling_rate=16_000, return_tensors="pt")
    return feats.input_features.squeeze(0)  # (80, 3000)


# ---------------------------------------------------------------------------
# Step 3: Whisper + LoRA transcription for one condition group
# ---------------------------------------------------------------------------


@torch.no_grad()
def transcribe_group(
    condition: str,
    sample_indices: list[int],
    dataset: EvalDataset,
    whisper_model_id: str,
    adapters_dir: Path,
    device: torch.device,
    processor: WhisperProcessor,
    batch_size: int,
) -> tuple[list[str], list[str]]:
    """
    Returns (hypotheses, references) for the given condition group.
    Loads the LoRA adapter for `condition`, runs greedy decoding.
    """
    adapter_path = adapters_dir / condition
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No LoRA adapter found for condition '{condition}' at {adapter_path}"
        )

    print(f"\n  Loading LoRA adapter: {condition} ({len(sample_indices)} samples)")

    base_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_model_id,
        torch_dtype=torch.bfloat16,
    )
    # Match training config exactly
    base_model.config.forced_decoder_ids = None
    base_model.generation_config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []  # empty list, not None
    base_model.generation_config.suppress_tokens = []
    base_model.generation_config.begin_suppress_tokens = []

    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    model = model.to(device)
    model.eval()

    # Build feature tensors for this group
    all_features = []
    all_refs = []
    for idx in sample_indices:
        item = dataset[idx]
        feat = extract_features(item["waveform"], processor)
        all_features.append(feat)
        all_refs.append(item["text"])

    hyps: list[str] = []

    # Process in mini-batches to avoid OOM
    for start in tqdm(
        range(0, len(all_features), batch_size),
        desc=f"  Transcribing [{condition}]",
        dynamic_ncols=True,
        leave=False,
    ):
        batch_feats = torch.stack(all_features[start : start + batch_size])
        batch_feats = batch_feats.to(device, dtype=torch.bfloat16, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            predicted_ids = model.generate(
                input_features=batch_feats,
                num_beams=1,
                language="en",
                task="transcribe",
            )

        batch_hyps = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        hyps.extend([h.strip().lower() for h in batch_hyps])
        del predicted_ids, batch_feats

    del model, base_model
    torch.cuda.empty_cache()

    return hyps, all_refs


# ---------------------------------------------------------------------------
# WER table printer
# ---------------------------------------------------------------------------


def print_wer_table(
    per_condition: dict[str, dict],  # condition -> {wer, n_samples, n_correct_route}
    overall_refs: list[str],
    overall_hyps: list[str],
    routing_accuracy: float | None,
) -> None:
    overall_wer = compute_wer(overall_refs, overall_hyps)

    col_w = [12, 10, 10, 16]
    header = (
        f"{'Condition':<{col_w[0]}}"
        f"{'Samples':>{col_w[1]}}"
        f"{'WER (%)':>{col_w[2]}}"
        f"{'Route Acc (%)':>{col_w[3]}}"
    )
    sep = "-" * sum(col_w)

    print("\n" + "=" * sum(col_w))
    print("  WER by Condition (LoRA-routed)")
    print("=" * sum(col_w))
    print(header)
    print(sep)

    for cond in CLASSES:
        if cond not in per_condition:
            continue
        info = per_condition[cond]
        wer_pct = info["wer"] * 100
        route_acc = (
            f"{info['route_acc'] * 100:.1f}" if info["route_acc"] is not None else "N/A"
        )
        print(
            f"{cond:<{col_w[0]}}"
            f"{info['n_samples']:>{col_w[1]}}"
            f"{wer_pct:>{col_w[2]}.2f}"
            f"{route_acc:>{col_w[3]}}"
        )

    print(sep)
    total_samples = sum(v["n_samples"] for v in per_condition.values())
    ra_str = f"{routing_accuracy * 100:.1f}" if routing_accuracy is not None else "N/A"
    print(
        f"{'OVERALL':<{col_w[0]}}"
        f"{total_samples:>{col_w[1]}}"
        f"{overall_wer * 100:>{col_w[2]}.2f}"
        f"{ra_str:>{col_w[3]}}"
    )
    print("=" * sum(col_w))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing manifest.jsonl",
    )
    p.add_argument(
        "--ecapa-ckpt",
        type=Path,
        required=True,
        help="Path to ecapa_best.pt checkpoint",
    )
    p.add_argument(
        "--adapters-dir",
        type=Path,
        required=True,
        help="Root dir with per-condition LoRA adapter folders",
    )
    p.add_argument(
        "--whisper-model",
        default="openai/whisper-large-v3",
        help="HuggingFace model ID for Whisper (default: whisper-large-v3)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for both ECAPA routing and Whisper decoding",
    )
    p.add_argument(
        "--clip-sec",
        type=float,
        default=5.0,
        help="Audio clip length in seconds fed to ECAPA (default: 5.0)",
    )
    p.add_argument(
        "--threshold-db",
        type=float,
        default=-50.0,
        help="dBFS RMS threshold for active-audio extraction (default: -20)",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--no-gt-routing",
        action="store_true",
        help="Force ECAPA routing even when GT condition labels exist "
        "(default: use ECAPA routing always; GT is used only for "
        "reporting routing accuracy)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Whisper model: {args.whisper_model}")
    print(f"Adapters dir:  {args.adapters_dir}")
    print(f"ECAPA ckpt:    {args.ecapa_ckpt}")
    print(f"Threshold:     {args.threshold_db} dBFS")
    print(f"Clip length:   {args.clip_sec} s")

    manifest = args.test_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    dataset = EvalDataset(manifest)
    print(f"Test samples:  {len(dataset):,}")

    # -----------------------------------------------------------------------
    # Step 1: ECAPA routing
    # -----------------------------------------------------------------------
    predicted_conditions = classify_all(
        dataset=dataset,
        ecapa_ckpt=args.ecapa_ckpt,
        device=device,
        batch_size=args.batch,
        workers=args.workers,
        threshold_db=args.threshold_db,
        clip_seconds=args.clip_sec,
    )

    # Compute routing accuracy if GT labels are available
    gt_conditions = [dataset.samples[i]["condition"] for i in range(len(dataset))]
    has_gt = all(c is not None for c in gt_conditions)
    routing_accuracy: float | None = None
    if has_gt:
        n_correct = sum(
            p == g
            for p, g in zip(predicted_conditions, gt_conditions)
            if g in CLASS_TO_IDX
        )
        n_valid = sum(1 for g in gt_conditions if g in CLASS_TO_IDX)
        routing_accuracy = n_correct / n_valid if n_valid > 0 else None
        if routing_accuracy:
            print(
                f"\nECAPA routing accuracy: {routing_accuracy * 100:.2f}%  "
                f"({n_correct}/{n_valid})"
            )

    # -----------------------------------------------------------------------
    # Step 2: Group sample indices by predicted condition
    # -----------------------------------------------------------------------
    condition_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, cond in enumerate(predicted_conditions):
        condition_to_indices[cond].append(i)

    print("\nSamples per predicted condition:")
    for cond in CLASSES:
        n = len(condition_to_indices.get(cond, []))
        if n:
            print(f"  {cond:<10} {n:>6}")

    # -----------------------------------------------------------------------
    # Step 3: Load Whisper processor once
    # -----------------------------------------------------------------------
    processor = WhisperProcessor.from_pretrained(args.whisper_model)
    processor.tokenizer.clean_up_tokenization_spaces = False

    # -----------------------------------------------------------------------
    # Step 4: Transcribe each condition group with its LoRA adapter
    # -----------------------------------------------------------------------
    per_condition: dict[str, dict] = {}
    all_refs_global: list[str] = []
    all_hyps_global: list[str] = []

    # We'll collect (idx, hyp, ref) to reconstruct global lists in order
    idx_to_hyp: dict[int, str] = {}
    idx_to_ref: dict[int, str] = {}

    for condition in CLASSES:
        indices = condition_to_indices.get(condition, [])
        if not indices:
            print(f"\n  [{condition}] no samples routed here — skipping")
            continue

        hyps, refs = transcribe_group(
            condition=condition,
            sample_indices=indices,
            dataset=dataset,
            whisper_model_id=args.whisper_model,
            adapters_dir=args.adapters_dir,
            device=device,
            processor=processor,
            batch_size=args.batch,
        )

        for sample_idx, hyp, ref in zip(indices, hyps, refs):
            idx_to_hyp[sample_idx] = hyp
            idx_to_ref[sample_idx] = ref

        # Per-condition WER
        cond_wer = compute_wer(refs, hyps)

        # Routing accuracy for this condition's GT samples
        route_acc: float | None = None
        if has_gt:
            gt_for_group = [gt_conditions[i] for i in indices]
            correct = sum(g == condition for g in gt_for_group if g in CLASS_TO_IDX)
            total = sum(1 for g in gt_for_group if g in CLASS_TO_IDX)
            route_acc = correct / total if total > 0 else None

        per_condition[condition] = {
            "wer": cond_wer,
            "n_samples": len(indices),
            "route_acc": route_acc,
        }

    # Rebuild global lists in original sample order
    for i in range(len(dataset)):
        if i in idx_to_hyp:
            all_hyps_global.append(idx_to_hyp[i])
            all_refs_global.append(idx_to_ref[i])

    print_wer_table(per_condition, all_refs_global, all_hyps_global, routing_accuracy)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
