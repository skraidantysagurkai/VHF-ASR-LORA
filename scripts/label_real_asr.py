from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.class_models import (
    CNNClassifier,
    CRNNClassifier,
    ECAPAClassifier,
    SAMPLE_RATE,
)

CLASSES = ["perfect", "good", "okay", "bad"]



class ManifestDataset(Dataset):
    """Loads raw waveforms from a manifest; preserves the full row for output."""

    def __init__(self, manifest_path: Path, clip_seconds: float = 5.0) -> None:
        self.root = manifest_path.parent
        self.clip_len = int(clip_seconds * SAMPLE_RATE)
        self.rows: list[dict[str, Any]] = []

        with manifest_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

        if not self.rows:
            raise ValueError(f"No samples found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path = self.root / self.rows[idx]["audio_filepath"]
        wav, sr = torchaudio.load(str(path))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)
        if wav.shape[0] < self.clip_len:
            wav = torch.nn.functional.pad(wav, (0, self.clip_len - wav.shape[0]))
        else:
            start = (wav.shape[0] - self.clip_len) // 2
            wav = wav[start : start + self.clip_len]
        return wav, idx  # idx lets us map predictions back to rows safely


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[int]:
    """Returns a list of predicted class indices in original sample order."""
    model.eval()
    preds: list[tuple[int, int]] = []
    with torch.no_grad():
        for wavs, idxs in tqdm(
            loader, desc="    inference", leave=False, dynamic_ncols=True
        ):
            wavs = wavs.to(device, non_blocking=True)
            logits, _, _ = model(wavs)
            class_preds = logits.argmax(-1).cpu().tolist()
            preds.extend(zip(idxs.tolist(), class_preds))
    preds.sort(key=lambda x: x[0])
    return [p for _, p in preds]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--splits",
        type=Path,
        nargs="+",
        required=True,
        help="One or more split directories each containing a manifest.jsonl",
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing <model>_best.pt files (default: checkpoints)",
    )
    p.add_argument(
        "--model",
        choices=["cnn", "crnn", "ecapa", "all"],
        default="all",
        help="Which classifier(s) to run (default: all)",
    )
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--clip-sec", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=4)
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    num_classes = len(CLASSES)
    all_model_defs = {
        "cnn": CNNClassifier(num_classes=num_classes),
        "crnn": CRNNClassifier(num_classes=num_classes),
        "ecapa": ECAPAClassifier(num_classes=num_classes),
    }
    selected_names = (
        list(all_model_defs.keys()) if args.model == "all" else [args.model]
    )

    loaded_models: dict[str, torch.nn.Module] = {}
    for name in selected_names:
        ckpt_path = args.ckpt_dir / f"{name}_best.pt"
        if not ckpt_path.exists():
            print(f"[SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue
        model = all_model_defs[name].to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        loaded_models[name] = model
        print(f"Loaded {name} (epoch {ckpt.get('epoch', '?')})")

    if not loaded_models:
        raise RuntimeError("No models loaded — check --ckpt-dir and --model.")

    print()

    for split_dir in args.splits:
        manifest_path = split_dir / "manifest.jsonl"
        if not manifest_path.exists():
            print(f"[SKIP] {split_dir}: manifest.jsonl not found")
            continue

        print(f"{'=' * 60}")
        print(f"Split: {split_dir}")

        ds = ManifestDataset(manifest_path, clip_seconds=args.clip_sec)
        loader = DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        print(f"  Samples: {len(ds):,}")

        for name, model in loaded_models.items():
            print(f"  Running {name}...")
            preds = run_inference(model, loader, device)

            out_rows = []
            for row, pred_idx in zip(ds.rows, preds):
                new_row = dict(row)
                new_row["condition"] = CLASSES[pred_idx]
                out_rows.append(new_row)

            out_path = split_dir / f"manifest_{name}.jsonl"
            with out_path.open("w") as f:
                for row in out_rows:
                    f.write(json.dumps(row) + "\n")

            label_counts = Counter(r["condition"] for r in out_rows)
            print(
                f"    -> {out_path.name}  "
                + "  ".join(f"{cls}: {label_counts.get(cls, 0)}" for cls in CLASSES)
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
