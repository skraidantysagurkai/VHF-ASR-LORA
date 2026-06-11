from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchaudio
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from class_models import CNNClassifier, CRNNClassifier, ECAPAClassifier, SAMPLE_RATE

CLASSES = ["perfect", "good", "okay", "bad"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


class AudioDataset(Dataset):
    def __init__(self, manifest_path: Path, clip_seconds: float = 5.0) -> None:
        self.root = manifest_path.parent
        self.clip_len = int(clip_seconds * SAMPLE_RATE)

        self.samples: list[dict[str, Any]] = []
        with manifest_path.open() as f:
            for line in f:
                row = json.loads(line)
                cond = row.get("condition") or row.get("label")
                if cond not in CLASS_TO_IDX:
                    continue
                self.samples.append({"path": row["audio_filepath"], "label": cond})

        if not self.samples:
            raise ValueError(f"No usable samples in {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.samples[idx]
        path = self.root / item["path"]
        wav, sr = torchaudio.load(str(path))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)
        if wav.shape[0] < self.clip_len:
            wav = torch.nn.functional.pad(wav, (0, self.clip_len - wav.shape[0]))
        else:
            start = (wav.shape[0] - self.clip_len) // 2
            wav = wav[start : start + self.clip_len]
        return wav, CLASS_TO_IDX[item["label"]]


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for wavs, labels in tqdm(
            loader, desc="  Evaluating", leave=False, dynamic_ncols=True
        ):
            wavs = wavs.to(device, non_blocking=True)
            logits, _, _ = model(wavs)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.tolist())
    return all_preds, all_labels


def save_confusion(
    preds: list[int], labels: list[int], name: str, plot_dir: Path
) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False)
    ax.set_title(f"{name} — test confusion matrix")
    fig.tight_layout()
    path = plot_dir / f"{name}_test_confusion.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved {path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing manifest.jsonl to evaluate on",
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("../checkpoints"),
        help="Directory with <model>_best.pt files (default: checkpoints)",
    )
    p.add_argument("--model", choices=["cnn", "crnn", "ecapa", "all"], default="all")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--clip-sec", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--plot-dir", type=Path, default=Path("../plots"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = args.test_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    args.plot_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(CLASSES)
    ds = AudioDataset(manifest, clip_seconds=args.clip_sec)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Test samples: {len(ds):,}")

    model_map = {
        "cnn": CNNClassifier(num_classes=num_classes),
        "crnn": CRNNClassifier(num_classes=num_classes),
        "ecapa": ECAPAClassifier(num_classes=num_classes),
    }
    selected = list(model_map.keys()) if args.model == "all" else [args.model]

    for name in selected:
        ckpt_path = args.ckpt_dir / f"{name}_best.pt"
        if not ckpt_path.exists():
            print(f"\n[SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue

        model = model_map[name].to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])

        print(f"\n{'=' * 60}")
        print(f"  {name}  (epoch {ckpt.get('epoch', '?')})")
        print(f"{'=' * 60}")

        preds, labels = evaluate(model, loader, device)

        acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Macro-F1:  {macro_f1:.4f}")
        print()
        print(
            classification_report(labels, preds, target_names=CLASSES, zero_division=0)
        )

        save_confusion(preds, labels, name, args.plot_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
