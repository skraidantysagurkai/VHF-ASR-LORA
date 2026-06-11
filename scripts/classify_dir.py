"""
Classify all audio files in a directory — no manifest required.

Active-audio extraction (per file)
-----------------------------------
1. Split waveform into 10 ms frames.
2. Keep frames whose RMS exceeds --threshold-db dBFS (default -20 dBFS).
3. Concatenate kept frames; fall back to full audio if none pass.
4. Clip / zero-pad to --clip-sec seconds (default 5.0) and feed to model.

Output
------
  Per-file table  : printed when --per-file is set
  Summary table   : always printed (class counts + % per model)

Usage
-----
    python classify_dir.py \\
        --audio-dir  data/liveatc_chunks \\
        --ckpt-dir   ../checkpoints \\
        --model      all \\
        [--threshold-db -20] \\
        [--clip-sec   5.0] \\
        [--batch 32] [--workers 4] \\
        [--ext wav,flac,mp3] \\
        [--per-file]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from class_models import CNNClassifier, CRNNClassifier, ECAPAClassifier, SAMPLE_RATE

CLASSES = ["perfect", "good", "okay", "bad"]

FRAME_SIZE = 160  # 10 ms at 16 kHz


# ---------------------------------------------------------------------------
# Active-audio extraction
# ---------------------------------------------------------------------------


def active_clip(wav: torch.Tensor, threshold_db: float, clip_len: int) -> torch.Tensor:
    """Return a clip_len sample tensor of concatenated above-threshold frames."""
    thresh_amp = 10.0 ** (threshold_db / 20.0)

    # Trim tail so unfold works cleanly
    n_frames = wav.shape[0] // FRAME_SIZE
    trimmed = wav[: n_frames * FRAME_SIZE]
    frames = trimmed.unfold(0, FRAME_SIZE, FRAME_SIZE)  # (N, FRAME_SIZE)
    rms = frames.pow(2).mean(-1).sqrt()  # (N,)
    mask = rms > thresh_amp
    active = frames[mask].reshape(-1) if mask.any() else trimmed

    if active.shape[0] < clip_len:
        active = torch.nn.functional.pad(active, (0, clip_len - active.shape[0]))
    else:
        active = active[:clip_len]

    return active


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AudioDirDataset(Dataset):
    def __init__(
        self,
        audio_dir: Path,
        extensions: list[str],
        threshold_db: float,
        clip_seconds: float,
    ) -> None:
        self.clip_len = int(clip_seconds * SAMPLE_RATE)
        self.threshold_db = threshold_db

        self.files: list[Path] = []
        for ext in extensions:
            self.files.extend(sorted(audio_dir.rglob(f"*.{ext.lstrip('.')}")))

        if not self.files:
            raise ValueError(
                f"No audio files found in {audio_dir} with extensions {extensions}"
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.files[idx]
        wav, sr = torchaudio.load(str(path))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)
        clip = active_clip(wav, self.threshold_db, self.clip_len)
        return clip, path.name


def collate(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, list[str]]:
    wavs, names = zip(*batch)
    return torch.stack(wavs), list(names)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> list[str]:
    model.eval()
    preds: list[str] = []
    for wavs, _ in tqdm(loader, desc="  classifying", leave=False, dynamic_ncols=True):
        wavs = wavs.to(device, non_blocking=True)
        logits, _, _ = model(wavs)
        for idx in logits.argmax(-1).cpu().tolist():
            preds.append(CLASSES[idx])
    return preds


# ---------------------------------------------------------------------------
# Table printers
# ---------------------------------------------------------------------------


def print_per_file(names: list[str], results: dict[str, list[str]]) -> None:
    models = list(results.keys())
    col0 = max(len(n) for n in names)
    col0 = max(col0, 8)
    col_w = max(len(m) for m in models)
    col_w = max(col_w, 8)

    header = f"{'File':<{col0}}" + "".join(f"  {m:>{col_w}}" for m in models)
    sep = "-" * len(header)
    print("\n" + "=" * len(header))
    print("  Per-file predictions")
    print("=" * len(header))
    print(header)
    print(sep)
    for i, name in enumerate(names):
        row = f"{name:<{col0}}" + "".join(f"  {results[m][i]:>{col_w}}" for m in models)
        print(row)
    print("=" * len(header))


def print_summary(results: dict[str, list[str]]) -> None:
    models = list(results.keys())
    col0 = max(len(c) for c in CLASSES)
    col0 = max(col0, 9)
    col_w = max(len(m) for m in models)
    col_w = max(col_w, 10)

    # header: model names, each split into count + pct sub-columns
    print("\n" + "=" * (col0 + len(models) * (col_w * 2 + 5)))
    print("  Class distribution")
    print("=" * (col0 + len(models) * (col_w * 2 + 5)))

    header = f"{'Class':<{col0}}"
    for m in models:
        header += f"  {m + ' n':>{col_w}}  {m + ' %':>{col_w}}"
    print(header)
    print("-" * len(header))

    n_total = len(next(iter(results.values())))
    for cls in CLASSES:
        row = f"{cls:<{col0}}"
        for m in models:
            cnt = results[m].count(cls)
            pct = cnt / n_total * 100 if n_total else 0.0
            row += f"  {cnt:>{col_w}}  {pct:>{col_w - 1}.1f}%"
        print(row)

    print("-" * len(header))
    row = f"{'TOTAL':<{col0}}"
    for m in models:
        row += f"  {n_total:>{col_w}}  {'100.0%':>{col_w}}"
    print(row)
    print("=" * len(header))
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
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory to scan for audio files (recursive)",
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("../checkpoints"),
        help="Directory with <model>_best.pt files (default: ../checkpoints)",
    )
    p.add_argument("--model", choices=["cnn", "crnn", "ecapa", "all"], default="all")
    p.add_argument(
        "--threshold-db",
        type=float,
        default=-20.0,
        help="dBFS RMS threshold for active-audio extraction (default: -20)",
    )
    p.add_argument(
        "--clip-sec",
        type=float,
        default=5.0,
        help="Target clip length in seconds (default: 5.0)",
    )
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--ext",
        default="wav,flac,mp3",
        help="Comma-separated audio extensions to scan (default: wav,flac,mp3)",
    )
    p.add_argument(
        "--per-file", action="store_true", help="Also print a per-file prediction table"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extensions = [e.strip() for e in args.ext.split(",")]

    print(f"Device:       {device}")
    print(f"Audio dir:    {args.audio_dir}")
    print(f"Threshold:    {args.threshold_db} dBFS")
    print(f"Clip length:  {args.clip_sec} s")

    ds = AudioDirDataset(
        args.audio_dir,
        extensions=extensions,
        threshold_db=args.threshold_db,
        clip_seconds=args.clip_sec,
    )
    print(f"Files found:  {len(ds):,}")

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    num_classes = len(CLASSES)
    model_map = {
        "cnn": CNNClassifier(num_classes=num_classes),
        "crnn": CRNNClassifier(num_classes=num_classes),
        "ecapa": ECAPAClassifier(num_classes=num_classes),
    }
    selected = list(model_map.keys()) if args.model == "all" else [args.model]

    # Collect names once
    names: list[str] = [p.name for p in ds.files]
    results: dict[str, list[str]] = {}

    for name in selected:
        ckpt_path = args.ckpt_dir / f"{name}_best.pt"
        if not ckpt_path.exists():
            print(f"\n[SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue

        model = model_map[name].to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"\n{name}  (epoch {ckpt.get('epoch', '?')})")

        results[name] = run_model(model, loader, device)

    if not results:
        print("No models loaded — nothing to show.")
        return

    if args.per_file:
        print_per_file(names, results)

    print_summary(results)


if __name__ == "__main__":
    main()
