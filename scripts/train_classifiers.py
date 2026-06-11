from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange

from class_models import (
    CNNClassifier,
    CRNNClassifier,
    ClassifierLoss,
    ECAPAClassifier,
    SAMPLE_RATE,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Canonical ordering used everywhere in this script
CLASSES = ["perfect", "good", "okay", "bad"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


class AudioDataset(Dataset):
    """Reads a manifest.jsonl and returns (waveform, label) pairs."""

    def __init__(
        self,
        manifest_path: Path,
        clip_seconds: float = 5.0,
        augment: bool = False,
    ) -> None:
        self.root = manifest_path.parent
        self.clip_len = int(clip_seconds * SAMPLE_RATE)
        self.augment = augment

        self.samples: list[dict[str, Any]] = []
        with manifest_path.open() as f:
            for line in f:
                row = json.loads(line)
                cond = row.get("condition") or row.get("label")
                if cond not in CLASS_TO_IDX:
                    continue
                self.samples.append({"path": row["audio_filepath"], "label": cond})

        if not self.samples:
            raise ValueError(f"No usable samples found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, rel_path: str) -> torch.Tensor:
        path = self.root / rel_path
        wav, sr = torchaudio.load(str(path))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)  # mono
        # pad / crop to clip_len
        if wav.shape[0] < self.clip_len:
            pad = self.clip_len - wav.shape[0]
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            # random crop during training, centre crop during eval
            if self.augment:
                start = torch.randint(0, wav.shape[0] - self.clip_len + 1, ()).item()
            else:
                start = (wav.shape[0] - self.clip_len) // 2
            wav = wav[start : start + self.clip_len]
        return wav

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.samples[idx]
        wav = self._load(item["path"])
        label = CLASS_TO_IDX[item["label"]]
        return wav, label


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: ClassifierLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
    desc: str = "",
    iter_offset: int = 0,
) -> dict[str, Any]:
    """One train or eval pass. Returns dict of metrics."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = total_ce = total_gl = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_correct = 0
    n_seen = 0
    iter_log: list[tuple[int, float, float]] = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        for batch_idx, (wavs, labels) in enumerate(pbar):
            wavs = wavs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, probs, gate = model(wavs)
            loss, ce, gl = loss_fn(logits, labels, gate)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            preds = logits.argmax(-1)
            batch_correct = (preds == labels).sum().item()
            n_correct += batch_correct
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            batch_n = len(labels)
            total_loss += loss.item() * batch_n
            total_ce += ce.item() * batch_n
            total_gl += gl.item() * batch_n
            n_seen += batch_n
            pbar.set_postfix(
                loss=f"{total_loss / n_seen:.4f}", acc=f"{n_correct / n_seen:.3f}"
            )

            if training and (batch_idx + 1) % 10 == 0:
                iter_log.append(
                    (
                        iter_offset + batch_idx + 1,
                        loss.item(),
                        batch_correct / batch_n,
                    )
                )

    n = len(loader.dataset)  # type: ignore[arg-type]
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "gate_loss": total_gl / n,
        "accuracy": n_correct / n,
        "f1": float(macro_f1),
        "iter_log": iter_log,
    }


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    epochs: int,
    lr: float,
    ckpt_dir: Path,
) -> dict[str, list[float]]:
    """Train one model; return history dict."""
    model = model.to(device)
    loss_fn = ClassifierLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
        "train_iter_loss": [],
        "train_iter_acc": [],
    }
    best_val_loss = float("inf")
    ckpt_path = ckpt_dir / f"{name}_best.pt"
    global_step = 0

    print(f"\n{'=' * 60}")
    print(
        f"  Training {name}  ({sum(p.numel() for p in model.parameters() if p.requires_grad):,} params)"
    )
    print(f"{'=' * 60}")
    epoch_bar = trange(1, epochs + 1, desc=name, dynamic_ncols=True)
    for epoch in epoch_bar:
        t0 = time.time()
        tr = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            num_classes,
            desc=f"Epoch {epoch:>3} train",
            iter_offset=global_step,
        )
        global_step += len(train_loader)
        for step, loss_v, acc_v in tr["iter_log"]:
            history["train_iter_loss"].append((step, loss_v))
            history["train_iter_acc"].append((step, acc_v))
        vl = run_epoch(
            model,
            val_loader,
            loss_fn,
            None,
            device,
            num_classes,
            desc=f"Epoch {epoch:>3} val",
        )
        scheduler.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(vl["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["val_acc"].append(vl["accuracy"])
        history["val_f1"].append(vl["f1"])

        if vl["loss"] < best_val_loss:
            best_val_loss = vl["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": best_val_loss,
                },
                ckpt_path,
            )
            star = " *"
        else:
            star = ""

        elapsed = time.time() - t0
        row = (
            f"[{epoch:>3}/{epochs}]"
            f"  loss={tr['loss']:.4f}  acc={tr['accuracy']:.3f}"
            f"  | val loss={vl['loss']:.4f}  acc={vl['accuracy']:.3f}  F1={vl['f1']:.3f}"
            f"  {elapsed:.0f}s{star}"
        )
        epoch_bar.write(row)

    # confusion matrix on val set (best checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for wavs, labels in val_loader:
            wavs = wavs.to(device)
            logits, _, _ = model(wavs)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.tolist())

    history["val_preds"] = all_preds  # type: ignore[assignment]
    history["val_labels"] = all_labels  # type: ignore[assignment]
    return history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def save_curves(histories: dict[str, dict], plot_dir: Path) -> None:
    model_names = list(histories.keys())
    epochs = list(range(1, len(next(iter(histories.values()))["train_loss"]) + 1))

    def _plot(
        key_pairs: list[tuple[str, str]], title: str, ylabel: str, fname: str
    ) -> None:
        fig, axes = plt.subplots(
            1, len(model_names), figsize=(6 * len(model_names), 4), squeeze=False
        )
        for ax, mname in zip(axes[0], model_names):
            hist = histories[mname]
            for key, label in key_pairs:
                if key in hist:
                    ax.plot(epochs, hist[key], label=label)
            ax.set_title(mname)
            ax.set_xlabel("Epocha")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(alpha=0.3)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(plot_dir / fname, dpi=120)
        plt.close(fig)

    _plot(
        [("train_loss", "Mokymo aibė"), ("val_loss", "Validacijos aibė")],
        "Nuostolis",
        "Nuostolis",
        "loss_curves.png",
    )
    _plot(
        [("train_acc", "Mokymo aibė"), ("val_acc", "Validacijos aibė")],
        "Tikslumas",
        "Tikslumas",
        "accuracy_curves.png",
    )
    _plot(
        [("val_f1", "Validacijos aibės makro-F1")],
        "Validacijos aibės makro-F1",
        "F1",
        "f1_curves.png",
    )

    # confusion matrices
    for mname, hist in histories.items():
        if "val_preds" not in hist:
            continue
        cm = confusion_matrix(hist["val_labels"], hist["val_preds"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"{mname} — Klasifikavimo lentelė (validacijos aibė)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{mname}_confusion.png", dpi=120)
        plt.close(fig)

    def _ema_smooth(values: list[float], weight: float = 0.6) -> list[float]:
        smoothed, last = [], values[0]
        for v in values:
            last = last * weight + v * (1 - weight)
            smoothed.append(last)
        return smoothed

    for iter_key, ylabel, title, fname in [
        (
            "train_iter_loss",
            "Loss",
            "Train loss per 10 iterations",
            "iter_loss_curves.png",
        ),
        (
            "train_iter_acc",
            "Accuracy",
            "Train accuracy per 10 iterations",
            "iter_acc_curves.png",
        ),
    ]:
        if not any(histories[m].get(iter_key) for m in model_names):
            continue
        fig2, axes2 = plt.subplots(
            1, len(model_names), figsize=(6 * len(model_names), 4), squeeze=False
        )
        for ax2, mname2 in zip(axes2[0], model_names):
            data = histories[mname2].get(iter_key, [])
            if not data:
                continue
            steps = [s for s, _ in data]
            values = [v for _, v in data]
            smooth = _ema_smooth(values, weight=0.6)
            ax2.plot(steps, values, alpha=0.25, color="steelblue", label="raw")
            ax2.plot(steps, smooth, color="steelblue", label="EMA α=0.6")
            ax2.set_title(mname2)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel(ylabel)
            ax2.legend()
            ax2.grid(alpha=0.3)
        fig2.suptitle(title)
        fig2.tight_layout()
        fig2.savefig(plot_dir / fname, dpi=120)
        plt.close(fig2)

    print(f"\nPlots saved to {plot_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train VHF quality classifiers")
    p.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/LibriSpeech-train-sim-40h"),
        help="Directory containing manifest.jsonl for training",
    )
    p.add_argument(
        "--val-dir",
        type=Path,
        default=Path("../data/LibriSpeech-dev-clean"),
        help="Directory containing manifest.jsonl for validation",
    )
    p.add_argument("--model", choices=["cnn", "crnn", "ecapa", "all"], default="all")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--clip-sec",
        type=float,
        default=5.0,
        help="Audio clip length in seconds (shorter files are zero-padded)",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--ckpt-dir", type=Path, default=Path("../checkpoints"))
    p.add_argument("--plot-dir", type=Path, default=Path("../plots"))
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable random crop augmentation during training",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # resolve manifest paths
    train_manifest = args.train_dir / "manifest.jsonl"
    val_manifest = args.val_dir / "manifest.jsonl"
    for p in (train_manifest, val_manifest):
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {p}")

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(CLASSES)  # 4: perfect, good, okay, bad

    train_ds = AudioDataset(
        train_manifest, clip_seconds=args.clip_sec, augment=not args.no_augment
    )
    val_ds = AudioDataset(val_manifest, clip_seconds=args.clip_sec, augment=False)
    print(f"Train samples: {len(train_ds):,}   Val samples: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model_map = {
        "cnn": CNNClassifier(num_classes=num_classes),
        "crnn": CRNNClassifier(num_classes=num_classes),
        "ecapa": ECAPAClassifier(num_classes=num_classes),
    }
    selected = list(model_map.keys()) if args.model == "all" else [args.model]

    histories: dict[str, dict] = {}
    for name in selected:
        histories[name] = train_model(
            name=name,
            model=model_map[name],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
            ckpt_dir=args.ckpt_dir,
        )

    save_curves(histories, args.plot_dir)

    pkl_path = args.plot_dir / "histories.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(histories, f)
    print(f"Histories saved to {pkl_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
