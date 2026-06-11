
import argparse
import json
import logging
import multiprocessing
import os
import pickle
import random
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import librosa
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)



SR = 16_000
FMIN = 300
FMAX = 3_400
N_MELS = 64
HOP = 512
N_FFT = 2048

COND_COLORS = {
    "clean": "#E040FB",  # vivid violet
    "perfect": "#00E676",  # neon green
    "good": "#2979FF",  # electric blue
    "okay": "#FFEA00",  # sharp yellow
    "bad": "#FF1744",  # alarm red
    "atc_1": "#FF6D00",  # burnt orange
    "atc_2": "#00E5FF",  # cyan
}

# Ordered for plots
ALL_CONDITIONS = ["clean", "perfect", "good", "okay", "bad", "atc_1", "atc_2"]


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Acoustic feature analysis across ATC simulation conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument(
        "--data-dir", type=Path, required=True, help="Root data directory (DATA_DIR)"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_plots"),
        help="Directory for output PNG plots",
    )
    p.add_argument(
        "--cache-path",
        type=Path,
        default=Path("features_cache.pkl"),
        help="Pickle cache for extracted features",
    )

    # Sampling
    p.add_argument(
        "--subset",
        type=int,
        default=100,
        help="Utterances to sample from LibriSpeech + atc_1",
    )
    p.add_argument(
        "--subset-proc-atc",
        type=int,
        default=100,
        help="Clips to sample from proc_unlab_atc_clips",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for all sampling")

    # DSP
    p.add_argument(
        "--max-audio-s",
        type=float,
        default=2.0,
        help="Truncate clips to this many seconds before feature extraction",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Parallel worker processes",
    )

    # Misc
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and delete any existing feature cache",
    )
    p.add_argument("--dpi", type=int, default=150, help="DPI for saved plots")

    return p.parse_args()


# ── Manifest loading ──────────────────────────────────────────────────────────


def load_librispeech_manifest(
    jsonl_path: Path,
    data_root: Path,
    librispeech_root: Path,
    subset: int | None = None,
    seed: int = 42,
) -> dict[str, list[Path]]:
    sim_conds = {"perfect", "good", "okay", "bad"}
    utt_to_paths: dict[str, dict[str, Path]] = {}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cond, utt = entry["condition"], entry["utterance_id"]
            if cond not in sim_conds:
                continue
            utt_to_paths.setdefault(utt, {})[cond] = data_root / entry["audio_filepath"]

    complete_utts = [u for u, c in utt_to_paths.items() if sim_conds.issubset(c)]
    log.info(
        "LibriSpeech: %d utterances with all 4 simulated conditions", len(complete_utts)
    )

    if subset is not None:
        rng = random.Random(seed)
        complete_utts = rng.sample(complete_utts, min(subset, len(complete_utts)))

    valid_utts, missing = [], 0
    for utt in complete_utts:
        speaker, chapter = utt.split("-")[:2]
        flac = librispeech_root / speaker / chapter / f"{utt}.flac"
        if flac.exists():
            valid_utts.append(utt)
        else:
            missing += 1

    if missing:
        log.warning("%d clean FLACs not found — dropped", missing)

    manifest: dict[str, list[Path]] = {c: [] for c in ["clean"] + sorted(sim_conds)}
    for utt in valid_utts:
        speaker, chapter = utt.split("-")[:2]
        manifest["clean"].append(librispeech_root / speaker / chapter / f"{utt}.flac")
        for cond in sim_conds:
            manifest[cond].append(utt_to_paths[utt][cond])

    return manifest


def load_atc_manifest(
    jsonl_path: Path,
    data_root: Path,
    subset: int | None = None,
    seed: int = 42,
) -> list[Path]:
    paths = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            p = data_root / entry["audio_filepath"]
            if p.exists():
                paths.append(p)

    if subset is not None:
        rng = random.Random(seed)
        paths = rng.sample(paths, min(subset, len(paths)))

    return paths


def load_proc_unlab_atc_manifest(
    data_root: Path,
    subset: int | None = None,
    seed: int = 42,
    extensions: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg"),
) -> list[Path]:
    paths = [p for p in data_root.rglob("*") if p.suffix.lower() in extensions]

    if not paths:
        jsonl = data_root / "manifest.jsonl"
        if jsonl.exists():
            log.info("No loose audio files found; falling back to manifest.jsonl")
            return load_atc_manifest(jsonl, data_root, subset=subset, seed=seed)
        log.warning("No audio files found in %s", data_root)
        return []

    if subset is not None:
        rng = random.Random(seed)
        paths = rng.sample(paths, min(subset, len(paths)))

    return paths


# ── Feature extraction ────────────────────────────────────────────────────────

# Module-level so ProcessPoolExecutor can pickle it
_MAX_AUDIO_S: float = 10.0


def _init_worker(max_audio_s: float) -> None:
    """Initialise per-process globals."""
    global _MAX_AUDIO_S
    _MAX_AUDIO_S = max_audio_s


def _extract_features_worker(args: tuple) -> tuple:
    path_str, cond = args
    try:
        result = _extract(Path(path_str))
        return (path_str, cond, result, None)
    except Exception as e:
        return (path_str, cond, None, str(e))


def _extract(path: Path) -> dict:
    audio, _ = librosa.load(path, sr=SR, mono=True, duration=_MAX_AUDIO_S)

    D = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    mel_fb = librosa.filters.mel(
        sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel_fb @ D, ref=1.0)

    rms = librosa.feature.rms(y=audio, hop_length=HOP)[0]
    norm = D.sum(axis=0, keepdims=True) + 1e-9

    centroid = (freqs[:, None] * D).sum(axis=0) / norm[0]
    deviation = np.abs(freqs[:, None] - centroid[None, :])
    bandwidth = (deviation * D).sum(axis=0) / norm[0]
    cumulative = np.cumsum(D, axis=0) / norm
    rolloff = freqs[np.argmax(cumulative >= 0.85, axis=0)]

    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP)[0]

    f0, voiced_flag, _ = librosa.pyin(
        audio,
        sr=SR,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        hop_length=HOP,
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]

    return {
        "mel_mean": mel_db.mean(axis=1),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "dyn_range": float(20 * np.log10((rms.max() + 1e-9) / (rms.min() + 1e-9))),
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "zcr": zcr,
        "f0": f0_voiced,
    }


def collect_dataset(
    manifest: dict[str, list[Path]],
    conditions: list[str],
    cache_path: Path | None,
    workers: int,
    max_audio_s: float,
) -> tuple[dict, np.ndarray, list[str]]:
    if cache_path and cache_path.exists():
        log.info("Loading cached features from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    tasks = [
        (str(path), cond) for cond in conditions for path in manifest.get(cond, [])
    ]

    log.info(
        "Tasks: %d  |  workers: %d  |  max clip: %.1fs",
        len(tasks),
        workers,
        max_audio_s,
    )

    results: dict = {}
    errors: list = []

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(max_audio_s,),
    ) as pool:
        futures = {pool.submit(_extract_features_worker, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Extracting"):
            path_str, cond, feats, err = future.result()
            if err:
                errors.append((Path(path_str).name, err))
            else:
                results[(path_str, cond)] = feats

    if errors:
        log.warning("%d files failed:", len(errors))
        for name, err in errors[:10]:
            log.warning("  %s: %s", name, err)

    per_cond: dict[str, list[dict]] = {c: [] for c in conditions}
    mel_rows, labels = [], []
    for path_str, cond in tasks:
        if (path_str, cond) in results:
            feats = results[(path_str, cond)]
            per_cond[cond].append(feats)
            mel_rows.append(feats["mel_mean"])
            labels.append(cond)

    log.info("Successful: %d / %d", len(mel_rows), len(tasks))
    for cond in conditions:
        log.info("  %-12s %d", cond, len(per_cond[cond]))

    if not mel_rows:
        raise RuntimeError("No features extracted — check errors above")

    out = (per_cond, np.vstack(mel_rows), labels)
    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
        log.info("Cached → %s", cache_path)

    return out


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_pca_tsne(mel_matrix, labels, conditions, out, dpi):
    X = StandardScaler().fit_transform(mel_matrix)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    X_tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(labels) // 2),
        random_state=42,
        max_iter=1000,
        init="pca",
    ).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Condition Separability — Mel Feature Space", fontsize=13, fontweight="bold"
    )

    labels_arr = np.array(labels)
    for ax, X_emb, title, (xl, yl) in zip(
        axes,
        [X_pca, X_tsne],
        ["PCA", "t-SNE"],
        [
            (
                f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            ),
            ("t-SNE dim 1", "t-SNE dim 2"),
        ],
    ):
        for cond in conditions:
            idx = np.where(labels_arr == cond)[0]
            ax.scatter(
                X_emb[idx, 0],
                X_emb[idx, 1],
                c=COND_COLORS[cond],
                label=cond.replace("_", " ").capitalize(),
                alpha=0.65,
                s=40,
                edgecolors="none",
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_rms_dynrange(per_cond, conditions, out, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "RMS Energy & Dynamic Range per Condition", fontsize=13, fontweight="bold"
    )

    for ax, key, title in zip(
        axes,
        ["rms_mean", "rms_std", "dyn_range"],
        ["Mean RMS", "RMS Std Dev\n(fading proxy)", "Dynamic Range (dB)"],
    ):
        data = [np.array([f[key] for f in per_cond[c]]) for c in conditions]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "white", "linewidth": 2},
        )
        for patch, color in zip(bp["boxes"], [COND_COLORS[c] for c in conditions]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for el in ["whiskers", "caps", "fliers"]:
            for item in bp[el]:
                item.set_color("#555")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(range(1, len(conditions) + 1))
        ax.set_xticklabels(
            [c.replace("_", " ").capitalize() for c in conditions],
            rotation=15,
            ha="right",
        )
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_spectral(per_cond, conditions, out, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "Spectral Feature Distributions per Condition", fontsize=13, fontweight="bold"
    )

    for ax, key, title in zip(
        axes,
        ["centroid", "bandwidth", "rolloff"],
        [
            "Spectral Centroid (Hz)",
            "Spectral Bandwidth (Hz)",
            "Spectral Rolloff 85% (Hz)",
        ],
    ):
        for cond in conditions:
            vals = np.concatenate([f[key] for f in per_cond[cond]])
            vals = vals[(vals > 0) & np.isfinite(vals)]
            ax.hist(
                vals,
                bins=60,
                alpha=0.55,
                color=COND_COLORS[cond],
                label=cond.replace("_", " ").capitalize(),
                density=True,
            )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Hz")
        ax.set_ylabel("Density")
        ax.legend(framealpha=0.8)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_f0(per_cond, conditions, out, dpi):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        "F0 Distribution per Condition (pYIN, voiced frames only)",
        fontsize=13,
        fontweight="bold",
    )

    data = [
        np.concatenate([f["f0"] for f in per_cond[c] if len(f["f0"]) > 0])
        for c in conditions
    ]
    bp = ax.violinplot(data, positions=range(len(conditions)), showmedians=True)
    for body, color in zip(bp["bodies"], [COND_COLORS[c] for c in conditions]):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    bp["cmedians"].set_color("white")
    bp["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(
        [c.replace("_", " ").capitalize() for c in conditions], rotation=15, ha="right"
    )
    ax.set_ylabel("F0 (Hz)")
    ax.set_ylim(50, 400)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(
        0.98,
        0.97,
        "F0 should be stable across simulated conditions\n"
        "Real ATC may differ due to radio characteristics",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_zcr(per_cond, conditions, out, dpi):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        "Zero-Crossing Rate per Condition (noisiness proxy)",
        fontsize=13,
        fontweight="bold",
    )

    data = [np.concatenate([f["zcr"] for f in per_cond[c]]) for c in conditions]
    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.5,
        notch=True,
        medianprops={"color": "white", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], [COND_COLORS[c] for c in conditions]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for el in ["whiskers", "caps"]:
        for item in bp[el]:
            item.set_color("#555")

    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(
        [c.replace("_", " ").capitalize() for c in conditions], rotation=15, ha="right"
    )
    ax.set_ylabel("Zero-Crossing Rate")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    manifest, conditions, output_dir, cache_path, workers, max_audio_s, dpi
):
    output_dir.mkdir(parents=True, exist_ok=True)
    per_cond, mel_matrix, labels = collect_dataset(
        manifest, conditions, cache_path, workers, max_audio_s
    )
    log.info("%d utterances × %d mel features", len(labels), mel_matrix.shape[1])

    plots = [
        ("01_pca_tsne.png", plot_pca_tsne, (mel_matrix, labels, conditions)),
        ("02_rms_dynrange.png", plot_rms_dynrange, (per_cond, conditions)),
        ("03_spectral.png", plot_spectral, (per_cond, conditions)),
        ("04_f0.png", plot_f0, (per_cond, conditions)),
        ("05_zcr.png", plot_zcr, (per_cond, conditions)),
    ]
    for fname, fn, args in plots:
        out = output_dir / fname
        fn(*args, out, dpi)
        log.info("Saved %s", out)

    log.info("All plots saved to %s/", output_dir)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir
    data_root = data_dir / "LibriSpeech-dev-clean-sint"
    manifest_path = data_root / "manifest.jsonl"
    librispeech_root = data_dir / "LibriSpeech-dev-clean-bpf/dev-clean"
    atc_root = data_dir / "atc_asr_test"
    atc_manifest = atc_root / "manifest.jsonl"
    proc_unlab_root = data_dir / "proc_unlab_atc_clips"

    if args.no_cache and args.cache_path.exists():
        os.remove(args.cache_path)
        log.info("Cache cleared: %s", args.cache_path)

    log.info("Loading manifests …")

    manifest = load_librispeech_manifest(
        manifest_path,
        data_root,
        librispeech_root,
        subset=args.subset,
        seed=args.seed,
    )
    manifest["atc_1"] = load_atc_manifest(
        atc_manifest,
        atc_root,
        subset=args.subset,
        seed=args.seed,
    )
    manifest["atc_2"] = load_proc_unlab_atc_manifest(
        proc_unlab_root,
        subset=args.subset_proc_atc,
        seed=args.seed,
    )

    # Only include conditions that have data
    conditions = [c for c in ALL_CONDITIONS if manifest.get(c)]

    for cond in conditions:
        paths = manifest[cond]
        exists = sum(p.exists() for p in paths)
        log.info("  %-12s %d files, %d on disk", cond, len(paths), exists)

    run_analysis(
        manifest,
        conditions,
        output_dir=args.output_dir,
        cache_path=args.cache_path,
        workers=args.workers,
        max_audio_s=args.max_audio_s,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
