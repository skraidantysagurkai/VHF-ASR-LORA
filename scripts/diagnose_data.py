import polars as pl
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from tqdm import tqdm


def compute_audio_stats(
    manifest_path: str | Path, dataset_dir: str | Path
) -> pl.DataFrame:
    dataset_dir = Path(dataset_dir)
    df = pl.read_ndjson(manifest_path)

    rows = []
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="Computing stats"):
        path = dataset_dir / row["audio_filepath"]
        try:
            sr, data = wavfile.read(path)
            audio = data.astype(np.float32) / 32768.0

            rms = float(np.sqrt(np.mean(audio**2)))
            peak = float(np.abs(audio).max())
            db_rms = float(20 * np.log10(rms + 1e-8))
            db_peak = float(20 * np.log10(peak + 1e-8))
            crest = float(peak / (rms + 1e-8))  # crest factor: high = impulsive/clicky
            zero_cross = float(
                np.mean(np.diff(np.sign(audio)) != 0)
            )  # zero crossing rate
            silence_r = float(
                np.mean(np.abs(audio) < 0.01)
            )  # fraction of near-silent samples

            rows.append(
                {
                    **row,
                    "rms": rms,
                    "db_rms": db_rms,
                    "peak": peak,
                    "db_peak": db_peak,
                    "crest_factor": crest,
                    "zero_crossing_rate": zero_cross,
                    "silence_ratio": silence_r,
                }
            )
        except Exception as e:
            print(f"[SKIP] {path}: {e}")

    return pl.DataFrame(rows)


def print_stats(df: pl.DataFrame) -> None:
    stats = (
        df.group_by("condition")
        .agg(
            [
                pl.col("db_rms").mean().alias("db_rms_mean"),
                pl.col("db_rms").std().alias("db_rms_std"),
                pl.col("db_peak").mean().alias("db_peak_mean"),
                pl.col("crest_factor").mean().alias("crest_mean"),
                pl.col("crest_factor").std().alias("crest_std"),
                pl.col("zero_crossing_rate").mean().alias("zcr_mean"),
                pl.col("silence_ratio").mean().alias("silence_mean"),
                pl.col("duration").mean().alias("duration_mean"),
                pl.len().alias("count"),
            ]
        )
        .sort("condition")
    )
    with pl.Config(tbl_cols=-1, tbl_width_chars=300):
        print(stats)


if __name__ == "__main__":
    for dataset, path in [
        ("dev-combined", "data/LibriSpeech-dev-combined"),
        ("test-combined", "data/LibriSpeech-test-combined"),
        ("train-final", "data/LibriSpeech-train-final"),
        ("train-small", "data/LibriSpeech-train-small"),
    ]:
        print(f"\n{'=' * 60}\n  {dataset}\n{'=' * 60}")
        df = compute_audio_stats(f"{path}/manifest.jsonl", path)
        print_stats(df)
