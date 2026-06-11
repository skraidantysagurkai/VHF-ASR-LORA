import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

# Share filter constants, RNG, and simulation pipeline from the sibling script.
# This avoids duplicating the expensive kaiserord/firwin calls.
sys.path.insert(0, str(Path(__file__).parent))
from simulate_libre_data_to_vhf import normalize_audio, simulate_audio  # noqa: E402

_DEFAULT_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read any ffmpeg-supported audio file to 16 kHz mono int16."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.int16), 16_000


def collect_items(input_dir: Path, extensions: set[str]) -> list[Path]:
    """Return all audio files under input_dir matching the given extensions."""
    return sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and p.suffix.lower() in extensions
    )


def process_item(item: tuple[Path, Path, str]) -> list[dict]:
    """
    Read one audio file, run all conditions, write WAVs, return manifest entries.
    item = (audio_path, output_dir, language)
    """
    audio_path, output_dir, language = item
    output_dir = Path(output_dir)

    try:
        raw, sr = read_audio(audio_path)
        audio = normalize_audio(raw)
    except Exception as e:
        print(f"[SKIP] {audio_path.name}: {e}", file=sys.stderr)
        return []

    utt_id = audio_path.stem
    duration_s = len(raw) / sr
    entries: list[dict] = []

    for out_i16, out_sr, condition_type in simulate_audio(audio):
        cond = str(condition_type)
        rel_path = Path("audio") / cond / f"{utt_id}.wav"
        abs_path = output_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(abs_path, out_sr, out_i16)

        entries.append(
            {
                "audio_filepath": str(rel_path),
                "text": "",
                "language": language,
                "condition": cond,
                "duration": round(duration_s, 3),
                "sample_rate": out_sr,
                "utterance_id": utt_id,
            }
        )

    return entries


def run_pipe(
    input_dir: Path,
    output_dir: Path,
    workers: int,
    limit: int | None,
    language: str,
    extensions: set[str],
) -> None:
    if not input_dir.is_dir():
        raise ValueError(f"input_dir must be a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    items = collect_items(input_dir, extensions)
    if not items:
        print("No audio files found.", file=sys.stderr)
        return

    if limit is not None:
        items = items[:limit]

    n_conditions = 4  # very_bad / good / okay / bad (see simulate_audio)
    print(
        f"Found {len(items)} audio files → {len(items) * n_conditions} output files "
        f"using {workers} worker(s)."
    )

    worker_args = [(p, output_dir, language) for p in items]
    total = len(worker_args)
    total_entries = 0

    with open(manifest_path, "w", encoding="utf-8") as manifest_f:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_item, item): item[0] for item in worker_args}
            for future in tqdm(as_completed(futures), total=total, unit="file"):
                audio_path: Path = futures[future]
                try:
                    entries = future.result()
                    for entry in entries:
                        manifest_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    manifest_f.flush()
                    total_entries += len(entries)
                except Exception as e:
                    tqdm.write(f"[ERROR] {audio_path.name}: {e}")

    print(f"Done. {total_entries} entries written to {manifest_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing audio files to process",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for WAV files and manifest.jsonl",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Process at most N files (for testing)"
    )
    p.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language tag to embed in manifest (default: en)",
    )
    p.add_argument(
        "--extensions",
        type=str,
        default=",".join(sorted(_DEFAULT_EXTENSIONS)),
        help="Comma-separated audio extensions to scan "
        f"(default: {','.join(sorted(_DEFAULT_EXTENSIONS))})",
    )
    return p.parse_args()


if __name__ == "__main__":
    import time

    args = parse_args()
    extensions = {f".{e.lstrip('.')}" for e in args.extensions.split(",")}
    t0 = time.time()
    run_pipe(
        args.input, args.output, args.workers, args.limit, args.language, extensions
    )
    print(f"Total elapsed: {time.time() - t0:.1f}s")
