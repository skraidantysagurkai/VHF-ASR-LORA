from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from class_models import SAMPLE_RATE  # noqa: local import

warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
logging.getLogger("transformers.generation.utils").addFilter(
    lambda r: "max_new_tokens" not in r.getMessage()
)

_MODEL_ID = "jacktol/whisper-large-v3-finetuned-for-ATC"
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
_CLIP_SEC = 30  # Whisper's maximum context window


class AudioFileDataset(Dataset):
    def __init__(self, audio_paths: list[Path]) -> None:
        self.paths = audio_paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        wav, sr = torchaudio.load(str(path))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)
        max_samples = _CLIP_SEC * SAMPLE_RATE
        if wav.shape[0] > max_samples:
            wav = wav[:max_samples]
        return {"waveform": wav, "idx": idx}


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(b["waveform"].shape[0] for b in batch)
    padded = []
    for b in batch:
        wav = b["waveform"]
        pad = max_len - wav.shape[0]
        if pad > 0:
            wav = torch.nn.functional.pad(wav, (0, pad))
        padded.append(wav)
    return {
        "waveforms": torch.stack(padded),
        "idxs": [b["idx"] for b in batch],
    }


@torch.no_grad()
def transcribe_all(
    audio_paths: list[Path],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> list[str]:
    print(f"\nLoading {_MODEL_ID} …")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        _MODEL_ID, torch_dtype=torch.bfloat16
    )
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    model.generation_config.max_length = None
    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(_MODEL_ID)
    processor.tokenizer.clean_up_tokenization_spaces = False

    dataset = AudioFileDataset(audio_paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    idx_to_hyp: dict[int, str] = {}

    for batch in tqdm(loader, desc="Transcribing", dynamic_ncols=True):
        feats = processor(
            list(batch["waveforms"].numpy()),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(device, dtype=torch.bfloat16, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            predicted_ids = model.generate(
                input_features=feats,
                num_beams=1,
                max_new_tokens=256,
                language="en",
                task="transcribe",
            )

        hyps = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        for sample_idx, hyp in zip(batch["idxs"], hyps):
            idx_to_hyp[sample_idx] = hyp.strip()
        del feats, predicted_ids

    return [idx_to_hyp[i] for i in range(len(audio_paths))]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory to search for audio files (recursive)",
    )
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip audio files that already have a .txt label",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_audio = sorted(
        p for p in args.audio_dir.rglob("*") if p.suffix.lower() in _AUDIO_EXTS
    )
    if not all_audio:
        print(f"No audio files found in {args.audio_dir}")
        return

    if args.skip_existing:
        audio_paths = [p for p in all_audio if not p.with_suffix(".txt").exists()]
        skipped = len(all_audio) - len(audio_paths)
        if skipped:
            print(f"Skipping {skipped} already-labeled file(s).")
    else:
        audio_paths = all_audio

    if not audio_paths:
        print("Nothing to transcribe.")
        return

    print(f"Device:  {device}")
    print(f"Model:   {_MODEL_ID}")
    print(f"Files:   {len(audio_paths):,}")

    transcriptions = transcribe_all(audio_paths, device, args.batch, args.workers)

    written = 0
    for path, text in zip(audio_paths, transcriptions):
        out = path.with_suffix(".txt")
        out.write_text(text, encoding="utf-8")
        written += 1

    print(f"\nDone. Wrote {written:,} transcription file(s).")


if __name__ == "__main__":
    main()
