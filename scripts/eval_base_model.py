from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import torch
import torchaudio
from jiwer import wer as compute_wer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from class_models import SAMPLE_RATE  # noqa: local import

warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
logging.getLogger("transformers.generation.utils").addFilter(
    lambda r: "max_new_tokens" not in r.getMessage()
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EvalDataset(Dataset):
    def __init__(self, manifest_path: Path) -> None:
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
        wav = wav.mean(0)
        return {"waveform": wav, "text": item["text"], "idx": idx}


def collate_eval(batch: list[dict]) -> dict:
    clip_len = 5 * SAMPLE_RATE
    clipped = []
    for b in batch:
        wav = b["waveform"]
        if wav.shape[0] < clip_len:
            wav = torch.nn.functional.pad(wav, (0, clip_len - wav.shape[0]))
        else:
            start = (wav.shape[0] - clip_len) // 2
            wav = wav[start : start + clip_len]
        clipped.append(wav)
    return {
        "waveforms": torch.stack(clipped),
        "texts": [b["text"] for b in batch],
        "idxs": [b["idx"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


@torch.no_grad()
def transcribe_all(
    dataset: EvalDataset,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[list[str], list[str]]:
    """Returns (hypotheses, references) in dataset order."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_eval,
    )

    idx_to_hyp: dict[int, str] = {}
    idx_to_ref: dict[int, str] = {}

    for batch in tqdm(loader, desc="Transcribing", dynamic_ncols=True):
        feats = processor(
            list(batch["waveforms"].numpy()), sampling_rate=16_000, return_tensors="pt"
        ).input_features.to(device, dtype=torch.bfloat16, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            predicted_ids = model.generate(
                input_features=feats,
                num_beams=1,
                max_new_tokens=128,
                language="en",
                task="transcribe",
            )

        hyps = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        for sample_idx, hyp, ref in zip(batch["idxs"], hyps, batch["texts"]):
            idx_to_hyp[sample_idx] = hyp.strip().lower()
            idx_to_ref[sample_idx] = ref

        del feats, predicted_ids

    n = len(dataset)
    return [idx_to_hyp[i] for i in range(n)], [idx_to_ref[i] for i in range(n)]


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
        "--whisper-model",
        default="openai/whisper-large-v3",
        help="HuggingFace model ID for Whisper (default: whisper-large-v3)",
    )
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Whisper model: {args.whisper_model}")

    manifest = args.test_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    dataset = EvalDataset(manifest)
    print(f"Test samples:  {len(dataset):,}")

    print("\nLoading Whisper model …")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.whisper_model,
        torch_dtype=torch.bfloat16,
    )
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    model.generation_config.max_length = None
    model = model.to(device)
    model.eval()

    processor = WhisperProcessor.from_pretrained(args.whisper_model)
    processor.tokenizer.clean_up_tokenization_spaces = False

    hyps, refs = transcribe_all(
        dataset, model, processor, device, args.batch, args.workers
    )

    wer = compute_wer(refs, hyps)
    print(f"\nSamples: {len(dataset):,}")
    print(f"WER:     {wer * 100:.2f}%")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
