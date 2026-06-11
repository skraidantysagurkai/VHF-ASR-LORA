from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import torch
import torchaudio
from jiwer import wer as compute_wer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from class_models import SAMPLE_RATE  # noqa: local import

warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")
logging.getLogger("transformers.generation.utils").addFilter(
    lambda r: "max_new_tokens" not in r.getMessage()
)

_MODELS: dict[str, dict[str, Any]] = {
    "jacktol": {
        "hf_id": "jacktol/whisper-large-v3-finetuned-for-ATC",
        "backend": "transformers",
    },
    "aether-raid": {
        "hf_id": "aether-raid/astra-atc-models",
        "subfolder": "ASR/whisper",
        "backend": "faster-whisper",
    },
}


# ---------------------------------------------------------------------------
# Dataset (identical to eval_base_model.py)
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


_WHISPER_MAX_SAMPLES = 30 * SAMPLE_RATE  # Whisper hard limit


def collate_eval(batch: list[dict]) -> dict:
    wavs = []
    for b in batch:
        wav = b["waveform"]
        if wav.shape[0] > _WHISPER_MAX_SAMPLES:
            wav = wav[:_WHISPER_MAX_SAMPLES]
        wavs.append(wav)
    max_len = max(w.shape[0] for w in wavs)
    padded = [torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in wavs]
    return {
        "waveforms": torch.stack(padded),
        "texts": [b["text"] for b in batch],
        "idxs": [b["idx"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Transformers backend  (jacktol)
# ---------------------------------------------------------------------------


@torch.no_grad()
def transcribe_transformers(
    dataset: EvalDataset,
    hf_id: str,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[list[str], list[str]]:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    print(f"\nLoading {hf_id} …")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(hf_id, torch_dtype=torch.bfloat16)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    model.generation_config.max_length = None
    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(hf_id)
    processor.tokenizer.clean_up_tokenization_spaces = False

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
            list(batch["waveforms"].numpy()),
            sampling_rate=16_000,
            return_tensors="pt",
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
# faster-whisper backend  (aether-raid)
# ---------------------------------------------------------------------------


def transcribe_faster_whisper(
    dataset: EvalDataset,
    hf_id: str,
    subfolder: str,
    device: torch.device,
) -> tuple[list[str], list[str]]:
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    print(f"\nDownloading {hf_id}/{subfolder} …")
    repo_path = snapshot_download(hf_id, allow_patterns=f"{subfolder}/*")
    model_path = os.path.join(repo_path, subfolder)

    device_str = "cuda" if device.type == "cuda" else "cpu"
    compute_type = "float16" if device.type == "cuda" else "int8"

    print(f"Loading faster-whisper from {model_path} …")
    model = WhisperModel(model_path, device=device_str, compute_type=compute_type)

    hyps: list[str] = []
    refs: list[str] = []

    for item in tqdm(dataset.samples, desc="Transcribing", dynamic_ncols=True):
        wav, sr = torchaudio.load(str(dataset.root / item["path"]))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)

        segments, _ = model.transcribe(wav.numpy(), language="en", beam_size=5)
        hyp = " ".join(seg.text.strip() for seg in segments).strip().lower()
        hyps.append(hyp)
        refs.append(item["text"])

    return hyps, refs


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
        "--model",
        choices=list(_MODELS),
        required=True,
        help="Which pretrained model to evaluate",
    )
    p.add_argument(
        "--batch", type=int, default=16, help="Batch size (transformers backend only)"
    )
    p.add_argument("--workers", type=int, default=4)
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _MODELS[args.model]

    print(f"Device:  {device}")
    print(f"Model:   {cfg['hf_id']}")

    manifest = args.test_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    dataset = EvalDataset(manifest)
    print(f"Samples: {len(dataset):,}")

    if cfg["backend"] == "transformers":
        hyps, refs = transcribe_transformers(
            dataset,
            cfg["hf_id"],
            device,
            args.batch,
            args.workers,
        )
    else:
        hyps, refs = transcribe_faster_whisper(
            dataset,
            cfg["hf_id"],
            cfg["subfolder"],
            device,
        )

    wer = compute_wer(refs, hyps)
    print(f"\nSamples: {len(dataset):,}")
    print(f"WER:     {wer * 100:.2f}%")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
