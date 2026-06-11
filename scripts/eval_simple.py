from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
from jiwer import wer as compute_wer
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

warnings.filterwarnings("ignore", message=".*max_new_tokens.*max_length.*")

ALL_CONDITIONS = ["perfect", "good", "okay", "bad"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path) -> None:
        self.root = manifest_path.parent
        self.samples: list[dict] = []

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
                        "condition": row.get("condition", "unknown"),
                    }
                )

        if not self.samples:
            raise ValueError(f"No samples found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        wav, sr = torchaudio.load(str(self.root / item["path"]))
        if sr != 16_000:
            wav = torchaudio.functional.resample(wav, sr, 16_000)
        wav = wav.mean(0)
        return {
            "waveform": wav,
            "text": item["text"],
            "condition": item["condition"],
            "idx": idx,
        }


def collate_fn(batch: list[dict], processor: WhisperProcessor) -> dict:
    waveforms = [b["waveform"].numpy() for b in batch]
    feats = processor(
        waveforms, sampling_rate=16_000, return_tensors="pt"
    ).input_features
    return {
        "input_features": feats,
        "texts": [b["text"] for b in batch],
        "conditions": [b["condition"] for b in batch],
        "idxs": [b["idx"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_eval(
    dataset: ManifestDataset,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[list[str], list[str], list[str]]:
    """Returns (hypotheses, references, conditions) in dataset order."""
    _collate = lambda b: collate_fn(b, processor)  # noqa: E731
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
    )

    results: dict[int, tuple[str, str, str]] = {}

    import transformers

    prev_v = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()

    for batch in tqdm(loader, desc="Transcribing", dynamic_ncols=True):
        feats = batch["input_features"].to(
            device, dtype=torch.bfloat16, non_blocking=True
        )

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            predicted_ids = model.generate(
                input_features=feats,
                num_beams=1,
                max_new_tokens=128,
                language="en",
                task="transcribe",
            )

        hyps = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        for idx, hyp, ref, cond in zip(
            batch["idxs"], hyps, batch["texts"], batch["conditions"]
        ):
            results[idx] = (hyp.strip().lower(), ref, cond)

        del feats, predicted_ids

    transformers.logging.set_verbosity(prev_v)

    n = len(dataset)
    hyps = [results[i][0] for i in range(n)]
    refs = [results[i][1] for i in range(n)]
    conds = [results[i][2] for i in range(n)]
    return hyps, refs, conds


# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------


def print_results(
    hyps: list[str], refs: list[str], conds: list[str], no_labels: bool = False
) -> None:
    overall_wer = compute_wer(refs, hyps)

    print(f"\n{'=' * 50}")
    print(f"  Overall  n={len(refs):>6,}  WER={overall_wer * 100:.2f}%")

    if not no_labels:
        print(f"{'=' * 50}")
        by_cond: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
        for h, r, c in zip(hyps, refs, conds):
            by_cond[c][0].append(h)
            by_cond[c][1].append(r)

        for cond in ALL_CONDITIONS:
            if cond not in by_cond:
                continue
            c_hyps, c_refs = by_cond[cond]
            c_wer = compute_wer(c_refs, c_hyps)
            print(f"  {cond:8s}  n={len(c_refs):>6,}  WER={c_wer * 100:.2f}%")

        other = [c for c in by_cond if c not in ALL_CONDITIONS]
        for cond in sorted(other):
            c_hyps, c_refs = by_cond[cond]
            c_wer = compute_wer(c_refs, c_hyps)
            print(f"  {cond:8s}  n={len(c_refs):>6,}  WER={c_wer * 100:.2f}%")

    print(f"{'=' * 50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument(
        "--model-id",
        default="openai/whisper-small",
        help="HuggingFace Whisper model ID (default: openai/whisper-small)",
    )
    p.add_argument(
        "--adapter-dir",
        type=Path,
        default=None,
        help="Path to a PEFT adapter directory (e.g. adapters_quick/all). "
        "Loaded on top of --model-id after the base model is built.",
    )
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip per-condition WER breakdown; print overall WER only.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:   {device}")
    print(f"Model:    {args.model_id}")
    if args.adapter_dir:
        print(f"Adapter:  {args.adapter_dir}")
    print(f"Manifest: {args.manifest}")

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    dataset = ManifestDataset(args.manifest)
    print(f"Samples:  {len(dataset):,}")

    print("\nLoading model ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.generation_config.begin_suppress_tokens = []
    model.generation_config.max_length = None
    if args.adapter_dir is not None:
        if not (args.adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"No adapter_config.json in {args.adapter_dir}")
        print(f"Loading LoRA adapter from {args.adapter_dir} ...")
        model = PeftModel.from_pretrained(
            model, str(args.adapter_dir), is_trainable=False
        )

    model = model.to(device).eval()

    processor = WhisperProcessor.from_pretrained(args.model_id)
    processor.tokenizer.clean_up_tokenization_spaces = False

    hyps, refs, conds = run_eval(
        dataset, model, processor, device, args.batch, args.workers
    )
    print_results(hyps, refs, conds, no_labels=args.no_labels)


if __name__ == "__main__":
    main()
