from datasets import load_dataset
from paths import DATA_DIR
import polars as pl
import ast
import soundfile as sf
import io
import json

# Load dataset
ds = load_dataset(
    "jacktol/ATC-ASR-Dataset", token=""
)

for key in ds.keys():
    print(f"Key: {key}")
    print(f"Number of samples: {len(ds[key])}")
    # Don't index ds[key][0] — that triggers audio decoding via torchcodec,
    # which requires FFmpeg shared libraries that aren't present on this system.
    print("-" * 40)
    ds[key].to_csv(DATA_DIR / f"atc_asr_dataset_{key}.csv")

# Load CSVs using DATA_DIR
train_df = pl.read_csv(DATA_DIR / "atc_asr_dataset_train.csv")
validation_df = pl.read_csv(DATA_DIR / "atc_asr_dataset_validation.csv")
test_df = pl.read_csv(DATA_DIR / "atc_asr_dataset_test.csv")


def load_audio_to_numpy(audio_str):
    loaded = ast.literal_eval(audio_str)
    audio_array, sample_rate = sf.read(io.BytesIO(loaded["bytes"]))
    return audio_array, sample_rate


# Keep rows alongside audio so lengths stay in sync after filtering
def load_and_filter(df, target_sr=16000):
    rows, audios = [], []
    for row, audio_str in zip(df.iter_rows(named=True), df["audio"]):
        audio_array, sample_rate = load_audio_to_numpy(audio_str)
        if sample_rate == target_sr:
            rows.append(row)
            audios.append(audio_array)
    filtered_df = pl.DataFrame(rows)
    return filtered_df, audios


train_df, train_audio = load_and_filter(train_df)
validation_df, val_audio = load_and_filter(validation_df)
test_df, test_audio = load_and_filter(test_df)


def write_manifest(split_df, audios, output_dir):
    output_dir = DATA_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    audio_path = output_dir / "audio"
    audio_path.mkdir(parents=True, exist_ok=True)

    manifests = []
    for row, audio in zip(split_df.iter_rows(named=True), audios):
        uid = row["id"]
        transcription = row["text"]
        sf.write(audio_path / f"{uid}.wav", audio, 16000)
        manifests.append(
            {
                "audio_filepath": f"audio/{uid}.wav",
                "transcription": transcription,
                "language": "en",
                "condition": "real",
                "duration": round(len(audio) / 16000, 3),
                "sample_rate": 16000,
                "utterance_id": str(uid),
            }
        )

    with open(manifest_path, "w") as f:
        for m in manifests:
            f.write(json.dumps(m) + "\n")

    print(f"Wrote {len(manifests)} entries to {manifest_path}")


write_manifest(train_df, train_audio, "atc_asr_train")
write_manifest(validation_df, val_audio, "atc_asr_val")
write_manifest(test_df, test_audio, "atc_asr_test")

print(
    "Train data length: ",
    round(sum(len(a) / 16000 for a in train_audio) / 60, 3),
    "min",
)
print(
    "Validation data length: ",
    round(sum(len(a) / 16000 for a in val_audio) / 60, 3),
    "min",
)
print(
    "Test data length: ", round(sum(len(a) / 16000 for a in test_audio) / 60, 3), "min"
)
