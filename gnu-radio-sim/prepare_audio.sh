#!/bin/bash
# prepare_audio.sh
# Converts any audio file (FLAC, MP3, WAV, etc.) to the correct format
# for the ATC AM GNU Radio simulation:
#   - 8000 Hz sample rate (ATC telephony standard)
#   - Mono (single channel)
#   - 16-bit PCM WAV
#
# Usage:
#   chmod +x prepare_audio.sh
#   ./prepare_audio.sh input.flac output.wav
#   ./prepare_audio.sh recording.mp3 atc_audio.wav

set -e

INPUT="$1"
OUTPUT="$2"

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 <input_audio_file> <output.wav>"
    echo ""
    echo "Examples:"
    echo "  $0 atc_recording.flac input_audio.wav"
    echo "  $0 liveatc_feed.mp3 input_audio.wav"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' not found."
    exit 1
fi

echo "=== ATC Simulation Audio Converter ==="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "Format: 8000 Hz, Mono, 16-bit PCM WAV"
echo ""

# Get input file info
echo "--- Input file info ---"
ffprobe -v quiet -show_streams -select_streams a:0 \
    -print_format compact "$INPUT" 2>/dev/null | \
    grep -E "sample_rate|channels|codec_name|duration" | \
    sed 's/stream|0|//' || true
echo ""

# Convert
echo "--- Converting ---"
ffmpeg -i "$INPUT" \
       -ar 8000 \
       -ac 1 \
       -acodec pcm_s16le \
       -y \
       "$OUTPUT"

echo ""
echo "--- Output file info ---"
ffprobe -v quiet -show_streams -select_streams a:0 \
    -print_format compact "$OUTPUT" 2>/dev/null | \
    grep -E "sample_rate|channels|codec_name|duration" | \
    sed 's/stream|0|//' || true

echo ""
echo "Done. Output saved to: $OUTPUT"
echo ""
echo "Next steps:"
echo "  1. Edit atc_am_tx.grc: set wav_file_path = \"$(realpath "$OUTPUT")\""
echo "  2. Edit atc_am_rx.grc: set output_wav_path to your desired output path"
echo "  3. Run: gnuradio-companion atc_am_tx.grc  (Terminal 1)"
echo "  4. Run: gnuradio-companion atc_am_rx.grc  (Terminal 2)"
