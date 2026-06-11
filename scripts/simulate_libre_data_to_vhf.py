import argparse
import json
import math
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Generator

import numpy as np
from scipy.io import wavfile
from scipy.signal import firwin, kaiserord, lfilter, oaconvolve, resample_poly, upfirdn
from tqdm import tqdm


class ConditionType(Enum):
    PERFECT = "perfect"
    GOOD = "good"
    OKAY = "okay"
    BAD = "bad"

    def __str__(self) -> str:
        return self.value


CONDITIONS = {
    "perfect": {
        "MOD_INDEX": 0.85,
        "NOISE_VOLTAGE": 0.01,
        "MULTIPATH_TAPS": [1.0],
        "IMPULSE_AMPLITUDE": 0.0,
        "IMPULSE_RATE": 0.0,
        "FADING_RATE": 0.0,
        "FADING_DEPTH": 0.0,
        "FADING_OFFSET": 1.0,
        "SQUELCH_THRESHOLD": -60,
        "VOLUME": 1.0,
        "POST_CLICK_RATE": 0.0,
        "POST_CLICK_AMP": 0.0,
        "TONE_LEVEL": 0.0,
        "TONE_FREQ1": 400.0,
        "TONE_FREQ2": 800.0,
    },
    "good": {
        "MOD_INDEX": 0.80,
        "NOISE_VOLTAGE": 0.05,
        "MULTIPATH_TAPS": [1.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0, 0.03],
        "IMPULSE_AMPLITUDE": 0.6,
        "IMPULSE_RATE": 200.0,
        "FADING_RATE": 1.0,
        "FADING_DEPTH": 0.05,
        "FADING_OFFSET": 0.95,
        "SQUELCH_THRESHOLD": -60,
        "VOLUME": 1.0,
        "POST_CLICK_RATE": 0.0,
        "POST_CLICK_AMP": 0.0,
        "TONE_LEVEL": 0.0,
        "TONE_FREQ1": 400.0,
        "TONE_FREQ2": 800.0,
    },
    "okay": {
        "MOD_INDEX": 0.78,
        "NOISE_VOLTAGE": 0.09,
        "MULTIPATH_TAPS": [
            1.0,
            0.0,
            0.09,
            0.0,
            0.0,
            0.05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.02,
        ],
        "IMPULSE_AMPLITUDE": 0.9,
        "IMPULSE_RATE": 300.0,
        "FADING_RATE": 2.0,
        "FADING_DEPTH": 0.08,
        "FADING_OFFSET": 0.91,
        "SQUELCH_THRESHOLD": -58,
        "VOLUME": 1.0,
        "POST_CLICK_RATE": 0.0,
        "POST_CLICK_AMP": 0.0,
        "TONE_LEVEL": 0.0,
        "TONE_FREQ1": 400.0,
        "TONE_FREQ2": 800.0,
    },
    "bad": {
        "MOD_INDEX": 0.65,
        "NOISE_VOLTAGE": 0.080,
        "MULTIPATH_TAPS": [
            1.0,
            0.0,
            0.0,
            0.14,
            0.0,
            0.0,
            0.0,
            0.08,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.04,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.02,
        ],
        "IMPULSE_AMPLITUDE": 1.4,
        "IMPULSE_RATE": 600.0,
        "FADING_RATE": 5.0,
        "FADING_DEPTH": 0.12,
        "FADING_OFFSET": 0.83,
        "SQUELCH_THRESHOLD": -55,
        "VOLUME": 1.0,
        "POST_CLICK_RATE": 0.0,
        "POST_CLICK_AMP": 0.0,
        "TONE_LEVEL": 0.0,
        "TONE_FREQ1": 400.0,
        "TONE_FREQ2": 800.0,
    },
}

AUDIO_RATE = 16_000
QUAD_RATE = 240_000
SIM_RATE = 8_000  # RX chain processes at this rate
OUTPUT_RATE = 16_000  # final WAV rate (upsampled from SIM_RATE)
PAD_DURATION = 0.200


# =============================================================================
# Filter helpers
# =============================================================================


def _kaiser_lowpass(
    cutoff: float, fs: float, transition_width: float, attenuation_db: float = 60
) -> np.ndarray:
    nyq = fs / 2
    n_taps, beta = kaiserord(attenuation_db, transition_width / nyq)
    n_taps |= 1
    return firwin(n_taps, cutoff, window=("kaiser", beta), fs=fs)


def _kaiser_bandpass(
    low: float,
    high: float,
    fs: float,
    transition_width: float,
    attenuation_db: float = 60,
) -> np.ndarray:
    nyq = fs / 2
    n_taps, beta = kaiserord(attenuation_db, transition_width / nyq)
    n_taps |= 1
    return firwin(n_taps, [low, high], pass_zero=False, window=("kaiser", beta), fs=fs)


# Precomputed filter taps — computed once at import time, reused for every file.
_RX_DECIM = 6
_CHAN_RATE = QUAD_RATE // _RX_DECIM
_BP_AUDIO = _kaiser_bandpass(
    300, 3_400, AUDIO_RATE, transition_width=600, attenuation_db=80
)
_LP_RF = _kaiser_lowpass(12_500, QUAD_RATE, transition_width=2_500)
_BP_AUDIO_RX = _kaiser_bandpass(
    300, 3_400, SIM_RATE, transition_width=600, attenuation_db=80
)

_HP_DC_ALPHA = float(1.0 - 2.0 * np.pi * 30.0 / _CHAN_RATE)
# 300 ms AGC time constant at SIM_RATE.
_AGC_ALPHA = float(1.0 - np.exp(-1.0 / (0.300 * SIM_RATE)))


# =============================================================================
# Performance helpers
# =============================================================================


def _lfilter_cx(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """lfilter on complex input — splits real/imag to stay float32, avoids complex128 upcast."""
    xr = lfilter(b, a, x.real).astype(np.float32)
    xi = lfilter(b, a, x.imag).astype(np.float32)
    return (xr + 1j * xi).astype(np.complex64)


def _fir_causal(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """FFT overlap-add FIR; causal output identical to lfilter(h, 1.0, x)."""
    y = oaconvolve(x, h, mode="full")
    return y[: len(x)].astype(x.dtype)


def _fir_causal_cx(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    """_fir_causal on complex input via real/imag split."""
    return (_fir_causal(h, x.real) + 1j * _fir_causal(h, x.imag)).astype(np.complex64)


def _tile_to(template: np.ndarray, n: int) -> np.ndarray:
    """Tile `template` to length n."""
    reps = math.ceil(n / len(template))
    return np.tile(template, reps)[:n]


def _design_poly_filter(up: int, down: int) -> tuple[np.ndarray, int]:
    """Return (taps, half_len) matching scipy resample_poly's default Kaiser FIR design."""
    g = math.gcd(up, down)
    up_r, down_r = up // g, down // g
    max_rate = max(up_r, down_r)
    half_len = 10 * max_rate
    h = firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * up_r
    return h.astype(np.float64), half_len


def _poly_resample(
    x: np.ndarray, h: np.ndarray, half_h: int, up: int, down: int
) -> np.ndarray:
    """upfirdn with edge-padding that replicates resample_poly output exactly."""
    n_out = int(np.ceil(len(x) * up / down))
    pad = np.ones(half_h, dtype=x.dtype)
    x_pad = np.concatenate([x[0:1] * pad, x, x[-1:] * pad])
    return upfirdn(h, x_pad, up, down)[:n_out]


# Precomputed polyphase filters — designed once, reused for every utterance.
_H_TX_UP, _H_TX_UP_HALF = _design_poly_filter(15, 1)  # 16 kHz → 240 kHz
_H_RX_DN, _H_RX_DN_HALF = _design_poly_filter(1, 5)  # 40 kHz → 8 kHz
_H_RX_UP, _H_RX_UP_HALF = _design_poly_filter(2, 1)  # 8 kHz  → 16 kHz

# IF carrier/demod tiles — 10 kHz divides 240 kHz exactly (period = 24 samples).
_IF_PERIOD = QUAD_RATE // 10_000  # 24
_CARRIER_TILE = np.cos(2.0 * np.pi * np.arange(_IF_PERIOD) / _IF_PERIOD).astype(
    np.float32
)
_DEMOD_TILE = np.exp(-1j * 2.0 * np.pi * np.arange(_IF_PERIOD) / _IF_PERIOD).astype(
    np.complex64
)


# =============================================================================
# Module-level state
# =============================================================================

# Seeded from OS entropy once per process.  A fixed seed made every utterance's
# noise identical, defeating the stochastic augmentation.
_RNG = np.random.default_rng()


# =============================================================================
# Rayleigh fading
# =============================================================================


def _rayleigh_fading_envelope(
    n: int, fs: float, fading_rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Bandlimited Rayleigh envelope, zero-mean unit-std, upsampled to fs via np.interp."""
    INTERMEDIATE_RATE = 1_000.0
    n_mid = max(int(n / fs * INTERMEDIATE_RATE) + 2, 64)

    i_noise = rng.standard_normal(n_mid)
    q_noise = rng.standard_normal(n_mid)

    if fading_rate > 0:
        alpha = min(2.0 * np.pi * fading_rate / INTERMEDIATE_RATE, 0.5)
        i_noise = lfilter([alpha], [1.0, -(1.0 - alpha)], i_noise)
        q_noise = lfilter([alpha], [1.0, -(1.0 - alpha)], q_noise)

    envelope = np.sqrt(i_noise**2 + q_noise**2).astype(np.float32)
    mu, sigma = float(envelope.mean()), float(envelope.std())
    if sigma > 1e-10:
        envelope = (envelope - mu) / sigma

    x_mid = np.arange(n_mid)
    x_full = np.linspace(0, n_mid - 1, n)
    return np.interp(x_full, x_mid, envelope).astype(np.float32)


# =============================================================================
# Airplane background ambiance
# =============================================================================


def _load_bg_clips(bg_dir: Path) -> list[np.ndarray]:
    clips: list[np.ndarray] = []
    for p in sorted(bg_dir.glob("*.wav")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sr, data = wavfile.read(p)
            if data.ndim > 1:
                data = data[:, 0]
            clip = data.astype(np.float32) / 32768.0
            if sr != OUTPUT_RATE:
                clip = resample_poly(clip, OUTPUT_RATE, sr).astype(np.float32)
            clips.append(clip)
        except Exception as exc:
            print(f"[BG] could not load {p.name}: {exc}", file=sys.stderr)
    return clips


_BG_CLIPS: list[np.ndarray] = _load_bg_clips(
    Path(__file__).parent.parent / "data" / "airplane_background"
)


def _mix_background(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overlay a looped airplane ambiance clip at a random background level."""
    if not _BG_CLIPS:
        return audio
    clip = _BG_CLIPS[int(rng.integers(len(_BG_CLIPS)))]
    start = int(rng.integers(len(clip)))
    n = len(audio)
    bg = np.empty(n, dtype=np.float32)
    written, pos = 0, start
    while written < n:
        chunk = min(n - written, len(clip) - pos)
        bg[written : written + chunk] = clip[pos : pos + chunk]
        written += chunk
        pos = 0
    speech_rms = float(np.sqrt(np.mean(audio**2))) + 1e-8
    bg_rms = float(np.sqrt(np.mean(bg**2))) + 1e-8
    bg_gain = float(rng.uniform(0.04, 0.20)) * speech_rms / bg_rms
    return (audio + bg * bg_gain).astype(np.float32)


# =============================================================================
# Per-condition stochastic parameter ranges
# =============================================================================

_STOCHASTIC_PARAMS: dict[str, dict[str, tuple[float, float]]] = {
    "perfect": {
        "MOD_INDEX": (0.75, 0.92),
        "NOISE_VOLTAGE": (0.005, 0.020),
        "IMPULSE_AMPLITUDE": (0.00, 0.20),
        "IMPULSE_RATE": (0.0, 50.0),
        "FADING_RATE": (0.0, 0.5),
        "FADING_DEPTH": (0.000, 0.020),
        "FADING_OFFSET": (0.980, 1.000),
    },
    "good": {
        "MOD_INDEX": (0.60, 0.90),
        "NOISE_VOLTAGE": (0.030, 0.080),
        "IMPULSE_AMPLITUDE": (0.30, 1.00),
        "IMPULSE_RATE": (100.0, 350.0),
        "FADING_RATE": (0.3, 3.0),
        "FADING_DEPTH": (0.030, 0.080),
        "FADING_OFFSET": (0.900, 0.980),
        "POST_CLICK_RATE": (2.0, 10.0),
        "POST_CLICK_AMP": (0.05, 0.15),
    },
    "okay": {
        "MOD_INDEX": (0.58, 0.85),
        "NOISE_VOLTAGE": (0.065, 0.130),
        "IMPULSE_AMPLITUDE": (0.60, 1.20),
        "IMPULSE_RATE": (150.0, 450.0),
        "FADING_RATE": (1.0, 6.0),
        "FADING_DEPTH": (0.050, 0.110),
        "FADING_OFFSET": (0.860, 0.960),
        "POST_CLICK_RATE": (5.0, 25.0),
        "POST_CLICK_AMP": (0.08, 0.20),
        "TONE_LEVEL": (0.01, 0.04),
        "TONE_FREQ1": (300.0, 500.0),
        "TONE_FREQ2": (600.0, 1200.0),
    },
    "bad": {
        "MOD_INDEX": (0.40, 0.78),
        "NOISE_VOLTAGE": (0.060, 0.110),
        "IMPULSE_AMPLITUDE": (1.00, 1.80),
        "IMPULSE_RATE": (350.0, 800.0),
        "FADING_RATE": (3.0, 12.0),
        "FADING_DEPTH": (0.080, 0.160),
        "FADING_OFFSET": (0.760, 0.900),
        "POST_CLICK_RATE": (25.0, 80.0),
        "POST_CLICK_AMP": (0.15, 0.35),
        "TONE_LEVEL": (0.03, 0.08),
        "TONE_FREQ1": (300.0, 600.0),
        "TONE_FREQ2": (400.0, 1500.0),
    },
}


def _sample_condition(condition_str: str) -> dict:
    c = dict(CONDITIONS[condition_str])
    for key, (lo, hi) in _STOCHASTIC_PARAMS[condition_str].items():
        c[key] = float(_RNG.uniform(lo, hi))
    return c


# =============================================================================
# TX chain  —  simple AM: bandpass → modulate → upsample → fading → multipath → noise
# =============================================================================


def tx_chain(audio: np.ndarray, c: dict) -> np.ndarray:
    audio = _fir_causal(_BP_AUDIO, audio)

    baseband = (1.0 + c["MOD_INDEX"] * audio).astype(np.float32)
    baseband_up = _poly_resample(baseband, _H_TX_UP, _H_TX_UP_HALF, 15, 1)

    n = len(baseband_up)
    carrier = _tile_to(_CARRIER_TILE, n)
    rf = (baseband_up * carrier).astype(np.complex64)

    rayleigh = _rayleigh_fading_envelope(n, QUAD_RATE, c["FADING_RATE"], _RNG)
    fading = np.clip(
        c["FADING_OFFSET"] + c["FADING_DEPTH"] * rayleigh, 0.0, None
    ).astype(np.float32)
    rf = rf * fading

    rf = _lfilter_cx(c["MULTIPATH_TAPS"], [1.0], rf)

    awgn = (c["NOISE_VOLTAGE"] / np.sqrt(2)) * (
        _RNG.standard_normal(n) + 1j * _RNG.standard_normal(n)
    )
    rf = rf + awgn.astype(np.complex64)

    if c["IMPULSE_RATE"] > 0:
        n_bursts = max(1, int(c["IMPULSE_RATE"] / 10))
        burst_centres = _RNG.integers(0, n, n_bursts)
        impulse_idx = np.concatenate(
            [centre + _RNG.integers(-500, 500, 10) for centre in burst_centres]
        )
        impulse_idx = np.clip(impulse_idx, 0, n - 1)
        rf[impulse_idx] += (
            c["IMPULSE_AMPLITUDE"]
            * (
                _RNG.uniform(-1, 1, len(impulse_idx))
                + 1j * _RNG.uniform(-1, 1, len(impulse_idx))
            )
        ).astype(np.complex64)

    return rf.astype(np.complex64)


# =============================================================================
# RX chain
# =============================================================================


def _squelch(
    signal: np.ndarray, threshold_db: float, alpha: float = 0.01, hold_ms: float = 75.0
) -> np.ndarray:
    threshold_lin = 10.0 ** (threshold_db / 10.0)
    power = np.abs(signal) ** 2
    smooth = lfilter([alpha], [1.0, -(1.0 - alpha)], power)
    gate = smooth >= threshold_lin
    hold_n = max(1, int(hold_ms * 1e-3 * _CHAN_RATE))
    gate = (
        np.convolve(
            gate.astype(np.float32), np.ones(hold_n, dtype=np.float32), mode="full"
        )[: len(gate)]
        > 0
    )
    out = signal.copy()
    out[~gate] = 0.0
    return out


def rx_chain(rf: np.ndarray, c: dict) -> np.ndarray:
    n = len(rf)
    demod = _tile_to(_DEMOD_TILE, n)

    baseband = rf * demod
    baseband = _fir_causal_cx(_LP_RF, baseband)
    baseband = baseband[::_RX_DECIM].astype(np.complex64)
    baseband = _squelch(baseband, c["SQUELCH_THRESHOLD"])

    envelope = np.abs(baseband).astype(np.float32)
    envelope = lfilter([1.0, -1.0], [1.0, -_HP_DC_ALPHA], envelope).astype(np.float32)

    audio = _poly_resample(envelope, _H_RX_DN, _H_RX_DN_HALF, 1, 5).astype(np.float32)
    audio = _fir_causal(_BP_AUDIO_RX, audio)

    # Slow AGC (300 ms): normalises loudness while leaving fading variation audible.
    # max_gain=8 prevents noise blow-up during squelch-open silences.
    smooth_pwr = lfilter(
        [_AGC_ALPHA], [1.0, -(1.0 - _AGC_ALPHA)], audio.astype(np.float64) ** 2
    )
    gain = np.minimum(0.25 / np.sqrt(np.maximum(smooth_pwr, 1e-12)), 4.0)
    audio = (audio * gain.astype(np.float32)).astype(np.float32)

    n_audio = len(audio)

    if c["POST_CLICK_RATE"] > 0:
        n_clicks = int(c["POST_CLICK_RATE"] * n_audio / SIM_RATE)
        if n_clicks > 0:
            pos = _RNG.integers(0, n_audio, n_clicks)
            amp = (
                c["POST_CLICK_AMP"]
                * _RNG.uniform(0.5, 1.0, n_clicks)
                * _RNG.choice([-1.0, 1.0], n_clicks)
            ).astype(np.float32)
            np.add.at(audio, pos, amp)
            np.add.at(audio, np.minimum(pos + 1, n_audio - 1), amp * 0.3)

    if c["TONE_LEVEL"] > 0:
        t_a = np.arange(n_audio, dtype=np.float32) / SIM_RATE
        audio += (
            c["TONE_LEVEL"]
            * (
                np.sin(2.0 * np.pi * c["TONE_FREQ1"] * t_a)
                + 0.6 * np.sin(2.0 * np.pi * c["TONE_FREQ2"] * t_a)
            )
        ).astype(np.float32)

    audio = _fir_causal(_BP_AUDIO_RX, audio)

    return audio * c["VOLUME"]


# =============================================================================
# Audio I/O helpers
# =============================================================================


def read_flac(path: Path) -> tuple[np.ndarray, int]:
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


def normalize_audio(raw: np.ndarray) -> np.ndarray:
    if raw.dtype == np.int16:
        audio = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        audio = raw.astype(np.float32) / 2**31
    else:
        audio = raw.astype(np.float32)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


# =============================================================================
# Simulation entry point
# =============================================================================


def simulate_audio(
    audio: np.ndarray,
) -> Generator[tuple[np.ndarray, int, ConditionType], Any, None]:
    """Yield (int16_wav, sample_rate, condition) for each condition."""
    assert audio.dtype == np.float32
    # Normalise to a fixed TX level before the condition loop so all conditions
    # share the same input amplitude; per-condition noise/fading create the degradation.
    rms = float(np.sqrt(np.mean(audio**2))) + 1e-8
    audio = (audio / rms * 0.1).astype(np.float32)

    pad = np.zeros(int(PAD_DURATION * AUDIO_RATE), dtype=np.float32)
    padded = np.concatenate([pad, audio, pad])

    for condition_type in [
        ConditionType.GOOD,
        ConditionType.OKAY,
        ConditionType.BAD,
        ConditionType.PERFECT,
    ]:
        c = _sample_condition(str(condition_type))
        rf = tx_chain(_mix_background(padded, _RNG), c)
        audio_out = rx_chain(rf, c)  # 8 kHz — clip here

        # Normalize to safe range instead of clipping
        peak = np.abs(audio_out).max()
        if peak > 1.0:
            audio_out = audio_out / peak * 0.95

        audio_out = _poly_resample(
            audio_out, _H_RX_UP, _H_RX_UP_HALF, 2, 1
        )  # 8 kHz → 16 kHz
        out_i16 = (audio_out * 32767).astype(np.int16)
        yield out_i16, OUTPUT_RATE, condition_type


# =============================================================================
# Transcript parsing
# =============================================================================


def parse_trans_file(trans_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with open(trans_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def collect_items(input_dir: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for trans_path in sorted(input_dir.rglob("*.trans.txt")):
        transcripts = parse_trans_file(trans_path)
        flac_dir = trans_path.parent
        for utt_id, text in transcripts.items():
            flac_path = flac_dir / f"{utt_id}.flac"
            if flac_path.exists():
                items.append((flac_path, text))
    return items


# =============================================================================
# Worker function (top-level so multiprocessing can pickle it)
# =============================================================================


def process_item(item: tuple[Path, str, Path, str]) -> list[dict]:
    flac_path, text, output_dir, language = item
    output_dir = Path(output_dir)

    try:
        raw, sr = read_flac(flac_path)
        audio = normalize_audio(raw)
    except Exception as e:
        print(f"[SKIP] {flac_path.name}: {e}", file=sys.stderr)
        return []

    utt_id = flac_path.stem
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
                "text": text,
                "language": language,
                "condition": cond,
                "duration": round(duration_s, 3),
                "sample_rate": out_sr,
                "utterance_id": utt_id,
            }
        )

    return entries


# =============================================================================
# Main pipeline
# =============================================================================


def run_pipe(
    input_dir: Path,
    output_dir: Path,
    workers: int,
    limit: int | None,
    language: str = "en",
) -> None:
    if not input_dir.is_dir():
        raise ValueError(f"input_dir must be a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    items = collect_items(input_dir)
    if not items:
        print("No FLAC files with transcripts found.", file=sys.stderr)
        return
    if limit is not None:
        items = items[:limit]

    print(
        f"Found {len(items)} utterances → {len(items) * 4} output files "
        f"using {workers} worker(s)."
    )

    worker_args = [(flac_path, text, output_dir, language) for flac_path, text in items]
    total = len(worker_args)
    total_entries = 0

    with open(manifest_path, "w", encoding="utf-8") as manifest_f:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_item, item): item[0] for item in worker_args}
            for future in tqdm(as_completed(futures), total=total, unit="utt"):
                flac_path: Path = futures[future]
                try:
                    entries = future.result()
                    for entry in entries:
                        manifest_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    manifest_f.flush()
                    total_entries += len(entries)
                except Exception as e:
                    tqdm.write(f"[ERROR] {flac_path.name}: {e}")

    print(f"Done. {total_entries} entries written to {manifest_path}")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root of LibriSpeech dataset (contains .flac + .trans.txt files)",
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
        "--limit",
        type=int,
        default=None,
        help="Process at most N utterances (for testing)",
    )
    p.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language tag to embed in manifest (default: en)",
    )
    return p.parse_args()


if __name__ == "__main__":
    import time

    args = parse_args()
    t0 = time.time()
    run_pipe(args.input, args.output, args.workers, args.limit, args.language)
    print(f"Total elapsed: {time.time() - t0:.1f}s")
