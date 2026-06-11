#!/usr/bin/env python3
"""
ATC VHF AM signal chain — sequential numpy implementation.
Mirrors the GNU Radio TX→RX flow graph without ZMQ or a GUI.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.signal import firwin, kaiserord, lfilter, resample_poly
from scipy.io import wavfile
from paths import DATA_DIR


CONDITIONS = {
    # ── PERFECT ────────────────────────────────────────────────────────────
    "perfect": {
        "MOD_INDEX": 0.90,
        "IF_FREQ": 10_000,
        "NOISE_VOLTAGE": 0.01,
        "FREQ_OFFSET": 0.0,
        "MULTIPATH_TAPS": [1.0],
        "IMPULSE_AMPLITUDE": 0.0,
        "IMPULSE_RATE": 0.0,
        "FADING_RATE": 0.0,
        "FADING_DEPTH": 0.0,
        "FADING_OFFSET": 1.0,
        "CHANNEL_BW": 25_000,
        "SQUELCH_THRESHOLD": -60,
        "VOLUME": 1.0,
    },
    # ── GOOD ───────────────────────────────────────────────────────────────
    # Cruise altitude, solid LOS, light thermal noise, very sparse crackle.
    "good": {
        "MOD_INDEX": 0.90,
        "IF_FREQ": 10_000,
        "NOISE_VOLTAGE": 0.05,
        "FREQ_OFFSET": 0.0002,
        "MULTIPATH_TAPS": [1.0, 0.0, 0.05, 0.0, 0.02],
        "IMPULSE_AMPLITUDE": 0.6,  # was 0.3 — raised so crackle is audible
        "IMPULSE_RATE": 20.0,
        "FADING_RATE": 0.05,
        "FADING_DEPTH": 0.05,
        "FADING_OFFSET": 0.95,
        "CHANNEL_BW": 25_000,
        "SQUELCH_THRESHOLD": -60,
        "VOLUME": 1.0,
    },
    # ── OKAY ───────────────────────────────────────────────────────────────
    # Intermediate between good and bad — descending, moderate range,
    # light terrain shadow, occasional crackle bursts, gentle fading.
    "okay": {
        "MOD_INDEX": 0.90,
        "IF_FREQ": 10_000,
        "NOISE_VOLTAGE": 0.12,  # between good(0.05) and bad(0.20) — ~20dB SNR
        "FREQ_OFFSET": 0.0005,
        "MULTIPATH_TAPS": [1.0, 0.0, 0.08, 0.0, 0.04, 0.0, 0.01],
        "IMPULSE_AMPLITUDE": 1.2,  # clearly audible pops
        "IMPULSE_RATE": 45.0,  # between good(20) and bad(80)
        "FADING_RATE": 0.05,
        "FADING_DEPTH": 0.10,  # between good(0.05) and bad(0.15)
        "FADING_OFFSET": 0.88,  # between good(0.95) and bad(0.80)
        "CHANNEL_BW": 25_000,
        "SQUELCH_THRESHOLD": -58,
        "VOLUME": 1.0,
    },
    # ── BAD ────────────────────────────────────────────────────────────────
    # Was "okay" params — terrain obstruction, heavy crackle, noticeable fading.
    "bad": {
        "MOD_INDEX": 0.90,
        "IF_FREQ": 10_000,
        "NOISE_VOLTAGE": 0.20,
        "FREQ_OFFSET": 0.0008,
        "MULTIPATH_TAPS": [1.0, 0.0, 0.15, 0.0, 0.08, 0.0, 0.03],
        "IMPULSE_AMPLITUDE": 1.8,  # was 0.8 — raised significantly for audible crackle
        "IMPULSE_RATE": 80.0,
        "FADING_RATE": 0.08,  # slightly faster than okay
        "FADING_DEPTH": 0.15,
        "FADING_OFFSET": 0.80,
        "CHANNEL_BW": 25_000,
        "SQUELCH_THRESHOLD": -55,
        "VOLUME": 1.0,
    },
}


# =============================================================================
# Parameters
# =============================================================================

CONDITIONS_STR = "bad"

INPUT_WAV_PATH = str(DATA_DIR / "input_audio.wav")
OUTPUT_WAV_PATH = str(DATA_DIR / f"output_audio_np_{CONDITIONS_STR}.wav")

AUDIO_RATE = 16_000  # Hz  — source / sink sample rate
QUAD_RATE = 240_000  # Hz  — RF simulation sample rate

OUTPUT_RATE = 16_000  # Hz, output WAV sample rate (upsample from AUDIO_RATE)
PAD_DURATION = 0.200  # seconds of silence added before and after transmission


# =============================================================================
# Filter design helpers
# =============================================================================


def _kaiser_lowpass(
    cutoff: float, fs: float, transition_width: float, attenuation_db: float = 60
) -> np.ndarray:
    nyq = fs / 2
    n_taps, beta = kaiserord(attenuation_db, transition_width / nyq)
    n_taps |= 1  # force odd
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


# =============================================================================
# TX chain
# =============================================================================


def tx_chain(audio: np.ndarray) -> np.ndarray:
    """
    Float32 mono audio at AUDIO_RATE → complex64 RF signal at QUAD_RATE.

    Steps (matching atc_am_tx.py):
      band_pass_filter → mod_index scale → add_const(1) →
      rational_resampler (30×) → multiply carrier →
      float_to_complex → multiply fading →
      channel_model (multipath + freq_offset + AWGN) →
      impulse noise addition → throttle (no-op here)
    """

    c = CONDITIONS[CONDITIONS_STR]

    # 1. Band-pass filter: telephony band 300–3400 Hz  (transition 100 Hz)
    bp = _kaiser_bandpass(300, 3400, AUDIO_RATE, transition_width=100)
    audio = lfilter(bp, 1.0, audio)

    # 2. AM envelope: (1 + mod_index · audio)
    baseband = 1.0 + c["MOD_INDEX"] * audio

    # 3. Upsample AUDIO_RATE → QUAD_RATE  (30× interpolation)
    upsample = QUAD_RATE // AUDIO_RATE  # 30
    baseband_up = resample_poly(baseband, upsample, 1)

    # 4. Multiply by IF carrier → DSB-FC AM
    n = len(baseband_up)
    t = np.arange(n) / QUAD_RATE
    rf = baseband_up * np.cos(2.0 * np.pi * c["IF_FREQ"] * t)

    # 5. Convert to complex (imaginary = 0), matching float_to_complex block
    rf = rf.astype(np.complex64)

    # 6. Ricean fading: sinusoidal amplitude envelope
    fading = (
        c["FADING_DEPTH"] * np.sin(2.0 * np.pi * c["FADING_RATE"] * t)
        + c["FADING_OFFSET"]
    ).astype(np.float32)
    rf = rf * fading

    # 7. Multipath: FIR filter with taps
    rf = lfilter(c["MULTIPATH_TAPS"], 1.0, rf)

    # 8. Frequency offset (normalized: phase increment = 2π·FREQ_OFFSET per sample)
    rf = rf * np.exp(1j * 2.0 * np.pi * c["FREQ_OFFSET"] * np.arange(n))

    # 9. AWGN  (σ = NOISE_VOLTAGE / √2 per component → |noise| ~ NOISE_VOLTAGE)
    rng = np.random.default_rng(seed=42)
    awgn = (c["NOISE_VOLTAGE"] / np.sqrt(2)) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )
    rf = rf + awgn.astype(np.complex64)

    # 10. Impulse noise: burst mode — impulses cluster around burst centres
    impulses = np.zeros(n, dtype=np.complex64)
    if c["IMPULSE_RATE"] > 0:
        n_bursts = max(1, int(c["IMPULSE_RATE"] / 10))
        burst_centres = rng.integers(0, n, n_bursts)
        impulse_idx = np.concatenate(
            [
                centre
                + rng.integers(-500, 500, 10)  # 10 impulses within ±2ms at 240 kHz
                for centre in burst_centres
            ]
        )
        impulse_idx = np.clip(impulse_idx, 0, n - 1)
        impulses[impulse_idx] = c["IMPULSE_AMPLITUDE"] * (
            rng.uniform(-1, 1, len(impulse_idx))
            + 1j * rng.uniform(-1, 1, len(impulse_idx))
        )
    rf = rf + impulses

    return rf.astype(np.complex64)


# =============================================================================
# RX chain
# =============================================================================


def _agc(
    signal: np.ndarray,
    rate: float = 1e-3,
    reference: float = 1.0,
    max_gain: float = 65536.0,
) -> np.ndarray:
    """
    IIR-smoothed AGC matching analog.agc_cc(rate, reference, gain=1).
    Steady-state: gain ≈ sqrt(reference / smooth_power).
    """
    power = np.abs(signal) ** 2
    smooth = lfilter([rate], [1.0, -(1.0 - rate)], power)
    gain = np.sqrt(reference / np.maximum(smooth, 1e-20))
    gain = np.minimum(gain, max_gain)
    return (signal * gain).astype(np.complex64)


def _squelch(
    signal: np.ndarray, threshold_db: float, alpha: float = 0.01
) -> np.ndarray:
    """
    Power squelch: zero out samples where IIR-smoothed power < threshold.
    Matches analog.pwr_squelch_cc(threshold, alpha=0.01, ramp=10).
    """
    threshold_lin = 10.0 ** (threshold_db / 10.0)
    power = np.abs(signal) ** 2
    smooth = lfilter([alpha], [1.0, -(1.0 - alpha)], power)
    out = signal.copy()
    out[smooth < threshold_lin] = 0.0
    return out


def rx_chain(rf: np.ndarray) -> np.ndarray:
    """
    Complex64 RF signal at QUAD_RATE → float32 audio at OUTPUT_RATE (16 kHz).

    Steps (matching atc_am_rx.py):
      zeromq_sub (no-op) → freq_xlating_fir (÷6, 40 kHz) →
      agc_cc → pwr_squelch_cc →
      am_demod_cf (envelope) → radio bandlimit LPF (3–4 kHz) →
      resample_poly ×2/5 (40 kHz → 16 kHz) → multiply_const (volume)
    """

    c = CONDITIONS[CONDITIONS_STR]

    n = len(rf)
    t = np.arange(n) / QUAD_RATE
    rx_decim = 6  # 240 kHz → 40 kHz
    channel_rate = QUAD_RATE // rx_decim  # 40 000 Hz

    # 1. Frequency-translating FIR filter: shift IF → baseband, lowpass, decimate 6×
    lp_rf = _kaiser_lowpass(c["CHANNEL_BW"] / 2, QUAD_RATE, transition_width=2500)
    baseband = rf * np.exp(-1j * 2.0 * np.pi * c["IF_FREQ"] * t)
    baseband = lfilter(lp_rf, 1.0, baseband)
    baseband = baseband[::rx_decim].astype(np.complex64)

    # 2. AGC
    baseband = _agc(baseband)

    # 3. Power squelch
    baseband = _squelch(baseband, c["SQUELCH_THRESHOLD"])

    # 4. AM demodulation: envelope detection + DC removal
    envelope = np.abs(baseband).astype(np.float32)
    envelope -= envelope.mean()

    # 5. Radio bandlimitation: hard lowpass at 3.4 kHz (telephony ceiling),
    #    transition to stopband by 4.0 kHz — rejects channel noise above speech band.
    #    resample_poly handles the anti-aliasing for the 40 kHz → 16 kHz step (×2/5).
    lp_audio = _kaiser_lowpass(3400, channel_rate, transition_width=600)
    audio = lfilter(lp_audio, 1.0, envelope)
    audio = resample_poly(audio, 2, 5).astype(np.float32)  # 40 000 × 2/5 = 16 000 Hz

    # 6. Volume
    audio = audio * c["VOLUME"]

    return audio


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    fs, raw = wavfile.read(INPUT_WAV_PATH)
    if fs != AUDIO_RATE:
        raise ValueError(f"Expected {AUDIO_RATE} Hz input, got {fs} Hz")

    # Normalize to float32 in [-1, 1]
    if raw.dtype == np.int16:
        audio = raw.astype(np.float32) / 32768.0
    elif raw.dtype == np.int32:
        audio = raw.astype(np.float32) / 2**31
    else:
        audio = raw.astype(np.float32)

    if audio.ndim > 1:  # use first channel only
        audio = audio[:, 0]

    print(
        f"Input : {len(audio):,} samples  {len(audio) / AUDIO_RATE:.2f}s  @ {AUDIO_RATE} Hz"
    )

    # Pad with silence before and after transmission
    pad = np.zeros(int(PAD_DURATION * AUDIO_RATE), dtype=np.float32)
    audio = np.concatenate([pad, audio, pad])

    rf = tx_chain(audio)
    print(f"TX    : {len(rf):,} complex samples @ {QUAD_RATE} Hz")

    audio_out = rx_chain(rf)
    print(f"RX    : {len(audio_out):,} samples @ {OUTPUT_RATE} Hz")

    out_i16 = (audio_out * 32767).astype(np.int16)
    wavfile.write(OUTPUT_WAV_PATH, OUTPUT_RATE, out_i16)
    print(f"Output: {OUTPUT_WAV_PATH}")


if __name__ == "__main__":
    import time

    now = time.time()
    main()
    print(f"Elapsed time: {time.time() - now:.2f}s")
