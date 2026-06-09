#!/usr/bin/env python3
"""
Programmatically generate alert WAV files for the fatigue-detection system.

No external audio libraries are required — only **numpy** (for waveform
synthesis) and the standard-library **wave** module.

Generated files
---------------
* ``sounds/beep.wav``  — 0.3-second 800 Hz sine-wave tone
* ``sounds/alarm.wav`` — 1.5-second alternating 600 Hz / 900 Hz siren

Usage
-----
Run as a standalone script::

    python generate_sounds.py

Or import and call programmatically::

    from generate_sounds import generate_beep, generate_alarm, generate_all
    generate_all()                       # creates both files
    generate_beep("custom_beep.wav")     # single file
"""

from __future__ import annotations

import os
import struct
import wave
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
_SAMPLE_RATE: int = 44_100
"""Samples per second (CD quality)."""

_AMPLITUDE: int = 28_000
"""Peak amplitude for 16-bit PCM (leaves ~3 dB headroom)."""

_SOUNDS_DIR: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sounds"
)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def generate_beep(
    filepath: Optional[str] = None,
    frequency: float = 800.0,
    duration: float = 0.3,
    sample_rate: int = _SAMPLE_RATE,
    amplitude: int = _AMPLITUDE,
) -> str:
    """Create a short sine-wave beep and save it as a 16-bit mono WAV.

    Parameters
    ----------
    filepath:
        Destination path.  Defaults to ``sounds/beep.wav``.
    frequency:
        Tone frequency in Hz.
    duration:
        Length in seconds.
    sample_rate:
        Samples per second.
    amplitude:
        Peak amplitude (max 32 767 for 16-bit PCM).

    Returns
    -------
    str
        Absolute path of the generated file.
    """
    filepath = filepath or os.path.join(_SOUNDS_DIR, "beep.wav")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    num_samples = int(sample_rate * duration)
    t = np.linspace(0.0, duration, num_samples, endpoint=False)
    samples = (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.int16)

    _write_wav(filepath, samples, sample_rate)
    print(f"[OK] Generated beep  -> {filepath}")
    return filepath


def generate_alarm(
    filepath: Optional[str] = None,
    freq_low: float = 600.0,
    freq_high: float = 900.0,
    duration: float = 1.5,
    cycle_period: float = 0.3,
    sample_rate: int = _SAMPLE_RATE,
    amplitude: int = _AMPLITUDE,
) -> str:
    """Create an alternating two-tone siren and save it as a 16-bit mono WAV.

    The siren alternates between *freq_low* and *freq_high* every
    *cycle_period* seconds for the full *duration*.

    Parameters
    ----------
    filepath:
        Destination path.  Defaults to ``sounds/alarm.wav``.
    freq_low:
        Low siren frequency in Hz.
    freq_high:
        High siren frequency in Hz.
    duration:
        Total length in seconds.
    cycle_period:
        Time (seconds) spent on each frequency before switching.
    sample_rate:
        Samples per second.
    amplitude:
        Peak amplitude (max 32 767 for 16-bit PCM).

    Returns
    -------
    str
        Absolute path of the generated file.
    """
    filepath = filepath or os.path.join(_SOUNDS_DIR, "alarm.wav")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    num_samples = int(sample_rate * duration)
    t = np.linspace(0.0, duration, num_samples, endpoint=False)

    # Build a frequency array that alternates between low and high
    cycle_samples = int(sample_rate * cycle_period)
    freq_array = np.empty(num_samples, dtype=np.float64)
    for i in range(num_samples):
        # Determine which half of the cycle we are in
        position_in_cycle = (i % (2 * cycle_samples))
        if position_in_cycle < cycle_samples:
            freq_array[i] = freq_low
        else:
            freq_array[i] = freq_high

    # Compute instantaneous phase via cumulative sum of angular frequencies
    phase = np.cumsum(2.0 * np.pi * freq_array / sample_rate)
    samples = (amplitude * np.sin(phase)).astype(np.int16)

    _write_wav(filepath, samples, sample_rate)
    print(f"[OK] Generated alarm -> {filepath}")
    return filepath


def generate_all() -> None:
    """Generate both ``beep.wav`` and ``alarm.wav`` in the default location."""
    generate_beep()
    generate_alarm()


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _write_wav(filepath: str, samples: np.ndarray, sample_rate: int) -> None:
    """Write a 16-bit mono PCM WAV file.

    Parameters
    ----------
    filepath:
        Output file path.
    samples:
        1-D array of ``int16`` audio samples.
    sample_rate:
        Samples per second.
    """
    with wave.open(filepath, "wb") as wf:
        n_channels = 1
        sample_width = 2  # bytes (16-bit)
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        # Pack all samples as little-endian signed 16-bit integers
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


# ---------------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all()
