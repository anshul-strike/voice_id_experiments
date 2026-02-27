#Get audio into proper format first
from pathlib import Path
from typing import Optional, Tuple
from pydub import AudioSegment
import logging
import re
import subprocess
import tempfile
from datetime import time
from scipy.signal import butter, filtfilt
import ffmpeg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_audio_for_embedding(
    audio_path: Path,
    output_dir: Path,
) -> Optional[Path]:
    """
    Convert audio to 16kHz mono WAV.
    """
    try:
        audio = AudioSegment.from_file(str(audio_path))
        audio = audio.set_channels(1).set_frame_rate(16000)

        wav_path = output_dir / f"{audio_path.stem}_converted.wav"
        audio.export(str(wav_path), format="wav")

        return wav_path

    except Exception as e:
        logger.error(f"Failed to prepare audio {audio_path}: {e}")
        return None


import numpy as np
import soundfile as sf


def decode_audio(
    audio_path: Path,
    target_sample_rate: int = 16000,
) -> Tuple[np.ndarray, int]:
    """
    Decode audio file to mono WAV using FFmpeg.

    Supports MP3, WAV, M4A, and other common formats.

    Args:
        audio_path: Path to audio file.
        target_sample_rate: Target sample rate in Hz.

    Returns:
        Tuple of (audio_array, sample_rate).

    Raises:
        subprocess.CalledProcessError: If FFmpeg fails.
        FileNotFoundError: If audio file doesn't exist.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                "/opt/homebrew/bin/ffmpeg", "-y", "-i", str(audio_path),
                "-ar", str(target_sample_rate),
                "-ac", "1",
                "-f", "wav",
                str(tmp_path),
            ],
            capture_output=True,
            check=True,
        )

        audio, sr = sf.read(tmp_path, dtype="float32")

        # Ensure mono
        if audio.ndim != 1:
            audio = np.mean(audio, axis=1)

        return audio, sr

    finally:
        tmp_path.unlink(missing_ok=True)


#Preprocess Audio
def apply_bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    lowcut: float = 70.0,
    highcut: float = 7600.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply bandpass filter to remove noise outside voice frequencies.

    Args:
        audio: Audio samples.
        sample_rate: Sample rate in Hz.
        lowcut: Low frequency cutoff in Hz.
        highcut: High frequency cutoff in Hz.
        order: Filter order.

    Returns:
        Filtered audio samples.
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, audio)

    return filtered.astype(np.float32)


def apply_upward_agc(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 20,
    target_db: float = -20.0,
    max_gain_db: float = 15.0,
    noise_percentile: float = 10.0,
    speech_margin_db: float = 6.0,
    gain_smooth_ms: float = 200.0,
) -> np.ndarray:
    """
    Apply upward Automatic Gain Control to boost quiet speech.

    Only boosts audio that appears to be speech (above noise floor),
    preserving the relative dynamics of the recording.

    Args:
        audio: Audio samples.
        sample_rate: Sample rate in Hz.
        frame_ms: Analysis frame length in milliseconds.
        target_db: Target RMS level in dB.
        max_gain_db: Maximum gain to apply in dB.
        noise_percentile: Percentile to use for noise floor estimation.
        speech_margin_db: Margin above noise floor for speech detection.
        gain_smooth_ms: Gain smoothing time constant in milliseconds.

    Returns:
        Gain-adjusted audio samples.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    n = (len(audio) // frame_len) * frame_len

    if n <= 0:
        return audio

    frames = audio[:n].reshape(-1, frame_len)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)

    # Estimate noise floor
    noise_floor_db = np.percentile(db, noise_percentile)
    speech_like = db > (noise_floor_db + speech_margin_db)

    # Calculate desired gain
    desired_gain_db = np.clip(target_db - db, 0.0, max_gain_db)
    gain_db = np.zeros_like(desired_gain_db)
    gain_db[speech_like] = desired_gain_db[speech_like]

    # Smooth gain transitions
    frames_per_sec = 1000.0 / frame_ms
    alpha = np.exp(-1.0 / (frames_per_sec * (gain_smooth_ms / 1000.0)))

    smoothed = np.zeros_like(gain_db)
    g = 0.0
    for i in range(len(gain_db)):
        g = alpha * g + (1.0 - alpha) * gain_db[i]
        smoothed[i] = g

    # Apply gain
    gain = 10.0 ** (smoothed / 20.0)
    processed = (frames * gain[:, None]).reshape(-1)

    # Append any remaining samples
    if n < len(audio):
        processed = np.concatenate([processed, audio[n:]])

    return processed.astype(np.float32)


def apply_limiter(
    audio: np.ndarray,
    peak_limit: float = 0.98,
) -> np.ndarray:
    """
    Apply peak limiter to prevent clipping.

    Args:
        audio: Audio samples.
        peak_limit: Maximum allowed peak level (0-1).

    Returns:
        Limited audio samples.
    """
    max_val = np.max(np.abs(audio)) + 1e-9
    if max_val > peak_limit:
        audio = audio * (peak_limit / max_val)
    return audio


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Apply full preprocessing pipeline to audio.

    Includes bandpass filtering, AGC, and limiting.

    Args:
        audio: Raw audio samples.
        sample_rate: Sample rate in Hz.
        settings: Optional Settings instance for parameters.

    Returns:
        Preprocessed audio samples.
    """

    # Apply bandpass filter
    audio = apply_bandpass_filter(
        audio,
        sample_rate,
    )

    # Apply AGC
    audio = apply_upward_agc(
        audio,
        sample_rate,
    )

    # Apply limiter
    audio = apply_limiter(audio)

    return audio


def save_audio(
    audio: np.ndarray,
    sample_rate: int,
    output_path: Path,
) -> Path:
    """
    Save audio to a WAV file.

    Args:
        audio: Audio samples.
        sample_rate: Sample rate in Hz.
        output_path: Destination path.

    Returns:
        Path to saved file.
        """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    return output_path