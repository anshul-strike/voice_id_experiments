#Get audio into proper format first
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from pydub import AudioSegment
import logging
import re
import subprocess
import math
import tempfile
from datetime import time
from scipy.signal import butter, filtfilt
import ffmpeg
import rapidfuzz
import whisperx
from dataclasses import dataclass
from rapidfuzz import fuzz

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



from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import torch
import torchaudio
import numpy as np
import soundfile as sf


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Segment:
    start: int  # sample index (inclusive)
    end: int    # sample index (exclusive)

    @property
    def length(self) -> int:
        return self.end - self.start


# -----------------------------
# Audio I/O
# -----------------------------
def load_mono_audio(
    audio_path: Union[str, Path],
    target_sr: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:

    audio_path = Path(audio_path)

    audio, sr = sf.read(str(audio_path))

    # Convert to mono if needed
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Convert to float32 torch tensor
    audio = torch.from_numpy(audio).float()

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        import torchaudio
        audio = torchaudio.functional.resample(
            audio.unsqueeze(0), sr, target_sr
        ).squeeze(0)
        sr = target_sr

    return audio, sr


def save_wav(
    audio: Union[torch.Tensor, np.ndarray],
    sr: int,
    out_path: Union[str, Path],
    subtype: str = "PCM_16",
) -> Path:
    """
    Save mono audio to a wav file.

    Args:
      audio: torch Tensor (T,) or numpy array (T,)
      sr: sample rate
      out_path: output file path

    Returns:
      Path to saved file
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = audio

    sf.write(str(out_path), audio_np, sr, subtype=subtype)
    return out_path


# -----------------------------
# Silero VAD loading (cached)
# -----------------------------
_SILERO_CACHE = {"model": None, "utils": None}

def load_silero_vad(device: Optional[str] = None):
    """
    Loads Silero VAD model + utility functions from torch.hub.
    Caches the result so you can call it repeatedly in notebook cells.

    Args:
      device: 'cpu', 'cuda', 'mps', etc. If None, auto.

    Returns:
      model, utils_tuple
    """
    if _SILERO_CACHE["model"] is None or _SILERO_CACHE["utils"] is None:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        _SILERO_CACHE["model"] = model
        _SILERO_CACHE["utils"] = utils

    model = _SILERO_CACHE["model"]
    utils = _SILERO_CACHE["utils"]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    return model, utils


# -----------------------------
# VAD + timestamps
# -----------------------------
def get_speech_timestamps_silero(
    audio: torch.Tensor,
    sr: int,
    *,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 500,
    device: Optional[str] = None,
) -> List[Segment]:
    """
    Runs Silero VAD and returns speech segments as sample-index Segments.

    Args:
      audio: mono float tensor (T,)
      sr: sample rate (ideally 16000)
      threshold: VAD threshold
      min_speech_duration_ms: drop speech shorter than this
      min_silence_duration_ms: silence required to split segments
      device: optional torch device for model

    Returns:
      List[Segment] in sample indices (start inclusive, end exclusive)
    """
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1D mono tensor, got shape {tuple(audio.shape)}")

    model, utils = load_silero_vad(device=device)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Silero expects float tensor on CPU for get_speech_timestamps;
    # model can be on device, but audio should be CPU tensor.
    audio_cpu = audio.detach().cpu()

    ts = get_speech_timestamps(
        audio_cpu,
        model,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        return_seconds=False,
    )

    return [Segment(d["start"], d["end"]) for d in ts]


# -----------------------------
# Merging logic (your 7s rule)
# -----------------------------
def merge_segments_by_gap(
    segments: List[Segment],
    sr: int,
    gap_s: float = 7.0,
    *,
    pad_ms: int = 0,
    audio_len_samples: Optional[int] = None,
) -> List[Segment]:
    """
    Merge segments into clips: if gap between consecutive segments <= gap_s,
    treat them as one clip.

    Adds optional padding around merged clips and clamps to audio bounds.

    Args:
      segments: speech segments (sample indices)
      sr: sample rate
      gap_s: max allowed silence gap to merge
      pad_ms: pad each side of the merged clip
      audio_len_samples: if provided, clamp to [0, audio_len_samples]

    Returns:
      merged clips as List[Segment]
    """
    if not segments:
        return []

    segs = sorted(segments, key=lambda s: s.start)
    gap_samples = int(round(gap_s * sr))
    pad_samples = int(round((pad_ms / 1000.0) * sr))

    merged: List[Segment] = []
    cur_s, cur_e = segs[0].start, segs[0].end

    for seg in segs[1:]:
        if seg.start - cur_e <= gap_samples:
            cur_e = max(cur_e, seg.end)
        else:
            merged.append(Segment(cur_s, cur_e))
            cur_s, cur_e = seg.start, seg.end

    merged.append(Segment(cur_s, cur_e))

    # Apply padding + clamp
    padded: List[Segment] = []
    for seg in merged:
        s = max(0, seg.start - pad_samples)
        e = seg.end + pad_samples
        if audio_len_samples is not None:
            e = min(audio_len_samples, e)
        if e > s:
            padded.append(Segment(s, e))

    return padded


# -----------------------------
# Splitting + exporting
# -----------------------------
def split_audio_by_segments(
    audio: torch.Tensor,
    segments: List[Segment],
) -> List[torch.Tensor]:
    """
    Slice audio into a list of clip tensors.

    Args:
      audio: mono float tensor (T,)
      segments: sample-index segments

    Returns:
      list of audio clips, each shape (T_clip,)
    """
    if audio.ndim != 1:
        raise ValueError(f"audio must be 1D mono tensor, got shape {tuple(audio.shape)}")

    clips = []
    for seg in segments:
        clips.append(audio[seg.start:seg.end].contiguous())
    return clips


def export_clips(
    clips: List[torch.Tensor],
    sr: int,
    out_dir: Union[str, Path],
    *,
    base_name: str = "audio",
    include_sample_bounds: bool = True,
    subtype: str = "PCM_16",
    segments: Optional[List[Segment]] = None,
) -> List[Path]:
    """
    Save clips to wav files.

    If segments are provided, can include bounds in filenames.

    Returns list of output paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths: List[Path] = []
    for i, clip in enumerate(clips):
        if include_sample_bounds and segments is not None:
            s, e = segments[i].start, segments[i].end
            fname = f"{base_name}_clip_{i:03d}_{s}_{e}.wav"
        else:
            fname = f"{base_name}_clip_{i:03d}.wav"

        out_path = out_dir / fname
        save_wav(clip, sr, out_path, subtype=subtype)
        out_paths.append(out_path)

    return out_paths


# -----------------------------
# One-shot convenience pipeline
# -----------------------------
def vad_split_pipeline(
    audio_path: Union[str, Path],
    *,
    target_sr: int = 16000,
    gap_s: float = 7.0,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 500,
    pad_ms: int = 200,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, int, List[Segment], List[Segment], List[torch.Tensor]]:
    """
    Convenience wrapper:
      load -> silero speech segments -> merge into clips -> slice audio

    Returns:
      audio, sr, speech_segments, merged_clips, clip_tensors
    """
    audio, sr = load_mono_audio(audio_path, target_sr=target_sr)

    speech_segments = get_speech_timestamps_silero(
        audio,
        sr,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        device=device,
    )

    merged_clips = merge_segments_by_gap(
        speech_segments,
        sr,
        gap_s=gap_s,
        pad_ms=pad_ms,
        audio_len_samples=int(audio.numel()),
    )

    clip_tensors = split_audio_by_segments(audio, merged_clips)

    return audio, sr, speech_segments, merged_clips, clip_tensors

import wave


def split_audio_equal_parts_gapless(
    input_path: str | Path,
    output_dir: str | Path,
    num_parts: int,
    target_sample_rate: int = 16000,
    mono: bool = True,
):
    """
    Split audio into `num_parts` parts such that concatenating outputs yields
    EXACTLY the original PCM timeline (no gaps, no overlaps).

    Implementation:
      1) Convert input once to PCM WAV (fixed SR/ch)
      2) Read exact sample count from WAV header (no float duration math)
      3) Cut using sample-index atrim and re-mux as PCM WAV
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_parts <= 0:
        raise ValueError("num_parts must be >= 1")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 1) Convert once to PCM WAV
    tmp_wav = output_dir / f"{input_path.stem}__tmp_pcm.wav"
    cmd_pcm = ["ffmpeg", "-y", "-i", str(input_path)]
    if mono:
        cmd_pcm += ["-ac", "1"]
    if target_sample_rate:
        cmd_pcm += ["-ar", str(target_sample_rate)]
    cmd_pcm += ["-c:a", "pcm_s16le", str(tmp_wav)]
    subprocess.run(cmd_pcm, check=True)

    # 2) Read exact sample count + SR from WAV header (sample-accurate)
    with wave.open(str(tmp_wav), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        total_frames = wf.getnframes()  # frames = samples per channel

    if total_frames <= 0:
        tmp_wav.unlink(missing_ok=True)
        raise ValueError("Audio duration is zero or invalid after PCM conversion.")
    if mono and channels != 1:
        tmp_wav.unlink(missing_ok=True)
        raise ValueError(f"Expected mono WAV but got {channels} channels.")
    if target_sample_rate and sr != target_sample_rate:
        tmp_wav.unlink(missing_ok=True)
        raise ValueError(f"Expected {target_sample_rate} Hz but got {sr} Hz.")

    outputs: list[Path] = []

    # 3) Exact tiling boundaries: [start_i, end_i) partitions [0, total_frames)
    for i in range(num_parts):
        start = (i * total_frames) // num_parts
        end = ((i + 1) * total_frames) // num_parts

        # Guard: if num_parts > total_frames, some segments may be empty
        if end <= start:
            continue

        out = output_dir / f"{input_path.stem}_part_{i:03d}.wav"
        af = f"atrim=start_sample={start}:end_sample={end},asetpts=N/SR/TB"

        cmd_cut = [
            "ffmpeg", "-y",
            "-i", str(tmp_wav),
            "-af", af,
            "-c:a", "pcm_s16le",
            str(out),
        ]
        subprocess.run(cmd_cut, check=True)
        outputs.append(out)

    tmp_wav.unlink(missing_ok=True)
    return outputs



def clip_audio_gapless_with_padding(
    input_path: str | Path,
    output_dir: str | Path,
    start_s: float,
    end_s: float,
    pad_before_s: float = 2.0,
    pad_after_s: float = 2.0,
    target_sample_rate: int = 16000,
    mono: bool = True,
) -> Path:
    """
    Clip [start_s, end_s] from `input_path`, adding padding on both sides,
    WITHOUT going past the audio length, and cut sample-accurately (gapless).

    Strategy (mirrors your gapless splitter):
      1) Convert input once to PCM WAV (fixed SR/ch) to avoid encoder delay / iframe issues
      2) Read exact sample count from WAV header
      3) Convert (seconds -> sample indices), clamp to [0, total_frames]
      4) Cut using atrim with start_sample/end_sample and re-mux as PCM WAV
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not (math.isfinite(start_s) and math.isfinite(end_s)):
        raise ValueError("start_s and end_s must be finite floats.")
    if end_s <= start_s:
        raise ValueError(f"end_s must be > start_s (got start_s={start_s}, end_s={end_s}).")
    if pad_before_s < 0 or pad_after_s < 0:
        raise ValueError("pad_before_s and pad_after_s must be >= 0.")

    # 1) Convert once to PCM WAV (stable timeline, sample-accurate cutting)
    tmp_wav = output_dir / f"{input_path.stem}__tmp_pcm.wav"
    cmd_pcm = ["ffmpeg", "-y", "-i", str(input_path)]
    if mono:
        cmd_pcm += ["-ac", "1"]
    if target_sample_rate:
        cmd_pcm += ["-ar", str(target_sample_rate)]
    cmd_pcm += ["-c:a", "pcm_s16le", str(tmp_wav)]
    subprocess.run(cmd_pcm, check=True)

    try:
        # 2) Read exact sample count + SR from WAV header
        with wave.open(str(tmp_wav), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            total_frames = wf.getnframes()  # frames = samples per channel

        if total_frames <= 0:
            raise ValueError("Audio duration is zero or invalid after PCM conversion.")
        if mono and channels != 1:
            raise ValueError(f"Expected mono WAV but got {channels} channels.")
        if target_sample_rate and sr != target_sample_rate:
            raise ValueError(f"Expected {target_sample_rate} Hz but got {sr} Hz.")

        # 3) Clamp padded seconds to audio bounds, then convert to sample indices
        audio_dur_s = total_frames / sr

        clip_start_s = max(0.0, start_s - pad_before_s)
        clip_end_s = min(audio_dur_s, end_s + pad_after_s)

        if clip_end_s <= clip_start_s:
            raise ValueError(
                f"Clamped clip is empty (clip_start_s={clip_start_s}, clip_end_s={clip_end_s})."
            )

        start_sample = int(round(clip_start_s * sr))
        end_sample = int(round(clip_end_s * sr))

        # Clamp again at sample-level (paranoia)
        start_sample = max(0, min(start_sample, total_frames))
        end_sample = max(0, min(end_sample, total_frames))
        if end_sample <= start_sample:
            raise ValueError(
                f"Clamped sample range is empty (start_sample={start_sample}, end_sample={end_sample})."
            )

        # 4) Sample-accurate cut
        out_path = output_dir / (
            f"{input_path.stem}_clip_{clip_start_s:.2f}s_{clip_end_s:.2f}s.wav"
        )
        af = f"atrim=start_sample={start_sample}:end_sample={end_sample},asetpts=N/SR/TB"

        cmd_cut = [
            "ffmpeg", "-y",
            "-i", str(tmp_wav),
            "-af", af,
            "-c:a", "pcm_s16le",
            str(out_path),
        ]
        subprocess.run(cmd_cut, check=True)

        return out_path

    finally:
        tmp_wav.unlink(missing_ok=True)

def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def best_segment_window_full(
    result,
    fed_transcript: str,
    max_segs: int = 60,
    min_cover: float = 0.70,
):
    """
    Returns (t0, t1, score, i, j, matched_text) for the best contiguous segment range,
    OR None if fed_transcript is empty after normalization.
    """
    fed_n = norm(fed_transcript)
    if not fed_n:
        return None  # <-- key behavior

    fed_tokens = fed_n.split()
    Lf = len(fed_tokens)

    segs = result.segments
    best = (-1e9, None, None, None, None, None)

    for i in range(len(segs)):
        acc = []
        for j in range(i, min(len(segs), i + max_segs)):
            acc.append(segs[j].text)
            cand = norm(" ".join(acc))
            cand_tokens = cand.split()
            Lc = len(cand_tokens)

            if Lc < int(min_cover * Lf):
                continue

            sim = fuzz.token_sort_ratio(cand, fed_n)  # 0..100
            length_penalty = 40.0 * abs(Lc - Lf) / max(1, Lf)
            score = sim - length_penalty

            if score > best[0]:
                best = (score, segs[i].start, segs[j].end, i, j, " ".join(acc))

    score, t0, t1, i, j, matched = best
    if t0 is None or t1 is None:
        return None  # no viable window found
    return (t0, t1, score, i, j, matched)

def _make_anchors(
    fed_n: str,
    anchor_words: int = 7,
    stride: int = 5,
    max_anchors: int = 8,
    min_anchor_words: int = 4,
) -> list[str]:
    """
    Create overlapping word-anchors from normalized fed transcript.
    """
    toks = fed_n.split()
    if len(toks) < min_anchor_words:
        return []

    anchors = []
    # If transcript is short, just one anchor
    if len(toks) <= anchor_words:
        return [" ".join(toks)]

    for start in range(0, len(toks) - anchor_words + 1, stride):
        anchors.append(" ".join(toks[start:start + anchor_words]))
        if len(anchors) >= max_anchors:
            break

    # Also include last window (often contains distinctive item names)
    last = " ".join(toks[-anchor_words:])
    if last not in anchors and len(anchors) < max_anchors:
        anchors.append(last)

    return anchors


def best_segment_window_anchors(
    result,
    fed_transcript: str,
    # anchor generation
    anchor_words: int = 7,
    stride: int = 5,
    max_anchors: int = 8,
    # search over whisper segments
    max_segs_per_anchor: int = 8,     # how many contiguous segments to concatenate while matching an anchor
    # acceptance criteria
    min_anchor_score: float = 72.0,    # raise to be stricter; lower if whisper text is awful
    min_hits: int = 2,                # require at least this many anchors to hit
):
    """
    Anchor-based fallback when full-paragraph matching fails.

    Returns:
      (t0, t1, score, i, j, matched_text)

    Where:
      - t0/t1 are derived from min/max of anchor-hit windows
      - score is the mean of the accepted anchor scores (0..100)
      - i/j are min/max segment indices covered by accepted hits
      - matched_text is the concatenated whisper text from i..j

    OR None if fed_transcript empty OR not enough anchor hits.
    """
    fed_n = norm(fed_transcript)
    if not fed_n:
        return None

    anchors = _make_anchors(
        fed_n,
        anchor_words=anchor_words,
        stride=stride,
        max_anchors=max_anchors,
    )
    if not anchors:
        return None

    segs = result.segments
    if not segs:
        return None

    hits = []  # list of dicts: {score, i, j, start, end, anchor}

    # For each anchor, find best matching contiguous segment window (small)
    for anchor in anchors:
        best = (-1.0, None, None, None, None)  # score, i, j, t0, t1

        for i in range(len(segs)):
            acc = []
            for j in range(i, min(len(segs), i + max_segs_per_anchor)):
                acc.append(segs[j].text)
                cand = norm(" ".join(acc))
                if not cand:
                    continue

                # partial_ratio works well for "anchor inside noisy candidate"
                score = fuzz.partial_ratio(cand, anchor)

                if score > best[0]:
                    best = (score, i, j, segs[i].start, segs[j].end)

        score, i, j, t0, t1 = best
        if i is not None and score >= min_anchor_score:
            hits.append(
                {"score": score, "i": i, "j": j, "t0": t0, "t1": t1, "anchor": anchor}
            )

    if len(hits) < min_hits:
        return None

    # Combine hits into a single window
    t0 = min(h["t0"] for h in hits)
    t1 = max(h["t1"] for h in hits)
    i0 = min(h["i"] for h in hits)
    j0 = max(h["j"] for h in hits)

    matched_text = " ".join(segs[k].text for k in range(i0, j0 + 1))
    avg_score = sum(h["score"] for h in hits) / len(hits)

    return (t0, t1, avg_score, i0, j0, matched_text), hits
