"""
Audio processor: transcribes audio/video files using Whisper.
Decodes audio with imageio-ffmpeg (bundled binary, no system install needed)
and passes a numpy array directly to Whisper — completely bypasses
Whisper's internal ffmpeg subprocess call that fails on Windows.
"""
from __future__ import annotations
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def load_audio_array(file_path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Decode any audio/video file to a 16 kHz mono float32 numpy array
    using the ffmpeg binary bundled inside imageio-ffmpeg.
    No system ffmpeg install required.
    """
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-threads", "0",
        "-i", file_path,
        "-f", "s16le",       # 16-bit PCM little-endian
        "-ac", "1",           # mono
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-",                  # output to stdout
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    return audio / 32768.0   # normalise to [-1, 1]


@dataclass
class AudioChunk:
    text: str
    start_time: float
    end_time: float
    source: str
    chunk_index: int
    modality: str = "audio"
    metadata: dict = field(default_factory=dict)


class AudioProcessor:
    SUPPORTED_FORMATS = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

    def __init__(
        self,
        whisper_model: str = "base",
        chunk_duration_secs: float = 30.0,
        overlap_secs: float = 5.0,
    ):
        self.whisper_model_name = whisper_model
        self.chunk_duration = chunk_duration_secs
        self.overlap = overlap_secs
        self._model = None

    def _load_model(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.whisper_model_name)
        return self._model

    def transcribe_full(self, audio_path: str | Path) -> dict:
        import torch
        # Decode audio ourselves — avoids Whisper's internal ffmpeg subprocess
        audio_array = load_audio_array(str(audio_path))
        model = self._load_model()
        # Pass numpy array directly — Whisper accepts this and skips ffmpeg
        result = model.transcribe(
            audio_array,
            fp16=torch.cuda.is_available(),
            verbose=False,
        )
        return result

    def process(self, audio_path: str | Path) -> list[AudioChunk]:
        path = Path(audio_path)
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {path.suffix}")

        result = self.transcribe_full(path)
        segments = result.get("segments", [])

        if not segments:
            return [AudioChunk(
                text=result["text"].strip(),
                start_time=0.0,
                end_time=0.0,
                source=str(path),
                chunk_index=0,
                metadata={"language": result.get("language", "en"),
                          "full_transcription": True},
            )]

        chunks: list[AudioChunk] = []
        current_segs: list[dict] = []
        chunk_start = segments[0]["start"]
        chunk_idx = 0

        for seg in segments:
            current_segs.append(seg)
            if seg["end"] - chunk_start >= self.chunk_duration:
                text = " ".join(s["text"].strip() for s in current_segs)
                chunks.append(AudioChunk(
                    text=text,
                    start_time=chunk_start,
                    end_time=seg["end"],
                    source=str(path),
                    chunk_index=chunk_idx,
                    metadata={"language": result.get("language", "en"),
                              "start_time": chunk_start,
                              "end_time": seg["end"]},
                ))
                chunk_idx += 1
                overlap_start = seg["end"] - self.overlap
                current_segs = [s for s in current_segs if s["end"] > overlap_start]
                chunk_start = current_segs[0]["start"] if current_segs else seg["end"]

        if current_segs:
            text = " ".join(s["text"].strip() for s in current_segs)
            chunks.append(AudioChunk(
                text=text,
                start_time=chunk_start,
                end_time=current_segs[-1]["end"],
                source=str(path),
                chunk_index=chunk_idx,
                metadata={"language": result.get("language", "en"),
                          "start_time": chunk_start,
                          "end_time": current_segs[-1]["end"]},
            ))

        return chunks
