"""Microphone capture at Whisper's 16 kHz mono float32."""

from __future__ import annotations

import threading

import numpy as np
import sounddevice

SAMPLE_RATE = 16000


class Recorder:
    """Records between `start()` and `stop()`.

    The stream is opened per recording, so the mic (and macOS's orange
    indicator) is only on while dictating. `level` is the RMS of the latest
    block, for a level meter.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._blocks: list[np.ndarray] = []
        self._stream: sounddevice.InputStream | None = None
        self.level = 0.0

    def _callback(self, indata, frames, time_info, status) -> None:
        block = indata[:, 0].copy()
        with self._lock:
            self._blocks.append(block)
        self.level = float(np.sqrt(np.mean(block**2))) if len(block) else 0.0

    def start(self) -> None:
        if self._stream is not None:
            return
        with self._lock:
            self._blocks = []
        self._stream = sounddevice.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=self._callback
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop and return the recorded audio (empty if not recording)."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        self.level = 0.0
        with self._lock:
            blocks, self._blocks = self._blocks, []
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)

    @property
    def recording(self) -> bool:
        return self._stream is not None
