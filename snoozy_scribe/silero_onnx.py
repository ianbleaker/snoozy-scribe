"""Torch-free Silero VAD: the batched ("sequence") ONNX model run through onnxruntime.

Vendored from Silero VAD 6.2.3 (https://github.com/snakers4/silero-vad),
MIT-licensed; the license is at snoozy_scribe/vendor/silero_vad/LICENSE. Copied from
upstream:

- `frame_blocks` (silero_vad/sequence_vad.py), verbatim.
- `SileroVADSequence` session setup and `audio_forward` block loop
  (silero_vad/sequence_vad.py), as `speech_probs`, fixed to 16kHz and with a
  per-block progress callback added.
- `get_speech_timestamps_from_probs` (silero_vad/utils_vad.py), verbatim
  minus the matplotlib `visualize_probs` option.

The `silero_vad` package itself can't be imported without torch, which is why
this code is vendored rather than imported.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent / "vendor" / "silero_vad" / "silero_vad_16k_sequence.onnx"
SAMPLING_RATE = 16000
FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
# Max frames per ONNX call. 512 frames ~= 16.4 s at 16 kHz.
MAX_FRAMES = 512


def frame_blocks(audio: np.ndarray,
                 frame_samples: int,
                 context_samples: int,
                 max_frames: int):
    """Yield blocks of shape (block_frames, context_samples + frame_samples).

    Each frame is prefixed with the trailing ``context_samples`` of the previous
    frame (zeros for the very first frame), matching the streaming model's
    internal context handling. The final frame is zero-padded to a full frame.
    """
    frame_count = (audio.size + frame_samples - 1) // frame_samples
    previous_context = np.zeros(context_samples, dtype=np.float32)
    for first_frame in range(0, frame_count, max_frames):
        block_frames = min(max_frames, frame_count - first_frame)
        first_sample = first_frame * frame_samples
        samples = audio[
            first_sample: min(audio.size, (first_frame + block_frames) * frame_samples)
        ]
        frames = np.zeros((block_frames, frame_samples), dtype=np.float32)
        frames.reshape(-1)[: samples.size] = samples

        contexts = np.empty((block_frames, context_samples), dtype=np.float32)
        contexts[0] = previous_context
        if block_frames > 1:
            contexts[1:] = frames[:-1, -context_samples:]
        previous_context = frames[-1, -context_samples:].copy()
        yield np.concatenate((contexts, frames), axis=1)


class SileroVADSequence:
    """Loads the sequence ONNX model and computes per-frame speech probabilities."""

    def __init__(self, path: str | Path = MODEL_PATH):
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def speech_probs(
        self,
        audio: np.ndarray,
        progress_callback: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        """Return a 1-D float32 array of per-frame speech probabilities.

        Runs the model as a single ONNX call per block of up to `MAX_FRAMES`
        frames. `progress_callback`, if given, is called after each block with
        the cumulative percent (0-100) of frames processed.
        """
        frame_count = (audio.size + FRAME_SAMPLES - 1) // FRAME_SAMPLES
        hidden = np.zeros((1, 1, 128), dtype=np.float32)
        cell = np.zeros((1, 1, 128), dtype=np.float32)
        probabilities = []
        frames_done = 0
        for block in frame_blocks(audio, FRAME_SAMPLES, CONTEXT_SAMPLES, MAX_FRAMES):
            values, hidden, cell = self.session.run(
                ["speech_probs", "hn", "cn"],
                {"input": block, "h": hidden, "c": cell},
            )
            probabilities.append(values)
            frames_done += block.shape[0]
            if progress_callback is not None and frame_count:
                progress_callback(min(frames_done, frame_count) / frame_count * 100)
        if not probabilities:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(probabilities).reshape(-1)


def get_speech_timestamps_from_probs(speech_probs: List[float],
                                     sampling_rate: int = 16000,
                                     threshold: float = 0.5,
                                     min_speech_duration_ms: int = 250,
                                     max_speech_duration_s: float = float('inf'),
                                     min_silence_duration_ms: int = 100,
                                     speech_pad_ms: int = 30,
                                     return_seconds: bool = False,
                                     time_resolution: int = 1,
                                     neg_threshold: float = None,
                                     min_silence_at_max_speech: int = 98,
                                     use_max_poss_sil_at_max_speech: bool = True,
                                     audio_length_samples: int = None,
                                     step: int = 1):
    """
    Convert a sequence of per-frame speech probabilities into speech timestamps.

    Parameters
    ----------
    speech_probs: List[float]
        Per-frame speech probabilities, one per window_size_samples frame,
        in order. window_size_samples is 512 for 16000 Hz and 256 for 8000 Hz.

    audio_length_samples: int (default - None)
        Total number of samples in the original audio (at ``sampling_rate``).
        If None, it is estimated as len(speech_probs) * window_size_samples.

    step: int (default - 1)
        Decimation step applied to the input audio before framing (used when
        the caller down-sampled a multiple-of-16000 rate to 16000). Output
        sample coordinates are multiplied by ``step`` to map back.

    All other parameters match ``get_speech_timestamps``.

    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """
    window_size_samples = 512 if sampling_rate == 16000 else 256
    if audio_length_samples is None:
        audio_length_samples = len(speech_probs) * window_size_samples

    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000

    triggered = False
    speeches = []
    current_speech = {}

    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    temp_end = 0  # to save potential segment end (and tolerate some silence)
    prev_end = next_start = 0  # to save potential segment limits in case of maximum segment size reached
    possible_ends = []

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        # If speech returns after a temp_end, record candidate silence if long enough and clear temp_end
        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        # Start of speech
        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = cur_sample
            continue

        # Max speech length reached: decide where to cut
        if triggered and (cur_sample - current_speech['start'] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(possible_ends, key=lambda x: x[1])  # use the longest possible silence segment in the current speech chunk
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur

                if next_start < prev_end + cur_sample:  # previously reached silence (< neg_thres) and is still not speech (< thres)
                    current_speech['start'] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                # Legacy max-speech cut (use_max_poss_sil_at_max_speech=False): prefer last valid silence (prev_end) if available
                if prev_end:
                    current_speech['end'] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech['start'] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    # No prev_end -> fallback to cutting at current sample
                    current_speech['end'] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        # Silence detection while in speech
        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end

            if not use_max_poss_sil_at_max_speech and sil_dur_now > min_silence_samples_at_max_speech:  # condition to avoid cutting in very short silence
                prev_end = temp_end

            if sil_dur_now < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    if return_seconds:
        audio_length_seconds = audio_length_samples / sampling_rate
        for speech_dict in speeches:
            speech_dict['start'] = max(round(speech_dict['start'] / sampling_rate, time_resolution), 0)
            speech_dict['end'] = min(round(speech_dict['end'] / sampling_rate, time_resolution), audio_length_seconds)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict['start'] *= step
            speech_dict['end'] *= step

    return speeches


def get_speech_timestamps(
    audio: np.ndarray,
    model: SileroVADSequence,
    *,
    min_silence_duration_ms: int,
    min_speech_duration_ms: int,
    progress_callback: Callable[[float], None] | None = None,
) -> list[dict]:
    """Speech timestamps in seconds for 16kHz mono float32 `audio`.

    Equivalent to upstream's `get_speech_timestamps_sequence(...,
    return_seconds=True)` with every other parameter at its default.
    """
    probs = model.speech_probs(audio, progress_callback=progress_callback)
    return get_speech_timestamps_from_probs(
        probs,
        sampling_rate=SAMPLING_RATE,
        min_silence_duration_ms=min_silence_duration_ms,
        min_speech_duration_ms=min_speech_duration_ms,
        return_seconds=True,
        audio_length_samples=audio.size,
    )
