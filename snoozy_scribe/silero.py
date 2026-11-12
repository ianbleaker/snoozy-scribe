"""Torch-free Silero VAD: the batched ("sequence") model run in numpy.

Vendored from Silero VAD 6.2.3 (https://github.com/snakers4/silero-vad),
MIT-licensed; the license is at snoozy_scribe/vendor/silero_vad/LICENSE.

The weights in `vendor/silero_vad/silero_vad_16k_sequence.npz` are the
initializers of upstream's `silero_vad_16k_sequence.onnx` (sha256
9ccdacc4719d8aa7e45a77536bfabec45a03ba1f2fad5e241ab4060b24238a85), unchanged
except that the LSTM's `W`/`R`/`B` are renamed `lstm.W`/`lstm.R`/`lstm.B`.
`SileroVADSequence` re-implements that ONNX graph in numpy: per-frame STFT
magnitude, four Conv1d+ReLU encoder layers, an LSTM across frames, then a
1x1 conv and sigmoid.

Copied from upstream:

- `frame_blocks` (silero_vad/sequence_vad.py), verbatim.
- The `audio_forward` block loop (silero_vad/sequence_vad.py), as
  `speech_probs`, fixed to 16kHz and with a per-block progress callback added.
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

MODEL_PATH = Path(__file__).resolve().parent / "vendor" / "silero_vad" / "silero_vad_16k_sequence.npz"
SAMPLING_RATE = 16000
FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64
# Max frames per block. 512 frames ~= 16.4 s at 16 kHz.
MAX_FRAMES = 512
HIDDEN_SIZE = 128
# STFT: reflect-pad each 576-sample frame by 128 each side, then a 256-tap
# conv with stride 128 whose 258 output channels are 129 real then 129
# imaginary bins. The first of its 5 time steps is dropped.
STFT_PAD = 128
STFT_HOP = 128
STFT_BINS = 129
# (stride, padding) of encoder.0-3; together they reduce 4 STFT steps to 1.
ENCODER_LAYERS = ((1, 1), (2, 1), (2, 1), (1, 1))


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


def _conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None,
            stride: int, padding: int) -> np.ndarray:
    """ONNX/torch Conv1d (no dilation or groups): `x` is (N, C, T), `weight` (O, C, K)."""
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
    kernel = weight.shape[2]
    steps = (x.shape[2] - kernel) // stride + 1
    # (N, C, steps, K): every receptive field, gathered per kernel tap.
    windows = np.stack(
        [x[:, :, k: k + stride * (steps - 1) + 1: stride] for k in range(kernel)],
        axis=-1,
    )
    out = np.einsum("nctk,ock->not", windows, weight, optimize=True)
    if bias is not None:
        out += bias[None, :, None]
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SileroVADSequence:
    """Loads the sequence model's weights and computes per-frame speech probabilities."""

    def __init__(self, path: str | Path = MODEL_PATH):
        with np.load(path, allow_pickle=False) as weights:
            self.weights = {name: weights[name].astype(np.float32) for name in weights.files}
        w = self.weights
        # ONNX LSTM layout: gates stacked i, o, f, c; B is input bias then recurrent bias.
        self._lstm_input = np.ascontiguousarray(w["lstm.W"][0].T)
        self._lstm_recurrent = np.ascontiguousarray(w["lstm.R"][0].T)
        self._lstm_bias = w["lstm.B"][0, : 4 * HIDDEN_SIZE] + w["lstm.B"][0, 4 * HIDDEN_SIZE:]
        self._output_weight = w["output.weight"][0, :, 0]
        self._output_bias = w["output.bias"][0]

    def _encode(self, block: np.ndarray) -> np.ndarray:
        """(frames, 576) audio block -> (frames, 128) encoder features."""
        padded = np.pad(block, ((0, 0), (STFT_PAD, STFT_PAD)), mode="reflect")
        spectrum = _conv1d(padded[:, None, :], self.weights["stft.forward_basis_buffer"],
                           None, STFT_HOP, 0)
        real = spectrum[:, :STFT_BINS, 1:]
        imaginary = spectrum[:, STFT_BINS:, 1:]
        x = np.sqrt(real ** 2 + imaginary ** 2)
        for i, (stride, padding) in enumerate(ENCODER_LAYERS):
            x = _conv1d(x, self.weights[f"encoder.{i}.weight"],
                        self.weights[f"encoder.{i}.bias"], stride, padding)
            np.maximum(x, 0, out=x)
        return x[:, :, 0]

    def _lstm(self, features: np.ndarray, hidden: np.ndarray, cell: np.ndarray):
        """Run the LSTM across frames; returns (outputs, hidden, cell)."""
        h = HIDDEN_SIZE
        input_gates = features @ self._lstm_input + self._lstm_bias
        outputs = np.empty((features.shape[0], h), dtype=np.float32)
        for t in range(features.shape[0]):
            gates = input_gates[t] + hidden @ self._lstm_recurrent
            input_gate = _sigmoid(gates[:h])
            output_gate = _sigmoid(gates[h: 2 * h])
            forget_gate = _sigmoid(gates[2 * h: 3 * h])
            candidate = np.tanh(gates[3 * h:])
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * np.tanh(cell)
            outputs[t] = hidden
        return outputs, hidden, cell

    def speech_probs(
        self,
        audio: np.ndarray,
        progress_callback: Callable[[float], None] | None = None,
    ) -> np.ndarray:
        """Return a 1-D float32 array of per-frame speech probabilities.

        Runs the model one block of up to `MAX_FRAMES` frames at a time,
        carrying LSTM state across blocks. `progress_callback`, if given, is
        called after each block with the cumulative percent (0-100) of frames
        processed.
        """
        frame_count = (audio.size + FRAME_SAMPLES - 1) // FRAME_SAMPLES
        hidden = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        cell = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        probabilities = []
        frames_done = 0
        for block in frame_blocks(audio, FRAME_SAMPLES, CONTEXT_SAMPLES, MAX_FRAMES):
            outputs, hidden, cell = self._lstm(self._encode(block), hidden, cell)
            logits = np.maximum(outputs, 0) @ self._output_weight + self._output_bias
            probabilities.append(_sigmoid(logits).astype(np.float32))
            frames_done += block.shape[0]
            if progress_callback is not None and frame_count:
                progress_callback(min(frames_done, frame_count) / frame_count * 100)
        if not probabilities:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(probabilities)


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
