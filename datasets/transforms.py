"""Shared audio transforms; currently populated with the ESC-50 pipeline."""

import torch
import torch.nn.functional as F
import torchaudio


class LogMelTransform:
    """Log-Mel transform matching the supplied ESC-50 notebook."""

    def __init__(
        self,
        sample_rate=32000,
        clip_duration_seconds=5,
        n_fft=4096,
        win_length=4096,
        hop_length=502,
        n_mels=256,
        power=2.0,
        top_db=80.0,
        center=True,
        pad_mode="reflect",
        mel_scale="htk",
        emulate_float16_cache=True,
        device=None,
    ):
        self.sample_rate = int(sample_rate)
        self.target_num_samples = int(sample_rate * clip_duration_seconds)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.mel_frames = self.target_num_samples // self.hop_length + 1
        self.emulate_float16_cache = bool(emulate_float16_cache)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            center=center,
            pad_mode=pad_mode,
            norm=None,
            mel_scale=mel_scale,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=top_db
        )
        if device is not None:
            self.to(device)

    def to(self, device):
        self.mel_transform = self.mel_transform.to(device)
        self.db_transform = self.db_transform.to(device)
        return self

    def __call__(self, waveforms):
        mels = self.db_transform(self.mel_transform(waveforms))

        if mels.shape[-1] < self.mel_frames:
            mels = F.pad(mels, (0, self.mel_frames - mels.shape[-1]))
        elif mels.shape[-1] > self.mel_frames:
            mels = mels[..., : self.mel_frames]

        expected_shape = (waveforms.shape[0], 1, self.n_mels, self.mel_frames)
        if tuple(mels.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected mel shape {tuple(mels.shape)}; expected {expected_shape}."
            )

        if self.emulate_float16_cache:
            # Training cached NumPy float16 log-Mels, then converted each sample
            # back to float32 before model input. This reproduces that quantization.
            mels = mels.to(torch.float16).to(torch.float32)
        return mels
