"""RMVPE f0 추정기 — DeepUnet + BiGRU 기반, 노래 보컬 특화 SOTA 피치 트래커.

아키텍처는 RVC-Project의 Retrieval-based-Voice-Conversion-WebUI
(`infer/lib/rmvpe.py`, MIT License, https://github.com/RVC-Project/
Retrieval-based-Voice-Conversion-WebUI) 를 포팅했다. 원 모델 자체는
Dream-High/RMVPE(Apache-2.0)에서 학습된 것이며, 배포 가중치는 HuggingFace
`lj1995/VoiceConversionWebUI` 저장소의 `rmvpe.pt`가 RVC 생태계 사실상 표준이다.

원본 대비 축소한 부분 (이 프로젝트는 CUDA/CPU만 지원하면 되므로):
  - DirectML(privateuseone)/ONNX 추론 경로 제거
  - torch.jit 컴파일 경로 제거 (Intel XPU용 ipex 포함)
  - STFT를 1D conv로 구현한 폴백 클래스 제거 (DirectML 전용이었음) —
    표준 torch.stft를 사용.

가중치의 state_dict 키(unet.encoder.*, cnn.*, fc.0.gru.*, fc.1.*)는 원본과
동일해야 `rmvpe.pt`를 그대로 로드할 수 있으므로 클래스 구조는 그대로 유지한다.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — PyTorch 표준 관례
from librosa.filters import mel as librosa_mel_fn

logger = logging.getLogger(__name__)

RMVPE_SAMPLE_RATE = 16000
RMVPE_HOP_LENGTH = 160  # 16000 / 160 = 10ms — extractor의 FCPE 10ms hop과 동일 규격


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum)
            )
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor):
        concat_tensors: list[torch.Tensor] = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList([ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)])
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, (3, 3), stride, (1, 1),
                output_padding=out_padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: list[torch.Tensor]):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    """DeepUnet → 3ch conv → BiGRU → 360-bin cent salience. rmvpe.pt와 1:1 매칭."""

    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, n_gru),
            nn.Linear(512, 360),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


class MelSpectrogram(nn.Module):
    """RMVPE 학습 시 사용한 mel 스펙트로그램 (128 mel, 30~8000Hz, win=1024, hop=160)."""

    def __init__(self, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-5):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp
        self._hann_window: dict[str, torch.Tensor] = {}

    def forward(self, audio: torch.Tensor, center: bool = True) -> torch.Tensor:
        key = str(audio.device)
        if key not in self._hann_window:
            self._hann_window[key] = torch.hann_window(self.win_length).to(audio.device)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._hann_window[key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class RMVPEPredictor:
    """rmvpe.pt를 로드해 파형 → 10ms hop f0(Hz) 배열을 뽑는 래퍼.

    torchfcpe의 spawn_bundled_infer_model()이 반환하는 모델과 유사한 최소
    인터페이스(infer)를 제공해 extractor.py에서 백엔드를 교체하기 쉽게 한다.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.mel_extractor = MelSpectrogram(128, RMVPE_SAMPLE_RATE, 1024, RMVPE_HOP_LENGTH, None, 30, 8000).to(self.device)
        self.model = E2E(4, 1, (2, 2))
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = self.model.to(self.device)
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def _mel_to_hidden(self, mel: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            hidden = self.model(mel.float())
        return hidden[:, :n_frames].squeeze(0).cpu().numpy()

    def _to_local_average_cents(self, salience: np.ndarray, threshold: float) -> np.ndarray:
        """360-bin salience → 가중평균 cent, 최댓값이 threshold 이하면 무성(0)."""
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        starts = center - 4
        ends = center + 5
        todo_salience = np.stack([salience[idx, starts[idx] : ends[idx]] for idx in range(salience.shape[0])])
        todo_cents = np.stack([self.cents_mapping[starts[idx] : ends[idx]] for idx in range(salience.shape[0])])
        product_sum = np.sum(todo_salience * todo_cents, axis=1)
        weight_sum = np.sum(todo_salience, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            devided = product_sum / weight_sum
        maxx = np.max(salience, axis=1)
        devided[maxx <= threshold] = 0
        return devided

    def infer(self, waveform: np.ndarray, threshold: float = 0.03) -> np.ndarray:
        """16kHz mono 파형(1D np.ndarray) → 10ms hop f0(Hz) 배열 (unvoiced=0)."""
        audio_t = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).to(self.device)
        mel = self.mel_extractor(audio_t.unsqueeze(0), center=True)
        hidden = self._mel_to_hidden(mel)
        cents = self._to_local_average_cents(hidden, threshold=threshold)
        f0 = 10 * (2 ** (cents / 1200))
        f0[f0 == 10] = 0  # cents==0(무성) → 10*2^0=10Hz는 실질 무성 표식
        return f0.astype(np.float64)


def is_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True
