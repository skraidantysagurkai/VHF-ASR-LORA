"""
VHF AM Channel Quality Classifiers  —  GPU-native
===================================================
Three architectures for 5-class degradation classification:
    0 - perfect  |  1 - good  |  2 - okay  |  3 - bad  |  4 - very_bad

"""

import torch
import torch.nn as nn
import torchaudio.transforms as T

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
N_MELS = 128
N_FFT = 1024
WIN_LENGTH = 400  # 25 ms
HOP_LENGTH = 160  # 10 ms
F_MIN = 300.0
F_MAX = 3_400.0
NUM_CLASSES = 4

DEGRADATION_LEVELS = {
    0: "perfect",
    1: "good",
    2: "okay",
    3: "bad",
}

LABEL_TO_IDX = {v: k for k, v in DEGRADATION_LEVELS.items()}

# ---------------------------------------------------------------------------
# Mel frontend  — fully GPU-native via torchaudio MelSpectrogram submodule
# ---------------------------------------------------------------------------


class MelFrontend(nn.Module):
    """
    Waveform (B, samples) -> log-mel spectrogram (B, 1, n_mels, T).

    Uses torchaudio.transforms.MelSpectrogram registered as an nn.Module
    so it moves to GPU with model.to(device) and is included in state_dict.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        win_length: int = WIN_LENGTH,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        f_min: float = F_MIN,
        f_max: float = F_MAX,
    ) -> None:
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel(waveform)  # (B, n_mels, T)
        mel_db = self.amp_to_db(mel)  # (B, n_mels, T)
        return mel_db.unsqueeze(1)  # (B, 1, n_mels, T)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class GateHead(nn.Module):
    """Scalar sigmoid gate g in (0,1):  clean -> 0,  degraded -> 1."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 1)


class ClassifierLoss(nn.Module):
    """
    CrossEntropy (classifier) + MSE (gate supervision).
    Gate target: linearly maps label 0->0.0, 4->1.0.
    Returns (total, ce_loss, gate_loss) for logging.
    """

    def __init__(self, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,  # (B, num_classes)
        labels: torch.Tensor,  # (B,)  long
        gate: torch.Tensor,  # (B, 1)
        lambda_gate: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = self.ce(logits, labels)
        gate_target = labels.float() / (NUM_CLASSES - 1)
        gate_loss = nn.functional.mse_loss(gate.squeeze(-1), gate_target)
        total = ce_loss + lambda_gate * gate_loss
        return total, ce_loss, gate_loss


def _init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, p in m.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)


# ---------------------------------------------------------------------------
# 1.  CNN baseline
# ---------------------------------------------------------------------------


class _CNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNClassifier(nn.Module):
    """
    3x CNNBlock -> GlobalAvgPool -> Dense(256) -> Dense(num_classes)
    ~500K parameters
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.frontend = MelFrontend()
        self.cnn = nn.Sequential(
            _CNNBlock(1, 32),
            _CNNBlock(32, 64),
            _CNNBlock(64, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.gate_head = GateHead(128)
        _init_weights(self)

    def forward(self, waveform: torch.Tensor):
        x = self.frontend(waveform)  # (B,1,n_mels,T)
        x = self.cnn(x)  # (B,128,H,W)
        x = self.gap(x).flatten(1)  # (B,128)
        logits = self.classifier(x)
        gate = self.gate_head(x)
        return logits, torch.softmax(logits, -1), gate


# ---------------------------------------------------------------------------
# 2.  CRNN
# ---------------------------------------------------------------------------


class CRNNClassifier(nn.Module):
    """
    2x Conv2D (freq-pool only) -> BiGRU(256) -> mean-pool -> Dense(num_classes)
    ~1.5M parameters

    MaxPool on frequency axis only (2,1) preserves the time dimension
    so the GRU can model temporal fading and click patterns.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.frontend = MelFrontend()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # freq /2, time unchanged
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # freq /4, time unchanged
        )
        cnn_out_dim = 64 * (N_MELS // 4)  # 2048

        self.gru = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        gru_out = gru_hidden * 2

        self.classifier = nn.Sequential(
            nn.Linear(gru_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.gate_head = GateHead(gru_out)
        _init_weights(self)

    def forward(self, waveform: torch.Tensor):
        x = self.frontend(waveform)  # (B,1,n_mels,T)
        x = self.cnn(x)  # (B,64,n_mels/4,T)

        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)  # (B,T,C,F)
        x = x.reshape(B, T, C * F)  # (B,T,2048)

        x, _ = self.gru(x)  # (B,T,gru_out)
        x = x.mean(dim=1)  # (B,gru_out)

        logits = self.classifier(x)
        gate = self.gate_head(x)
        return logits, torch.softmax(logits, -1), gate


# ---------------------------------------------------------------------------
# 3.  ECAPA-TDNN
# ---------------------------------------------------------------------------


class _Res2Conv1d(nn.Module):
    """Res2Net multi-scale 1-D convolution."""

    def __init__(
        self, channels: int, kernel_size: int, scale: int = 8, dilation: int = 1
    ) -> None:
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        width = channels // scale
        pad = (kernel_size - 1) * dilation // 2
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    width,
                    width,
                    kernel_size,
                    dilation=dilation,
                    padding=pad,
                    bias=False,
                )
                for _ in range(scale - 1)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(self.scale, dim=1)
        out = [chunks[0]]
        y = chunks[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            y = chunks[i + 1] if i == 0 else chunks[i + 1] + y
            y = torch.relu(bn(conv(y)))
            out.append(y)
        return torch.cat(out, dim=1)


class _SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).unsqueeze(-1)


class _SE_Res2Block(nn.Module):
    """Conv1d -> Res2Conv1d -> Conv1d -> SE + residual."""

    def __init__(
        self, channels: int, scale: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.res2 = _Res2Conv1d(channels, kernel_size, scale, dilation)
        self.conv2 = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = _SEBlock1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.res2(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        return self.se(x) + r


class _AttentiveStatPool(nn.Module):
    """Weighted mean + weighted std -> (B, 2*C)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, 128, 1),
            nn.Tanh(),
            nn.Conv1d(128, channels, 1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.attn(x)
        mean = (w * x).sum(-1)
        std = (w * (x - mean.unsqueeze(-1)) ** 2).sum(-1).clamp(1e-8).sqrt()
        return torch.cat([mean, std], dim=-1)


class ECAPAClassifier(nn.Module):
    """
    MelFrontend -> Conv1d projection -> 3x SE-Res2Block -> cat + Conv1d
    -> AttentiveStatPool -> Dense(num_classes)
    ~6M parameters
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        channels: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.frontend = MelFrontend()

        self.input_proj = nn.Sequential(
            nn.Conv1d(N_MELS, channels, 5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _SE_Res2Block(channels, scale=8, kernel_size=3, dilation=2)
        self.layer2 = _SE_Res2Block(channels, scale=8, kernel_size=3, dilation=3)
        self.layer3 = _SE_Res2Block(channels, scale=8, kernel_size=3, dilation=4)

        self.cat_conv = nn.Sequential(
            nn.Conv1d(channels * 3, channels * 3, 1, bias=False),
            nn.BatchNorm1d(channels * 3),
            nn.ReLU(inplace=True),
        )
        self.pool = _AttentiveStatPool(channels * 3)  # -> (B, channels*6)

        embed_dim = channels * 6
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.gate_head = GateHead(embed_dim)
        _init_weights(self)

    def forward(self, waveform: torch.Tensor):
        x = self.frontend(waveform).squeeze(1)  # (B,n_mels,T)
        x = self.input_proj(x)  # (B,C,T)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        x = self.cat_conv(torch.cat([l1, l2, l3], dim=1))
        x = self.pool(x)  # (B, C*6)
        logits = self.classifier(x)
        gate = self.gate_head(x)
        return logits, torch.softmax(logits, -1), gate


# ---------------------------------------------------------------------------
# Shared inference helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict(model: nn.Module, waveform: torch.Tensor) -> dict:
    model.eval()
    logits, probs, gate = model(waveform)
    idx = probs.argmax(-1)
    names = [DEGRADATION_LEVELS[i.item()] for i in idx]
    return {"label_idx": idx, "label_name": names, "probs": probs, "gate": gate}
