import torch
import torch.nn as nn


class CNNBiLSTMFireMap(nn.Module):
    """
    Input:
        x: (B, T, C, H, W)

    Output:
        logits: (B, H, W)

    Arsitektur:
        CNN encoder per timestep
        BiLSTM untuk temporal sequence setiap pixel
        Linear classifier untuk fire probability map
    """

    def __init__(
        self,
        in_channels: int,
        cnn_channels: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.GELU(),

            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.GELU(),

            nn.Dropout2d(dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if lstm_layers == 1 else dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape

        # CNN per timestep
        x = x.reshape(b * t, c, h, w)
        z = self.encoder(x)  # (B*T, E, H, W)

        e = z.shape[1]

        # reshape ke sequence per pixel
        z = z.reshape(b, t, e, h, w)
        z = z.permute(0, 3, 4, 1, 2)  # (B, H, W, T, E)
        z = z.reshape(b * h * w, t, e)  # (B*H*W, T, E)

        lstm_out, _ = self.bilstm(z)

        # ambil representasi timestep terakhir
        last = lstm_out[:, -1, :]  # (B*H*W, 2*lstm_hidden)

        logits = self.classifier(last).squeeze(-1)
        logits = logits.reshape(b, h, w)

        return logits