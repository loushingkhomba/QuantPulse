import torch
import torch.nn as nn


class QuantPulse(nn.Module):

    def __init__(self, input_size, hidden_size=48, num_layers=1, dropout=0.3):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # LSTM dropout works only if layers >1
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, 2)
        )

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        last_output = lstm_out[:, -1, :]

        last_output = self.dropout(last_output)

        return self.fc(last_output)


class QuantPulseMLP(nn.Module):

    def __init__(self, input_size, hidden_size=64, dropout=0.25):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, x):

        # Use only the latest timestep as a compact baseline.
        last_step = x[:, -1, :]
        return self.net(last_step)


class QuantPulseSimple(nn.Module):
    """Ultra-simple model: single hidden layer on last timestep.
    Designed to prevent overfitting and capture only robust signal.
    Includes regime-aware features (nifty_trend, market_volatility) in input.
    """

    def __init__(self, input_size, hidden_size=32, dropout=0.15):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        # Use only the latest timestep.
        last_step = x[:, -1, :]
        return self.net(last_step)