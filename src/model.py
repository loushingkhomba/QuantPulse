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