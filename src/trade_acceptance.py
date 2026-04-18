import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TradeAcceptanceMLP(nn.Module):
	"""Binary gate model for second-stage trade acceptance."""

	def __init__(self, input_size, hidden_size=64, dropout=0.1):
		super().__init__()

		self.net = nn.Sequential(
			nn.Linear(input_size, hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, hidden_size // 2),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size // 2, 1),
		)

	def forward(self, x):
		return self.net(x).squeeze(-1)


ACCEPTANCE_FEATURE_COLUMNS = [
	"confidence",
	"rank_pct",
	"signal_spread_raw",
	"regime_state",
	"volatility_regime",
	"nifty_drawdown_63d",
	"trend_strength",
]


def _prepare_features(frame, feature_cols):
	feature_df = frame[feature_cols].copy()
	feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
	feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
	feature_df = feature_df.fillna(0.0)
	return feature_df.astype(np.float32)


def prepare_acceptance_frame(frame, signal_col="confidence"):
	enriched = frame.copy()
	enriched["date"] = pd.to_datetime(enriched["date"])
	enriched["rank_pct"] = enriched.groupby("date")[signal_col].rank(pct=True, method="first")
	enriched["signal_spread_raw"] = (
		enriched.groupby("date")[signal_col].transform("max")
		- enriched.groupby("date")[signal_col].transform("min")
	)

	label_col = "future_alpha_5d"
	if label_col not in enriched.columns:
		label_col = "future_alpha_3d" if "future_alpha_3d" in enriched.columns else "future_alpha_1d"
	enriched["accept_label"] = np.where(enriched[label_col].notna(), (enriched[label_col] > 0).astype(float), np.nan)
	return enriched


def fit_trade_acceptance_model(
	frame,
	cutoff_date,
	feature_cols=None,
	epochs=25,
	batch_size=256,
	min_rows=800,
	lr=1e-3,
	weight_decay=1e-4,
	seed=42,
	hidden_size=None,
):
	if hidden_size is None:
		hidden_size = int(os.getenv("QUANT_REGIME_HIDDEN_SIZE", "64"))
	hidden_size = max(8, int(hidden_size))

	feature_cols = feature_cols or ACCEPTANCE_FEATURE_COLUMNS
	prepared = prepare_acceptance_frame(frame)
	prepared = prepared.sort_values("date").reset_index(drop=True)

	train_df = prepared[(prepared["date"] < pd.Timestamp(cutoff_date)) & prepared["accept_label"].notna()].copy()
	if len(train_df) < min_rows:
		return {
			"trained": False,
			"reason": f"insufficient_rows:{len(train_df)}",
			"feature_cols": feature_cols,
			"threshold": 0.5,
			"probs": np.full(len(prepared), 1.0, dtype=np.float32),
		}

	train_dates = np.sort(train_df["date"].unique())
	split_idx = max(1, int(len(train_dates) * 0.8))
	split_date = train_dates[split_idx - 1]

	tr_df = train_df[train_df["date"] <= split_date]
	val_df = train_df[train_df["date"] > split_date]
	if val_df.empty:
		val_df = tr_df.tail(max(100, len(tr_df) // 8)).copy()

	X_train_raw = _prepare_features(tr_df, feature_cols)
	X_val_raw = _prepare_features(val_df, feature_cols)
	X_all_raw = _prepare_features(prepared, feature_cols)
	y_train = tr_df["accept_label"].astype(np.float32).values
	y_val = val_df["accept_label"].astype(np.float32).values

	feat_mean = X_train_raw.mean(axis=0).values.astype(np.float32)
	feat_std = X_train_raw.std(axis=0).replace(0, 1.0).values.astype(np.float32)

	X_train = ((X_train_raw.values - feat_mean) / feat_std).astype(np.float32)
	X_val = ((X_val_raw.values - feat_mean) / feat_std).astype(np.float32)
	X_all = ((X_all_raw.values - feat_mean) / feat_std).astype(np.float32)

	torch.manual_seed(seed)
	np.random.seed(seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = TradeAcceptanceMLP(input_size=X_train.shape[1], hidden_size=hidden_size).to(device)
	train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
	val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

	pos_frac = float(np.clip(y_train.mean(), 1e-4, 1 - 1e-4))
	pos_weight = torch.tensor((1 - pos_frac) / pos_frac, device=device)
	criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

	best_state = None
	best_val_loss = float("inf")
	for _ in range(max(1, int(epochs))):
		model.train()
		for xb, yb in train_loader:
			xb = xb.to(device)
			yb = yb.to(device)
			optimizer.zero_grad()
			logits = model(xb)
			loss = criterion(logits, yb)
			loss.backward()
			optimizer.step()

		model.eval()
		total_val_loss = 0.0
		total_val_count = 0
		with torch.no_grad():
			for xb, yb in val_loader:
				xb = xb.to(device)
				yb = yb.to(device)
				logits = model(xb)
				val_loss = criterion(logits, yb)
				total_val_loss += float(val_loss.item()) * len(yb)
				total_val_count += len(yb)
		avg_val_loss = total_val_loss / max(total_val_count, 1)
		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

	if best_state is not None:
		model.load_state_dict(best_state)

	model.eval()
	with torch.no_grad():
		logits = model(torch.tensor(X_all, device=device))
		probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

	return {
		"trained": True,
		"reason": "ok",
		"feature_cols": feature_cols,
		"hidden_size": int(hidden_size),
		"threshold": float(np.clip(np.quantile(probs[prepared["date"] < pd.Timestamp(cutoff_date)], 0.35), 0.45, 0.70)),
		"probs": probs,
		"val_loss": float(best_val_loss),
		"train_rows": int(len(tr_df)),
		"val_rows": int(len(val_df)),
	}
