import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import download_data
from src.features import create_features
from src.dataset import prepare_dataset
from src.model import QuantPulse
from src.trainer import train
from src.backtest import calculate_metrics

MODEL_PATH = "models/quantpulse_model.pth"

print("Starting training for QuantPulse model...")

# -------------------------------------------------
# DEVICE
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

print("\nDownloading market data...")
data = download_data()

print("Creating features...")
data = create_features(data)

# -------------------------------------------------
# DATASET
# -------------------------------------------------

print("Preparing dataset...")

X_train, X_test, y_train, y_test, tickers_test, dates_test = prepare_dataset(data)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)

# -------------------------------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------------------------------

split = int(len(X_train) * 0.8)

X_tr = X_train[:split]
y_tr = y_train[:split]

X_val = X_train[split:]
y_val = y_train[split:]

print("Training samples:", len(X_tr))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# -------------------------------------------------
# BUILD MODEL
# -------------------------------------------------

print("\nInitializing model...")

model = QuantPulse(input_size=X_train.shape[2])
model.to(device)

# -------------------------------------------------
# LOAD EXISTING MODEL
# -------------------------------------------------

if os.path.exists(MODEL_PATH):
    try:
        print("Loading existing trained model...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError:
        print("Model architecture changed. Starting new training.")

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

print("\nStarting training...")

train(
    model,
    X_tr,
    y_tr,
    X_val,
    y_val,
    epochs=40,
    batch_size=64,
    save_every=20,
    model_path=MODEL_PATH
)

torch.save(model.state_dict(), MODEL_PATH)

print("\nModel saved to", MODEL_PATH)

# -------------------------------------------------
# MODEL PREDICTIONS
# -------------------------------------------------

print("\nRunning predictions...")

model.eval()
X_test = X_test.to(device)

with torch.no_grad():
    outputs = model(X_test)
    probs = torch.softmax(outputs, dim=1)

    confidence = probs[:, 1].cpu().numpy()

# 🔀 RANDOM BASELINE
random_confidence = confidence.copy()
np.random.shuffle(random_confidence)

print("\nFirst 20 predictions:")
print(confidence[:20])

print("\nActual labels:")
print(y_test[:20])

# -------------------------------------------------
# BUILD PORTFOLIO DATA
# -------------------------------------------------

print("\nBuilding portfolio signals...")

# prices = data["Close"].iloc[-len(confidence):].values

# pred_df = pd.DataFrame({
#     "ticker": tickers_test,
#     "confidence": confidence,
#     "random_confidence": random_confidence,
#     "price": prices
# })

pred_df = pd.DataFrame({
    "ticker": tickers_test,
    "date": dates_test,
    "confidence": confidence,
    "random_confidence": random_confidence
})
pred_df["date"] = pd.to_datetime(pred_df["date"])
# -------------------------------------------------
# MERGE WITH TRUE PRICES
# -------------------------------------------------

price_df = data.reset_index()[["Date", "Ticker", "Close"]]
price_df["Date"] = pd.to_datetime(price_df["Date"])

pred_df = pred_df.merge(
    price_df,
    left_on=["date", "ticker"],
    right_on=["Date", "Ticker"],
    how="left"
)

pred_df.rename(columns={"Close": "price"}, inplace=True)
pred_df.drop(columns=["Date", "Ticker"], inplace=True)
pred_df = pred_df.dropna(subset=["price"])

# SORT BY DATE (VERY IMPORTANT)
pred_df = pred_df.sort_values("date").reset_index(drop=True)

# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------

holding = 1
stop_loss = -0.03

# -------------------------------------------------
# BACKTEST FUNCTION
# -------------------------------------------------

def run_backtest(conf_col):

    portfolio_values = []
    capital = 10000

    #groups = list(pred_df.groupby(pred_df.index // group_size))
    groups = list(pred_df.groupby("date"))

    for i in range(len(groups) - holding):

        today = groups[i][1]
        tomorrow = groups[i + holding][1]

        if len(today) == 0:
            portfolio_values.append(capital)
            continue

        top = today.sort_values(conf_col, ascending=False).head(3)

        total_conf = top[conf_col].sum()
        daily_return = 0

        for _, row in top.iterrows():

            ticker = row["ticker"]
            weight = row[conf_col] / total_conf

            today_price = row["price"]

            next_price = tomorrow[
                tomorrow["ticker"] == ticker
            ]["price"].values

            if len(next_price) == 0:
                continue

            transaction_cost = 0.0015
            slippage = 0.0005

            entry_price = today_price * (1 + slippage)
            exit_price = next_price[0] * (1 - slippage)

            r = (exit_price - entry_price) / entry_price

            if r < stop_loss:
                r = stop_loss

            r -= transaction_cost

            daily_return += weight * r

        capital *= (1 + daily_return)
        portfolio_values.append(capital)

    sharpe, max_dd = calculate_metrics(portfolio_values)

    return capital, sharpe, max_dd, portfolio_values

# -------------------------------------------------
# RUN BOTH BACKTESTS
# -------------------------------------------------

print("\nRunning REAL model backtest...")
real_final, real_sharpe, real_dd, real_curve = run_backtest("confidence")

print("\nRunning RANDOM baseline backtest...")
rand_final, rand_sharpe, rand_dd, rand_curve = run_backtest("random_confidence")

# -------------------------------------------------
# RESULTS COMPARISON
# -------------------------------------------------

print("\n==============================")
print("RESULT COMPARISON")
print("==============================")

print("\nREAL MODEL")
print("Final Value:", round(real_final, 2))
print("Sharpe:", round(real_sharpe, 3))
print("Max Drawdown:", round(real_dd * 100, 2), "%")

print("\nRANDOM BASELINE")
print("Final Value:", round(rand_final, 2))
print("Sharpe:", round(rand_sharpe, 3))
print("Max Drawdown:", round(rand_dd * 100, 2), "%")

# -------------------------------------------------
# PLOT EQUITY CURVES
# -------------------------------------------------

plt.figure(figsize=(10,5))
plt.plot(real_curve, label="Real Model")
plt.plot(rand_curve, label="Random Baseline")
plt.title("Equity Curve Comparison")
plt.legend()
plt.grid(True)
plt.savefig("comparison_equity.png")
plt.close()

print("\nComparison chart saved: comparison_equity.png")