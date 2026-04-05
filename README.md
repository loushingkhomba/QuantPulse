# QuantPulse

QuantPulse is a machine learning-based stock trading project that trains a sequence model on historical market data, generates daily stock confidence scores, and evaluates a top-k portfolio strategy against a random baseline.

## Features

- Downloads market data with `yfinance`
- Builds technical indicators and a 5-day forward target
- Trains a QuantPulse classification model
- Runs a simple backtest and compares equity curves
- Saves charts and model checkpoints locally

## Project Structure

- `train.py` - end-to-end training and backtesting script
- `src/data_loader.py` - downloads market and index data
- `src/features.py` - creates technical indicators and labels
- `src/dataset.py` - prepares sequences for model training
- `src/model.py` - model architecture
- `src/trainer.py` - training loop
- `src/backtest.py` - performance metrics and backtest helpers
- `src/walkforward.py` - walk-forward evaluation utilities

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py
```

## Notes

- The project uses a top-k selection strategy for backtesting.
- Generated files such as model checkpoints, logs, and charts are ignored by Git.
