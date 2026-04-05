import pandas as pd
import torch
from trainer import train
from dataset import create_sequences
from backtest import run_backtest, calculate_metrics


def walk_forward_validation(
    data,
    model,
    train_years=4,
    test_years=1,
    sequence_length=20
):

    years = sorted(data.index.year.unique())

    results = []

    for i in range(train_years, len(years) - test_years + 1):

        train_start = years[0]
        train_end = years[i - 1]

        test_start = years[i]
        test_end = years[i + test_years - 1]

        print("\n==============================")
        print(f"Train: {train_start}-{train_end}")
        print(f"Test : {test_start}-{test_end}")
        print("==============================")

        train_data = data[
            (data.index.year >= train_start)
            & (data.index.year <= train_end)
        ]

        test_data = data[
            (data.index.year >= test_start)
            & (data.index.year <= test_end)
        ]

        # -------------------------
        # Create training sequences
        # -------------------------

        X_train, y_train, _ = create_sequences(train_data, sequence_length)

        # validation split
        split = int(len(X_train) * 0.8)

        X_tr = X_train[:split]
        y_tr = y_train[:split]

        X_val = X_train[split:]
        y_val = y_train[split:]

        # -------------------------
        # Train model
        # -------------------------

        train(
            model,
            X_tr,
            y_tr,
            X_val,
            y_val,
            epochs=500,
            model_path="walkforward_model.pth"
        )

        # -------------------------
        # Create test sequences
        # -------------------------

        X_test, y_test, prices = create_sequences(test_data, sequence_length)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        with torch.no_grad():

            X_test = X_test.to(device)

            outputs = model(X_test)

            predictions = torch.softmax(outputs, dim=1)[:, 1]

            predictions = predictions.cpu().numpy()

        # reshape predictions for backtest
        predictions = predictions.reshape(-1, prices.shape[1])

        # -------------------------
        # Run backtest
        # -------------------------

        values, final_value, profit = run_backtest(prices, predictions)

        metrics = calculate_metrics(values)

        results.append({
            "train_period": f"{train_start}-{train_end}",
            "test_period": f"{test_start}-{test_end}",
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "return": metrics["total_return"],
            "win_rate": metrics["win_rate"]
        })

    return pd.DataFrame(results)