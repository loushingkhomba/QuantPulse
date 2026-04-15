import numpy as np


def run_backtest(prices, predictions, initial_capital=10000, top_k=3):

    capital = initial_capital

    portfolio_values = []

    positions = {}

    for t in range(len(prices)):

        daily_prices = prices[t]
        daily_preds = predictions[t]

        ranked = np.argsort(daily_preds)[::-1]

        selected = ranked[:top_k]

        capital_per_stock = capital / top_k

        positions = {}

        for idx in selected:

            price = daily_prices[idx]

            shares = capital_per_stock / price

            positions[idx] = shares

        portfolio_value = 0

        for idx, shares in positions.items():

            price = daily_prices[idx]

            portfolio_value += shares * price

        portfolio_values.append(portfolio_value)

        capital = portfolio_value

    final_value = portfolio_values[-1]

    profit = final_value - initial_capital

    return portfolio_values, final_value, profit


def calculate_metrics(values):

    values = np.array(values)

    if values.size == 0:
        return 0.0, 0.0

    if values.size == 1:
        return 0.0, 0.0

    returns = np.diff(values) / values[:-1]

    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)

    cumulative_max = np.maximum.accumulate(values)

    drawdown = (values - cumulative_max) / cumulative_max

    max_drawdown = drawdown.min() if drawdown.size > 0 else 0.0

    return sharpe, max_drawdown