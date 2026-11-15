# EUR/USD Volatility Modeling with ARIMA-GARCH

Forecasting the daily volatility of EUR/USD with GARCH-family models,
compared with EWMA (RiskMetrics) and 20-day historical volatility.

## Research question
For a risk manager, the key input is tomorrow's volatility. It is not
observable and it changes over time. This project asks: can GARCH-family
models forecast next-day EUR/USD volatility significantly better than
standard industry benchmarks (EWMA RiskMetrics, 20-day historical volatility)?
The goal is to forecast the size of the moves, not the direction of the rate.

## Setup
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Then open `main.ipynb` and run all cells.

## Project status
- [x] Project setup
- [ ] Data and log-returns
- [ ] Stylized facts and stationarity
- [ ] Mean model (ARIMA vs random walk)
- [ ] Variance models (GARCH, GJR, EGARCH) and diagnostics
- [ ] Out-of-sample forecasting and evaluation