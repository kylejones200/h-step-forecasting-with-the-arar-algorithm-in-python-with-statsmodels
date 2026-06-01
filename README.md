# H-Step Forecasting with the ARAR Algorithm

Published: yes
Medium: [https://medium.com/@kyle-t-jones/h-step-forecasting-with-the-arar-algorithm-in-python-with-statsmodels-43017792a863](https://medium.com/@kyle-t-jones/h-step-forecasting-with-the-arar-algorithm-in-python-with-statsmodels-43017792a863)


This project demonstrates the ARAR (AutoRegressive AutoRegressive) algorithm for h-step ahead forecasting, with comparison to ARIMA.

## Business context

In time series forecasting, we often need to predict multiple steps ahead, which is known as *h-step forecasting*. The challenge lies in capturing both short-term fluctuations and long-term trends effectively. The AutoRegressive-to-AutoRegressive (ARAR) algorithm is particularly useful for forecasting time series with strong short-term dependencies but weak long-term correlations.

Originally proposed by McLeod and Hipel (1978), the ARAR algorithm is a practical approach that blends simple transformation, differencing, and autoregressive (AR) modeling to produce robust forecasts. This chapter explores the ARAR methodology, its advantages, and a Python implementation using real-world data.

The ARAR algorithm is designed to model time series data efficiently while avoiding the complexity of full autoregressive integrated moving average (ARIMA) models. It operates in three main stages:

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # ARAR forecasting functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source and column names
- ARAR model parameters (lag selection strategy, lags)
- ARIMA model parameters (order)
- Forecast horizon
- Output settings

## ARAR Algorithm

The ARAR algorithm:
1. Applies differencing to remove trend
2. Selects reduced lag set (typically powers of 2: 1, 2, 4, 8, 16)
3. Fits autoregressive model on differenced data
4. Generates h-step forecasts
5. Reverses differencing to reconstruct original scale

## Caveats

- The algorithm requires sufficient data for differencing and lag selection.
- Reduced lag sets (powers of 2) are computationally efficient but may not capture all dependencies.
- ARAR is compared with ARIMA to demonstrate relative performance.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).