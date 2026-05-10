---
author: "Kyle Jones"
date_published: "February 1, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/h-step-forecasting-with-the-arar-algorithm-in-python-with-statsmodels-43017792a863"
---

# H-Step Forecasting with the ARAR Algorithm in Python

with statsmodels ARAR: a simpler approach to multi-step time series forecasting
### H-Step Forecasting with the ARAR Algorithm in Python with statsmodels
#### ARAR: a simpler approach to multi-step time series forecasting
In time series forecasting, we often need to predict multiple steps ahead, which is known as *h-step forecasting*. The challenge lies in capturing both short-term fluctuations and long-term trends effectively. The AutoRegressive-to-AutoRegressive (ARAR) algorithm is particularly useful for forecasting time series with strong short-term dependencies but weak long-term correlations.

Originally proposed by McLeod and Hipel (1978), the ARAR algorithm is a practical approach that blends simple transformation, differencing, and autoregressive (AR) modeling to produce robust forecasts. This chapter explores the ARAR methodology, its advantages, and a Python implementation using real-world data.

### Understanding the ARAR Algorithm
The ARAR algorithm is designed to model time series data efficiently while avoiding the complexity of full autoregressive integrated moving average (ARIMA) models. It operates in three main stages:

1.  [**Data Preprocessing:** The series undergoes transformation and differencing to stabilize variance and remove non-stationarity.]
2.  [**Reduced Lag Selection:** The method selects a reduced subset of lagged values for modeling, avoiding unnecessary complexity.]
3.  [**Final AR Modeling:** An autoregressive model is fitted to the transformed and reduced data.]

This streamlined approach is computationally efficient, making it well-suited for practical applications in time series forecasting.

ARAR starts by transforming the series using a first-order difference and a nonlinear transformation if necessary. If the data exhibits a strong trend, a logarithmic or power transformation may be applied.

Instead of considering all possible lags, ARAR selects a reduced set of lag values using a heuristic approach. Typically, lags at powers of two (e.g., 1, 2, 4, 8, ...) are chosen. This reduces computational burden while maintaining sufficient historical information for forecasting.

The process involves:

- Identifying significant autocorrelations in the differenced series.
- Selecting a minimal subset of lagged observations based on statistical significance.

By focusing on a reduced number of lags, ARAR ensures that the final AR model remains parsimonious yet effective.

Once the AR model is fitted, forecasts are generated recursively:

1.  [Compute the predicted values for future time steps based on past observations.]
2.  [Use predicted values iteratively to extend forecasts to hh-steps ahead.]

The final forecast is obtained by reversing the differencing transformation.

### Python Implementation of ARAR Forecasting
Let's implement the ARAR method in Python using a sample time series dataset.

I'm using data from ERCOT, the grid balancing authority in Texas. The values are electricity demand every 15 mins. The data is publicly available. [Here is my cleaned up version ready for analysis.](https://github.com/kylejones200/time_series/blob/main/ercot_load_data.csv)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_absolute_percentage_error

# Load dataset
data = pd.read_csv("ercot_load_data.csv", parse_dates=["date"], index_col="date")
y = data["values"]
# Set the Date index to 15 min frequency
data = data.asfreq("15min")
```

The data comes in every 15 mins. I want to back up 1 day and then forecast. So I'll separate out the last 96 observations.

``` 
# Define forecast horizon (1 day = 96 steps)
h = 96  

# Split data: training (everything except last 96 values) & test (last 96 values)
train, test = y.iloc[:-h], y.iloc[-h:]

# Apply differencing on training data for ARAR
z_train = np.diff(train)

# Compute autocorrelation and select reduced lags (powers of 2)
acf_vals = acf(z_train, nlags=20)
lags = [1, 2, 4, 8, 16]  # Selected lag set
```

I want to forecast for one day (which in this case has already happened).

``` 
# Fit ARAR model
arar_model = AutoReg(z_train, lags=lags, old_names=False).fit()

# Generate forecasts for next 96 steps
future_forecast_arar = arar_model.predict(start=len(z_train), end=len(z_train) + h - 1)

# Reverse differencing to reconstruct the original scale
y_forecast_arar = np.cumsum(future_forecast_arar) + train.iloc[-1]

# Create new time index for forecasts
forecast_index = pd.date_range(start=train.index[-1], periods=h+1, freq="15min")[1:]

# Compute MAPE for ARAR
mape_arar = mean_absolute_percentage_error(test, y_forecast_arar)
print(f"MAPE for ARAR: {mape_arar:.4f}")
```

ARAR is similar to ARIMA. So let's build an ARIMA model and compare them.

``` 
# Fit ARIMA(2,1,2) model
arima_model = ARIMA(train, order=(2,1,2)).fit()

# Generate forecasts for next 96 steps
y_forecast_arima = arima_model.forecast(steps=h)

# Convert forecast index to match ARIMA output
y_forecast_arima.index = forecast_index

# Compute MAPE for ARIMA
mape_arima = mean_absolute_percentage_error(test, y_forecast_arima)
print(f"MAPE for ARIMA: {mape_arima:.4f}")
```

I'm using mean absolute percentage error to compare the accuracy of the forecasts against the test data (96 value). For MAPE, lower values are better.

``` 
MAPE for ARAR: 0.0646
MAPE for ARIMA: 0.0964
```

Let's plot ARAR and ARIMA.

``` 
# Plot full time series with forecasted values
plt.figure(figsize=(12, 6))

# Plot full historical series
plt.plot(y.index, y, label="Historical Data", linestyle="-", color="blue")

# Plot ARAR forecast
plt.plot(forecast_index, y_forecast_arar, label="ARAR Forecast", linestyle="dashed", color="red")

# Plot ARIMA forecast
plt.plot(forecast_index, y_forecast_arima, label="ARIMA Forecast", linestyle="dotted", color="green")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title(f"Full Time Series with ARAR vs ARIMA Forecasts\nMAPE (ARAR): {mape_arar:.4f}, MAPE (ARIMA): {mape_arima:.4f}")
plt.legend()

plt.savefig("arar_vs_arima_forecast.png")
plt.show()
```


ARAR is better in this case.

### Advantages of ARAR for H-Step Forecasting
The ARAR algorithm is particularly useful when the time series exhibits strong short-term dependencies but weak long-term correlations.

ARAR is a lightweight yet effective forecasting method. Compared to ARIMA, ARAR avoids unnecessary differencing and reduces computational complexity while achieving comparable forecasting accuracy.

### So what?
H-step forecasting requires a balance between capturing short-term variations and maintaining computational efficiency.
