# Time Series Analysis:
* test stationarity

    Dickey-Fuller Test
    p-value > 0.05: Accept the null hypothesis H0 (the data has a unit root and is non-stationary). It has some time dependent structure.
    p-value <= 0.05: Reject the null hypothesis H0, the data does not have a unit root and is stationary. It does not have time-dependent structure.
    Test Statistic                    -3.985090
    p-value                            0.032875
    Lags Used                       192.000000
    Number of Observations Used    46459.000000
    Critical Value (1%)               -4.371379
    Critical Value (5%)               -3.832517
    Critical Value (10%)              -3.553339
    dtype: float64

* rolling std, mean, variance

# Time Series Forecasting:
y = level + trend + seasonality + noise

## definition

* n-ary tree of grid (consumers, stations, main station)
* 15 minute interval data, 1.5y of data (53k measurements values) per entity
* multivariate problem: time, value, weather


## cleaning:
    * Outliers. Perhaps there are corrupt or extreme outlier values that need to be identified and handled.
    * Missing. Perhaps there are gaps or missing data that need to be interpolated or imputed.


## linear
    * ts fresh
    * pandas.rolling mean, min, max, etc. statistics
        * last hour, last 3 hours, last 6 hours, last 12 hours, last 24 hours, last 36 hours, last 96 hours

    * linear regression
    * ARIMA (acf, pacf)  = autoregression + moving average (+ integration)
    * triple exponential smoothening: https://grisha.org/blog/2016/01/29/triple-exponential-smoothing-forecasting/


### todo:
* Vector Autoregression Model (VAR)
* Vector Error Correction model (VEC)

## non linear
    - treshold models (https://newonlinecourses.science.psu.edu/stat510/node/82/)
    - lstm
    - Autoregressive Conditional Heteroskedasticity (ARCH) and its variations like Generalized ARCH (GARCH), Exponential Generalized ARCH (EGARCH) etc.,
    - support vector machine
    - The Time Lagged Neural Network (TLNN)
    - Artificial Neural Networks (ANNs) - most popular MLP (multilayer perceptron)

## evaluation
* A rolling-forecast scenario will be used, also called walk-forward model validation.
* we are predicting 1 day in advance (96 values)
* baseline prediction / naive forecast: persistence model as last 96 values

## results
* BASELINE
avg rmse 49.29696566763337
avg abs error 35.06244079415899
total avg variance 0.7725809653822341

* linear regression
avg rmse 29.273137752516288
avg abs error 22.969673210135845
total avg variance 0.932844998671164

* LSTM neural net
avg rmse 28.01031237366144
avg abs error 20.438305952785015
total avg variance 0.9489922298242489


* LSTM
TODO


TODO: how much each feature provides in the model? random or PCA?

# good visualization example
http://www.business-science.io/timeseries-analysis/2018/04/18/keras-lstm-sunspots-time-series-prediction.html


https://towardsdatascience.com/using-lstms-to-forecast-time-series-4ab688386b1f


# test gradient descend on linear regression?