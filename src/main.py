import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.models import model_from_json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.testing_data_preprocessor import Preprocessor
from src.testing_data_manager import DataManager

FORECAST_SIZE = 96


def get_identity_transform(data):
    return data[:, 0], data[:, 0]


def get_baseline_prediction(x_values, y_values, train_size):
    """
    Get last known values as prediction.
    :param values: 
    :param x_values_length: 
    :return: 
    """
    last_day = y_values[train_size - FORECAST_SIZE:train_size]
    return last_day


def get_AR_prediction(x_values, y_values, train_size):
    """
    Get Autoregressive model result as prediction.
    :param values: 
    :param train_size: 
    :return: 
    """
    model = AR(x_values[:train_size])
    fitted_model = model.fit(maxlag=96)
    return fitted_model.predict(start=train_size, end=train_size + (FORECAST_SIZE - 1))


def get_ARIMA_prediction(x_values, y_values, train_size):
    """
    Get Autoregressive model result as prediction.
    :param values:
    :param train_size:
    :return:
    """
    model = ARIMA(x_values[:train_size], order=(16, 0, 1))
    fitted_model = model.fit()
    return fitted_model.predict(start=train_size, end=train_size + (FORECAST_SIZE - 1))


def get_LR_prediction(x_values, y_values, train_size):
    """
    Get Linear Regression model result as prediction.
    :param x_values: 
    :param y_values: 
    :param train_size: 
    :return: 
    """

    x_train = x_values[:train_size]
    y_train = y_values[:train_size]
    x_test = x_values[train_size:train_size + FORECAST_SIZE]
    regr = LinearRegression()
    regr.fit(x_train, y_train)
    return regr.predict(X=x_test)


def get_LR_transform(data):
    x_values, y_values = Preprocessor().fit_transform(X=data[:, 1:], y=data[:, 0])
    return x_values, y_values


def get_LSTM_prediction(x_values, y_values, train_size):
    # extract train data as in real world scenario - we don't know anything else
    x_train = x_values[:train_size]
    y_train = y_values[:train_size]
    x_test = x_values[train_size:train_size + FORECAST_SIZE]

    # rescale the train data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = scaler.fit_transform(x_train)
    y_train = scaler.transform(y_train)

    # convert from 2d (samples, features) to 3d (samples, timesteps=1, features)
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

    n_batch = 1
    nb_epoch = 150
    n_neurons = 1

    if os.path.exists('cached_models/model.json'):
        json_model_file = open('cached_models/model.json', 'r')
        loaded_model_json = json_model_file.read()
        json_model_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("cached_models/model.h5")
        print("Loaded model from disk")
    else:
        model = Sequential()
        model.add(LSTM(n_neurons, batch_input_shape=(n_batch, x_train.shape[1], x_train.shape[2]), stateful=True))
        model.add(Dense(y_train.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')

        for i in range(nb_epoch):
            model.fit(x_train, y_train, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
            print("fit epochs", i)
            model.reset_states()

        model_json = model.to_json()
        with open("cached_models/model.json", "w") as json_model_file:
            json_model_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("cached_models/model.h5")
        print("Saved Keras model to disk")

    forecasts = []
    for i in range(len(x_test)):
        x_test = x_test.reshape(1, 1, len(x_test))
        forecast = model.predict(x_test, batch_size=n_batch)
        forecasts.append(forecast)

    print("forecasts", forecasts)

    # transform back and return
    return scaler.transform(forecasts)


def get_LSTM_transform(data):
    n = len(data)

    y_values = []

    empty_extension = [0] * FORECAST_SIZE
    x_values_extended = np.append(data[:, 0], empty_extension)
    for i in range(FORECAST_SIZE):
        y_values.append(x_values_extended[i + 1:n + i + 1])

    y_values = np.array(y_values, dtype=np.float64).T

    return np.array(data[:, 0:1]), y_values


def cross_validate_model(data, prediction_fn, transform_fn):
    x_values, y_values = transform_fn(data)

    rmse_sum = 0
    abs_error_sum = 0
    variance_sum = 0

    cnt = 0
    # start after one year of data (96*365 values) and continue with steps of one day (96 values)
    for train_size in range(96 * 365, len(data) - 2 * FORECAST_SIZE, 96):
        if cnt % 50 == 0:
            print('Cross validating @', train_size, "measurement place", "-1")

        predicted_values = prediction_fn(x_values, y_values, train_size)
        test_values = y_values[train_size:train_size + FORECAST_SIZE]

        rmse_sum += mean_squared_error(test_values, predicted_values) ** 0.5
        abs_error_sum += mean_absolute_error(test_values, predicted_values)
        variance_sum += explained_variance_score(test_values, predicted_values)
        cnt += 1

    print('avg rmse', rmse_sum / cnt)
    print('avg abs error', abs_error_sum / cnt)
    print('total avg variance', variance_sum / cnt)


def timeseries_analysis(data):
    values = data[:, 0]
    #
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(data[:, 1], values)
    #
    # plt.subplot(212)
    # plt.hist(values, bins=20)
    #
    # plt.show()

    # print('Dickey-Fuller Test')
    # print(
    #     'p-value > 0.05: Accept the null hypothesis H0 (the data has a unit root and is non-stationary). It has some time dependent structure.')
    # print(
    #     'p-value <= 0.05: Reject the null hypothesis H0, the data does not have a unit root and is stationary. It does not have time-dependent structure.')
    # dftest = adfuller(values, regression="ctt", maxlag=192)
    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    # for key, value in dftest[4].items():
    #     dfoutput['Critical Value (%s)' % key] = value
    # print(dfoutput)

    # Determing rolling statistics
    values_df = pd.DataFrame(values)
    rolmean = values_df.rolling(window=96 * 7).mean()
    rolstd = values_df.rolling(window=96 * 7).std()

    plt.figure()
    # Plot rolling statistics:
    orig = plt.plot(data[:, 1], values_df, color='blue', label='Original')
    mean = plt.plot(data[:, 1], rolmean, color='red', label='Rolling Mean')
    std = plt.plot(data[:, 1], rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.show()


# timeseries_analysis(data)

data = DataManager().get_data("-1")
cross_validate_model(data, get_baseline_prediction, get_identity_transform)
cross_validate_model(data, get_LR_prediction, get_LR_transform)
# cross_validate_model(data, get_AR_prediction, get_identity_transform)
# cross_validate_model(data, get_ARIMA_prediction, get_identity_transform)
# cross_validate_model(data, get_LSTM_prediction, get_LSTM_transform)
