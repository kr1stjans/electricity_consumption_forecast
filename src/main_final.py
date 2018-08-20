import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# from keras import Sequential
# from keras.layers import LSTM, Dense
# from keras.models import model_from_json
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.testing_data_preprocessor import Preprocessor
from src.testing_data_manager import DataProcessor

FORECAST_SIZE = 48


def get_identity_X_y_pair(data):
    return data, data['value']


def get_baseline_forecast(x_values, y_values, train_end_index):
    """
    Method returns last FORECAST_SIZE values as baseline prediction.
    :param values: 
    :param x_values_length: 
    :return: 
    """
    return y_values[train_end_index - FORECAST_SIZE:train_end_index]


def get_AR_forecast(X, y, train_size):
    """
    Get Autoregressive model result as prediction.
    :param values: 
    :param train_size: 
    :return: 
    """
    model = AR(X['value'][:train_size], dates=X.index, freq=)
    fitted_model = model.fit(maxlag=48)
    return fitted_model.predict(start=train_size, end=train_size + (FORECAST_SIZE - 1))


def get_ARIMA_forecast(X, y, train_size):
    """
    Get Autoregressive model result as prediction.
    :param values:
    :param train_size:
    :return:
    """
    model = ARIMA(X[:train_size], order=(16, 0, 1))
    fitted_model = model.fit()
    return fitted_model.predict(start=train_size, end=train_size + (FORECAST_SIZE - 1))


def get_LR_forecast(x_values, y_values, train_end_index):
    """
    Get Linear Regression model result as prediction.
    :param x_values: 
    :param y_values: 
    :param train_end_index: 
    :param model: 
    :return: 
    """
    x_test = x_values[train_end_index:train_end_index + FORECAST_SIZE]
    model = LinearRegression()
    model.fit(x_values[:train_end_index], y_values[:train_end_index])
    return model.predict(X=x_test)


def get_ridge_forecast(x_values, y_values, train_end_index):
    """
    Get Ridge model result as prediction.
    :param x_values: 
    :param y_values: 
    :param train_end_index: 
    :param model: 
    :return: 
    """
    x_test = x_values[train_end_index:train_end_index + FORECAST_SIZE]
    model = Ridge(alpha=0.05)
    model.fit(x_values[:train_end_index], y_values[:train_end_index])
    return model.predict(X=x_test)


def get_lasso_forecast(x_values, y_values, train_end_index):
    """
    Get Lasso model result as prediction.
    :param x_values: 
    :param y_values: 
    :param train_end_index: 
    :param model: 
    :return: 
    """
    x_test = x_values[train_end_index:train_end_index + FORECAST_SIZE]
    model = Lasso(alpha=0.05)
    model.fit(x_values[:train_end_index], y_values[:train_end_index])
    return model.predict(X=x_test)


def get_LR_transform(df):
    x_values, y_values = Preprocessor().fit_transform(X=df, y=df['value'])
    return x_values, y_values


'''
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
'''


def cross_validate_model(df, forecast_fn, get_X_y):
    X, y = get_X_y(df)

    rmse_sum = 0
    abs_error_sum = 0
    variance_sum = 0

    cnt = 0
    # start after one year of data (48*365 values) and continue with steps of FORECAST_SIZE
    for train_end_index in range(FORECAST_SIZE * 365, len(df) - FORECAST_SIZE * 2, FORECAST_SIZE):
        if cnt % 50 == 0:
            print('Cross validating @', train_end_index, "measurement place", "-1")

        y_hat = forecast_fn(X, y, train_end_index)
        y_actual = y[train_end_index:train_end_index + FORECAST_SIZE]

        rmse_sum += mean_squared_error(y_actual, y_hat) ** 0.5
        abs_error_sum += mean_absolute_error(y_actual, y_hat)
        variance_sum += explained_variance_score(y_actual, y_hat)
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


def plot_data(df):
    plt.figure()
    plt.plot(df.index, df['value'])
    plt.show()


df = DataProcessor.get_public_data()

cross_validate_model(df, get_baseline_forecast, get_identity_X_y_pair)
# cross_validate_model(df, get_LR_forecast, DataProcessor.get_LR_transform)
# cross_validate_model(df, get_lasso_forecast, DataProcessor.get_LR_transform)
# cross_validate_model(df, get_ridge_forecast, DataProcessor.get_LR_transform)
# cross_validate_model(df, get_LR_forecast, get_LR_transform)
cross_validate_model(df, get_AR_forecast, get_identity_X_y_pair)
# cross_validate_model(data, get_ARIMA_prediction, get_identity_transform)
# cross_validate_model(data, get_LSTM_prediction, get_LSTM_transform)
