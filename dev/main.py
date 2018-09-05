import datetime
import os

import numpy

import pandas as pd

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf, pacf

from dev.data_util import DataProcessor
from dev.models.baseline import BaselineModel
from dev.models.ridge_regression import RidgeRegressionModel
from dev.settings import FORECAST_SIZE
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import autocorrelation_plot

sns.set(style="darkgrid")


def total_rmse():
    consumers = []  # DataProcessor.load_data_as_separate_dataframes()

    complete_df = pd.read_csv(filepath_or_buffer='../data/new.csv', sep=',',
                              header=1,
                              names=['dt', 'id', 'value'], index_col='id',
                              parse_dates=['dt'],
                              infer_datetime_format=True, dtype={'value': numpy.float64}, memory_map=True)

    print("dataframes len", len(consumers))
    print(consumers[0].head(100))

    models = [BaselineModel(),
              RidgeRegressionModel(True)
              # AutoregressiveModel(),
              # RandomForestModel()
              ]

    for model in models:
        model_rmse = 0
        for consumer_data in consumers:
            consumer_rmse = 0
            X, y = model.transform_data(consumer_data)
            cnt = 0
            # start after one year of data (FORECAST_SIZE * 365 values) and continue with steps of FORECAST_SIZE
            for train_end_index in enumerate(
                    range(FORECAST_SIZE * 365, len(consumer_data) - FORECAST_SIZE * 2, FORECAST_SIZE)):
                y_hat = model.get_prediction(X, y, train_end_index)
                y_actual = y[train_end_index:train_end_index + FORECAST_SIZE]
                consumer_rmse += mean_squared_error(y_actual, y_hat) ** 0.5
                cnt += 1

            iteration_cnt = (((len(consumer_data) - FORECAST_SIZE * 2) - FORECAST_SIZE * 365) / FORECAST_SIZE)
            print("iteration_cnt", iteration_cnt, "cnt", cnt)

            model_rmse += consumer_rmse / iteration_cnt

        print('total model_rmse for model', model, model_rmse / len(consumers))


def get_rmse_for_model(df, forecast_fn, get_X_y):
    rmse_sum = 0

    X, y = get_X_y(df)
    cnt = 0
    # start after one year of data (FORECAST_SIZE * 365 values) and continue with steps of FORECAST_SIZE
    for train_end_index in range(FORECAST_SIZE * 365, len(df) - FORECAST_SIZE * 2, FORECAST_SIZE):
        y_hat = forecast_fn(X, y, train_end_index)
        y_actual = y[train_end_index:train_end_index + FORECAST_SIZE]
        rmse_sum += mean_squared_error(y_actual, y_hat) ** 0.5
        cnt += 1
    return rmse_sum / cnt


def plot_nr_of_measurements_per_dt(df):
    measurements_per_dt = df.groupby(df.index).count()
    plt.figure()
    plt.plot(measurements_per_dt.index, measurements_per_dt['id'])
    plt.show()


def plot_summed_values(df):
    sum_per_dt = df.groupby(df.index).sum()
    print(sum_per_dt)
    plt.figure()
    plt.plot(sum_per_dt.index, sum_per_dt['value'])
    plt.show()


def acf_custom(series):
    n = len(series)
    data = numpy.asarray(series)
    mean = numpy.mean(data)
    c0 = numpy.sum((data - mean) ** 2) / float(n)

    def r(h):
        acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
        return round(acf_lag, 3)

    x = numpy.arange(n)  # Avoiding lag 0 calculation
    acf_coeffs = map(r, x)
    return list(acf_coeffs)


def plot_autocorrelations():
    data = DataProcessor.load_data_as_separate_dataframes()

    plt.figure(1)
    for d in data:
        acf_values = pacf(d['value'].values[:336], nlags=336)
        # pacf()
        # acf_values = acf(d['value'].values[:350])[1:]
        plt.plot(range(len(acf_values)), acf_values)
    plt.show()


if __name__ == "__main__":
    plot_autocorrelations()
    # total_rmse()
    # complete_df = pd.read_csv(filepath_or_buffer='../data/new.csv', sep=',',
    #                          header=1,
    #                          names=['dt', 'id', 'value'], index_col='dt',
    #                          parse_dates=True,
    #                          infer_datetime_format=True, dtype={'value': numpy.float64}, memory_map=True)
    # plot_nr_of_measurements_per_dt(complete_df)
