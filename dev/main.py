import datetime
import os

import numpy as np

import pandas as pd
import sklearn
from sklearn.cluster import KMeans

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
                              infer_datetime_format=True, dtype={'value': np.float64}, memory_map=True)

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
    # start after one year of data (FORECAST_SIZE * 365 values) a   nd continue with steps of FORECAST_SIZE
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


def plot_autocorrelations():
    data = DataProcessor.load_data_as_separate_dataframes()

    print('data loaded')
    avg_df = pd.concat(data, axis=1).mean(axis=1)
    avg_df.to_csv('total_average.csv')
    print('data merged')
    acf_values = acf(avg_df['value'].values[:3000], nlags=3000)
    print('values calculated')
    # acf_values = acf(d['value'].values[:350])[1:]
    plt.plot(range(len(acf_values)), acf_values)
    '''
    plt.figure(1)
    for d in data:
        acf_values = acf(d['value'].values[:192], nlags=192)
        # acf_values = acf(d['value'].values[:350])[1:]
        plt.plot(range(len(acf_values)), acf_values)
    '''
    plt.show()


def load_initial_consumers():
    original_data = DataProcessor.load_data_as_separate_dataframes()
    data = pd.concat(original_data, axis=1)
    data = data.fillna(method='ffill', axis='index').fillna(0).values.transpose()

    clusters = [[664722.6098773873, 639748.0777654473, 630854.7795726027, 626444.6719076467, 618110.443418544,
                 616578.6866641308, 612753.1607919122, 609630.8295018858, 605946.7952054872, 605662.9731812247,
                 599176.8038367201, 596376.1178825608, 593176.6868208277, 592780.2515878733, 591349.6256350856,
                 586922.5082948159, 585265.3377661232, 582881.8311003151, 581049.8068028545, 579026.451466421,
                 576382.3357024424, 574056.5614726953, 573409.7720791526, 570961.5226713157, 567218.3231113362,
                 564947.3855997741, 563831.9900033098, 560561.940054088, 558314.3214983398, 555226.5072961054,
                 555095.2757642752, 553191.7733401267, 550949.7789032019, 547721.3887849059, 546614.6202111957,
                 546185.5629633496, 543651.8154553492, 539997.8885186019, 540183.2744133768]]
    for k in range(200, 500, 5):
        print('calculating cluster', k)
        clusters.append(KMeans(n_clusters=k).fit(data).inertia_)

    print(clusters)
    plt.figure(1)
    plt.plot(range(len(clusters)), clusters, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    exit(1)
    cluster_size = 25
    kmeans = KMeans(n_clusters=cluster_size).fit(data)
    for i in range(cluster_size):
        plt.subplot(cluster_size / 5, cluster_size / (cluster_size / 5), i + 1)
        for idx, d in enumerate(original_data):
            if kmeans.labels_[idx] == i:
                plt.plot(range(len(d)), d)
    plt.show()


def datetime64_to_time_of_day(datetime64_array):
    day = datetime64_array.astype('datetime64[D]').astype(datetime64_array.dtype)
    time_of_day = datetime64_array - day
    return time_of_day


def avg_day_clusters():
    original_data = DataProcessor.load_data_as_separate_dataframes()

    daily_profile = []

    for d in original_data.copy():
        d['time'] = list(datetime64_to_time_of_day(d.index.values))
        d = d.set_index(keys='time', drop=True).groupby(level=0).mean()
        daily_profile.append(d)

    data = pd.concat(daily_profile, axis=1)
    data = data.fillna(method='ffill', axis='index').fillna(0).values.transpose()

    # clusters = []
    # for k in range(5, 500, 5):
    #     print('calculating cluster', k)
    #     clusters.append(KMeans(n_clusters=k).fit(data).inertia_)
    #
    # print(clusters)
    # plt.figure(1)
    # plt.plot(range(len(clusters)), clusters, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum_of_squared_distances')
    # plt.title('Elbow Method For Optimal k')
    # plt.show()
    #
    # exit(1)cd

    cluster_size = 300
    kmeans = KMeans(n_clusters=cluster_size).fit(data)
    for i in range(cluster_size):
        plt.subplot(50, 60, i + 1)
        for idx, d in enumerate(data):
            if kmeans.labels_[idx] == i:
                #values = d['value'].valuespip i
                plt.plot(range(len(d)), d)
    plt.show()


if __name__ == "__main__":
    avg_day_clusters()
    # load_initial_consumers()
# plot_autocorrelations()
# total_rmse()
# complete_df = pd.read_csv(filepath_or_buffer='../data/new.csv', sep=',',
#                          header=1,“
#                          names=['dt', 'id', 'value'], index_col='dt',
#                          parse_dates=True,
#                          infer_datetime_format=True, dtype={'value': numpy.float64}, memory_map=True)
# plot_nr_of_measurements_per_dt(complete_df)
