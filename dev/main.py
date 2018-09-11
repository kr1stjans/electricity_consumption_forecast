import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_error

from dev.data_util import DataProcessor
from dev.models.baseline import BaselineModel
from dev.models.conv1d import Conv1DModel
from dev.models.conv2d import Conv2DModel
from dev.models.lstm import LSTMModel
from dev.models.ridge_regression import RidgeRegressionModel
from dev.models.stacked_lstm import StackedLSTMModel
from dev.settings import FORECAST_SIZE

from dev.statistics import Statistics

TRAINING_START = '2013-09-01'
TESTING_INDEX_END = '2014-02-20'

CONSUMERS_TO_TEST = 10


def total_rmse():
    consumers = DataProcessor.load_data_as_separate_dataframes(CONSUMERS_TO_TEST)
    print("Loaded %s consumers" % len(consumers))

    models = [BaselineModel(),
              RidgeRegressionModel(True),
              #StackedLSTMModel(),
              #LSTMModel(),
              #Conv2DModel(),
              Conv1DModel()
              # AutoregressiveModel(),
              # RandomForestModel()
              ]

    results = pd.DataFrame()

    for model in models:
        model_rmse = 0
        for consumer_name, consumer_data in consumers.items():
            X, y = model.transform_data(consumer_data)

            s = Statistics()

            training_index_start = np.where(consumer_data.index == TRAINING_START)[0][0]
            training_index_end = np.where(consumer_data.index == TESTING_INDEX_END)[0][0]

            for train_end_index in range(training_index_start, training_index_end, FORECAST_SIZE):
                y_actual = consumer_data['value'][train_end_index:train_end_index + FORECAST_SIZE].values
                y_hat = model.get_prediction(X, y, train_end_index, consumer_name)

                if y_hat is None:
                    print("Cant get prediction for date ", consumer_data.index[train_end_index])
                    continue

                s.update_average(mean_squared_error(y_actual, y_hat) ** 0.5)

            model_rmse += s.get_average()
            results = results.append({'model': model.get_name(), 'consumer': consumer_name, 'rmse': s.get_average()},
                                     ignore_index=True)

        total_model_rmse = model_rmse / len(consumers)
        print('total rmse for model', model.get_name(), total_model_rmse)
        results = results.append({'model': model.get_name(), 'consumer': 'ALL', 'rmse': total_model_rmse},
                                 ignore_index=True)

    results.to_csv('ALL.csv')


if __name__ == "__main__":
    total_rmse()
