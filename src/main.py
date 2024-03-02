from sklearn.metrics import mean_squared_error
from src.models.autoregression import AutoregressiveModel
from src.models.baseline import BaselineModel
from src.models.lasso_regression import LassoRegressionModel
from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel
from src.data_util import DataProcessor
from src.models.ridge_regression import RidgeRegressionModel
from src.settings import FORECAST_SIZE


def total_rmse():
    data_frames = DataProcessor.load_data_as_separate_dataframes()

    models = [BaselineModel(),
              LinearRegressionModel(),
              LassoRegressionModel(),
              RidgeRegressionModel(),
              AutoregressiveModel()]

    for model in models:
        rmse = 0
        for df in data_frames:
            rmse += get_rmse_for_model(df, model.get_forecast, model.transform_data)

        print('total rmse for model', model, rmse / len(data_frames))


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


if __name__ == "__main__":
    total_rmse()
