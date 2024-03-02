from src.models.abstract_model import AbstractModel
from src.settings import FORECAST_SIZE


class BaselineModel(AbstractModel):

    @staticmethod
    def get_forecast(x_values, y_values, train_end_index):
        return y_values[train_end_index - FORECAST_SIZE:train_end_index]

    @staticmethod
    def transform_data(df):
        return df['value'], df['value']
