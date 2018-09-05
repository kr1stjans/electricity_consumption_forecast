from dev.models.abstract_model import AbstractModel
from dev.settings import FORECAST_SIZE


class BaselineModel(AbstractModel):

    @staticmethod
    def get_prediction(x_values, y_values, train_end_index):
        return y_values[train_end_index - FORECAST_SIZE:train_end_index]

    @staticmethod
    def transform_data(df):
        return df['value'], df['value']
