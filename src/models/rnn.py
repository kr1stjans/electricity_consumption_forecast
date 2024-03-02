from src.models.abstract_model import AbstractModel
from src.settings import FORECAST_SIZE


class RNNModel(AbstractModel):

    @staticmethod
    def get_forecast(x_values, y_values, train_end_index):

        # TODO: implement simple RNN with one sigmoid (tanh) level
        return y_values[train_end_index - FORECAST_SIZE:train_end_index]

    @staticmethod
    def transform_data(df):

        # TODO: transform data accordingly

        return df['value'], df['value']
