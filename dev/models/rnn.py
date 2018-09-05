from dev.models.abstract_model import AbstractModel
from dev.settings import FORECAST_SIZE


class RNNModel(AbstractModel):

    @staticmethod
    def get_prediction(x_values, y_values, train_end_index):

        # TODO: implement simple RNN
        return y_values[train_end_index - FORECAST_SIZE:train_end_index]

    @staticmethod
    def transform_data(df):

        # TODO: transform data accordingly

        return df['value'], df['value']
