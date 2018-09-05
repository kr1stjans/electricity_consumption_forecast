from dev.models.abstract_model import AbstractModel
from dev.settings import FORECAST_SIZE


class CNNModel(AbstractModel):

    @staticmethod
    def get_prediction(x_values, y_values, train_end_index):
        # TODO: implement simple 1D CNN
        # reference: https://gist.github.com/jkleint/1d878d0401b28b281eb75016ed29f2ee

        return y_values[train_end_index - FORECAST_SIZE:train_end_index]

    @staticmethod
    def transform_data(df):
        # TODO: transform data accordingly

        return df['value'], df['value']
