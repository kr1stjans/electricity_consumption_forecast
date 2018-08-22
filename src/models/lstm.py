from src.models.abstract_model import AbstractModel
from src.settings import FORECAST_SIZE


class LSTMModel(AbstractModel):

    @staticmethod
    def get_forecast(x_values, y_values, train_end_index):
        # TODO: implement LSTM model (reference: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

        return y_values[train_end_index - FORECAST_SIZE:train_end_index]

    @staticmethod
    def transform_data(df):
        # TODO: transform data accordingly

        return df['value'], df['value']
