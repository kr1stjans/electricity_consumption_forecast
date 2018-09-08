from dev.models.abstract_model import AbstractModel
from dev.settings import FORECAST_SIZE


class BaselineModel(AbstractModel):

    def get_prediction(self, x_values, y_values, train_end_index, consumer_index):
        return y_values[train_end_index - FORECAST_SIZE:train_end_index]

    def transform_data(self, df):
        return df['value'], df['value']

    def get_name(self):
        return "baseline"