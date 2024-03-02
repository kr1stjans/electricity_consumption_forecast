import pandas as pd
from statsmodels.tsa.ar_model import AR

from src.settings import FORECAST_SIZE


class AutoregressiveModel:
    @staticmethod
    def get_forecast(x_values, y_values, train_end_index):
        model = AR(x_values[:train_end_index], dates=x_values[:train_end_index].index, freq=pd.offsets.Minute(30))
        fitted_model = model.fit(maxlag=48)
        return fitted_model.predict(start=train_end_index, end=train_end_index + (FORECAST_SIZE - 1))

    @staticmethod
    def transform_data(df):
        return df['value'], df['value']
