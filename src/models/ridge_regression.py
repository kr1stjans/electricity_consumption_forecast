import pandas as pd

import numpy

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler

from src.models.abstract_model import AbstractModel
from src.settings import FORECAST_SIZE
from src.data_util import DataProcessor


class RidgeRegressionModel(AbstractModel):
    @staticmethod
    def get_forecast(x_values, y_values, train_end_index):
        x_test = x_values[train_end_index:train_end_index + FORECAST_SIZE]
        model = Ridge(alpha=0.05)
        model.fit(x_values[:train_end_index], y_values[:train_end_index])
        return model.predict(X=x_test)

    @staticmethod
    def transform_data(df):
        # create last known value
        df['last_known_value'] = df['value'].shift(periods=FORECAST_SIZE, freq=pd.offsets.Minute(30), axis=0)

        # cut first FORECAST_SIZE values, because they have NaN last_known_value
        df = df.drop(df.index[:FORECAST_SIZE], axis=0)

        # extract target variable
        y = df['value']
        df.drop('value', axis=1, inplace=True)

        # create cyclic time features
        hours = list(map(lambda x: x.hour, df.index))
        df['hour_sin'] = DataProcessor.map_datetime_to_circular(hours, numpy.sin)
        df['hour_cos'] = DataProcessor.map_datetime_to_circular(hours, numpy.cos)

        days = list(map(lambda x: x.day, df.index))
        df['days_sin'] = DataProcessor.map_datetime_to_circular(days, numpy.sin)
        df['days_cos'] = DataProcessor.map_datetime_to_circular(days, numpy.cos)

        weekdays = list(map(lambda x: x.weekday(), df.index))
        df['weekdays_sin'] = DataProcessor.map_datetime_to_circular(weekdays, numpy.sin)
        df['weekdays_cos'] = DataProcessor.map_datetime_to_circular(weekdays, numpy.cos)

        months = list(map(lambda x: x.month, df.index))
        df['months_sin'] = DataProcessor.map_datetime_to_circular(months, numpy.sin)
        df['months_cos'] = DataProcessor.map_datetime_to_circular(months, numpy.cos)

        # create categorical weather variables
        df = DataProcessor.create_dummies(df, "summary")
        df = DataProcessor.create_dummies(df, "precipType")

        df = DataProcessor.normalize_weather_category(df, "summary_Breezy and Mostly Cloudy")
        df = DataProcessor.normalize_weather_category(df, "summary_Breezy and Overcast")
        df = DataProcessor.normalize_weather_category(df, "summary_Breezy and Partly Cloudy")
        df = DataProcessor.normalize_weather_category(df, "summary_Windy and Mostly Cloudy")
        df = DataProcessor.normalize_weather_category(df, "summary_Windy and Overcast")

        # normalize between -1 and 1
        scaled_df = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df)
        df = pd.DataFrame(scaled_df, df.index, df.columns)

        return df, y
