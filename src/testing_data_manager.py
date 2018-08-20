import numpy

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FORECAST_SIZE = 48


class DataProcessor:

    @staticmethod
    def get_public_data():
        df = pd.read_csv('../data/data_MAC000002.csv', sep=',', header=None, usecols=['dt', 'value'],
                         names=['id', 'type', 'dt', 'value', 'subgroup', 'group'], index_col='dt',
                         parse_dates=True,
                         infer_datetime_format=True, dtype={'value': numpy.float64})

        weather = pd.read_csv('../data/weather.csv', sep=",", index_col='time', parse_dates=True,
                              infer_datetime_format=True,
                              usecols=['visibility', 'windBearing', 'temperature', 'dewPoint', 'pressure', 'time',
                                       'apparentTemperature', 'windSpeed', 'precipType', 'humidity', 'summary'])

        merged = pd.merge(df, weather, how='left', left_index=True, right_index=True)
        merged.fillna(method='bfill', inplace=True, axis='index')

        return merged

    @staticmethod
    def normalize_category(df, column):
        col1, col2 = column.split(" and ")
        df[col1] = df[col1] | df[column] if col1 in df else df[column]
        df["summary_" + col2] = df["summary_" + col2] | df[column] if "summary_" + col2 in df else df[column]
        df.drop(column, axis=1, inplace=True)
        return df

    @staticmethod
    def create_dummies(df, column_name):
        dummies = pd.get_dummies(df[column_name], prefix=column_name, drop_first=True)

        # always drop the column, because either the df are dummified
        # or there is only one value, which makes column useless
        df.drop(column_name, axis=1, inplace=True)
        if not dummies.empty:
            dummies = dummies.astype(numpy.bool)
            df = pd.concat([df, dummies], axis=1)
        return df

    @staticmethod
    def map_datetime_to_circular(data, fn):
        n = len(set(data))
        return [fn(2 * numpy.pi * x / n) + 1 for x in data]

    @staticmethod
    def get_LR_transform(df):
        # extract target variable
        y = df['value']

        # create last known value
        df['last_known_value'] = df['value'].shift(FORECAST_SIZE, axis='index')

        # drop target variable from data set
        df.drop('value', axis=1, inplace=True)

        # cut first FORECAST_SIZE values, because they have NaN last_known_value
        df.drop(df.index[:FORECAST_SIZE], inplace=True)

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

        df = DataProcessor.normalize_category(df, "summary_Breezy and Mostly Cloudy")
        df = DataProcessor.normalize_category(df, "summary_Breezy and Overcast")
        df = DataProcessor.normalize_category(df, "summary_Breezy and Partly Cloudy")
        df = DataProcessor.normalize_category(df, "summary_Windy and Mostly Cloudy")
        df = DataProcessor.normalize_category(df, "summary_Windy and Overcast")

        # normalize between -1 and 1
        X = MinMaxScaler(feature_range=(0, 1)).fit_transform(df)
        return df, y
