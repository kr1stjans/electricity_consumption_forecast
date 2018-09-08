import os

import numpy as np

import pandas as pd


class DataProcessor:

    @staticmethod
    def load_data_as_separate_dataframes(size=None):
        result = {}

        weather = pd.read_csv('data/weather.csv', sep=",", index_col='time', parse_dates=True,
                              infer_datetime_format=True,
                              usecols=['visibility', 'windBearing', 'temperature', 'dewPoint', 'pressure', 'time',
                                       'apparentTemperature', 'windSpeed', 'precipType', 'humidity', 'summary'])

        idx = 0
        for file in os.listdir('data/consumers'):
            idx += 1
            if size is not None and idx > size:
                break

            if 'MAC00' not in file:
                continue

            df = pd.read_csv('data/consumers/' + file, sep=',', header=0,
                             usecols=['dt', 'value'],
                             names=['dt', 'value'], index_col='dt',
                             parse_dates=True, memory_map=True,
                             infer_datetime_format=True, dtype={'value': np.float64})

            df = pd.merge(df, weather.copy(), how='left', left_index=True, right_index=True)

            # remove duplicated rows
            df = df[~df.index.duplicated(keep='first')]

            # fill missing indexes
            df = df.asfreq(pd.offsets.Minute(30), method='ffill')

            # fill missing values
            df = df.fillna(method='ffill', axis='index')

            # result must have no nulls or duplicated datetimes
            assert df.isnull().values.any() == False
            assert df.index.duplicated().any() == False

            # scale data
            result[file] = df

        return result

    @staticmethod
    def normalize_weather_category(df, column):
        if column not in df:
            print(column, 'not in df', df.shape)
            return df

        col1, col2 = column.split(" and ")
        if col1 in df:
            df[col1] = df[col1] | df[column]
        else:
            df[col1] = df[column]

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
            dummies = dummies.astype(np.bool)
            df = pd.concat([df, dummies], axis=1)
        return df

    @staticmethod
    def map_datetime_to_circular(data, fn):
        n = len(set(data))
        return [fn(2 * np.pi * x / n) + 1 for x in data]
