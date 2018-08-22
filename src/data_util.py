import os

import numpy

import pandas as pd


class DataProcessor:
    @staticmethod
    def load_data_as_separate_dataframes():
        result = []

        weather = pd.read_csv('../data/weather.csv', sep=",", index_col='time', parse_dates=True,
                              infer_datetime_format=True,
                              usecols=['visibility', 'windBearing', 'temperature', 'dewPoint', 'pressure', 'time',
                                       'apparentTemperature', 'windSpeed', 'precipType', 'humidity', 'summary'])

        for file in os.listdir('../data'):
            if 'MAC00' not in file:
                continue

            consumer_df = pd.read_csv('../data/' + file, sep=',', header=None,
                                      usecols=['dt', 'value'],
                                      names=['id', 'type', 'dt', 'value', 'subgroup', 'group'], index_col='dt',
                                      parse_dates=True,
                                      infer_datetime_format=True, dtype={'value': numpy.float64})
            df = pd.merge(weather.copy(), consumer_df, how='inner', left_index=True, right_index=True)

            # remove duplicated rows
            df = df[~df.index.duplicated(keep='first')]

            # fill missing indexes
            df = df.asfreq(pd.offsets.Minute(30))

            # fill missing values
            df.fillna(method='backfill', inplace=True, axis='index')

            # result must have no nulls or duplicated datetimes
            assert df.isnull().values.any() == False
            assert df.index.duplicated().any() == False

            result.append(df)

        return result

    @staticmethod
    def get_single_public_data(consumer_id):
        df = DataProcessor.load_data_in_single_dataframe()
        for col in df.columns:
            if 'value' in col:
                if 'value' + str(consumer_id) == col:
                    df['value'] = df[col]
                df.drop(col, axis=1, inplace=True)
        return df

    @staticmethod
    def load_data_in_single_dataframe():
        df = pd.read_csv('../data/weather.csv', sep=",", index_col='time', parse_dates=True,
                         infer_datetime_format=True,
                         usecols=['visibility', 'windBearing', 'temperature', 'dewPoint', 'pressure', 'time',
                                  'apparentTemperature', 'windSpeed', 'precipType', 'humidity', 'summary'])

        for i in range(2, 7, 1):
            consumer_df = pd.read_csv('../data/consumers/MAC00000' + str(i) + '.csv', sep=',', header=None,
                                      usecols=['dt', 'value' + str(i)],
                                      names=['id', 'type', 'dt', 'value' + str(i), 'subgroup', 'group'], index_col='dt',
                                      parse_dates=True,
                                      infer_datetime_format=True, dtype={'value' + str(i): numpy.float64})
            df = pd.merge(df, consumer_df, how='inner', left_index=True, right_index=True)

        # remove duplicated rows
        df = df[~df.index.duplicated(keep='first')]

        # fill missing indexes
        df = df.asfreq(pd.offsets.Minute(30))

        # fill missing values
        df.fillna(method='backfill', inplace=True, axis='index')

        # result must have no nulls or duplicated datetimes
        assert df.isnull().values.any() == False
        assert df.index.duplicated().any() == False

        return df

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
            dummies = dummies.astype(numpy.bool)
            df = pd.concat([df, dummies], axis=1)
        return df

    @staticmethod
    def map_datetime_to_circular(data, fn):
        n = len(set(data))
        return [fn(2 * numpy.pi * x / n) + 1 for x in data]
