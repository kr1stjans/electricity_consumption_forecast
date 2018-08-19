import collections
import logging

import os
from functools import reduce

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
import datetime


# from sklearn.decomposition import PCA
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.preprocessing import MinMaxScaler


class Preprocessor(TransformerMixin):
    """
    Preprocessor is used to transform given (value, date) data tuple to complete X feature matrix.
    """

    FORECAST_SIZE = 96

    def __init__(self, debug=True):
        self.logger = logging.getLogger(__name__)

        self.debug = debug

        self.Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
        self.seasons = [('winter', (datetime.date(self.Y, 1, 1), datetime.date(self.Y, 3, 20))),
                        ('spring', (datetime.date(self.Y, 3, 21), datetime.date(self.Y, 6, 20))),
                        ('summer', (datetime.date(self.Y, 6, 21), datetime.date(self.Y, 9, 22))),
                        ('autumn', (datetime.date(self.Y, 9, 23), datetime.date(self.Y, 12, 20))),
                        ('winter', (datetime.date(self.Y, 12, 21), datetime.date(self.Y, 12, 31)))]
        # TODO: hardcoded holidays should be replaced with a proper solution
        self.holidays = [datetime.date(2016, 1, 1), datetime.date(2016, 2, 8), datetime.date(2016, 3, 28),
                         datetime.date(2016, 4, 27),
                         datetime.date(2016, 5, 1), datetime.date(2016, 5, 2), datetime.date(2016, 6, 25),
                         datetime.date(2016, 8, 15), datetime.date(2016, 10, 31), datetime.date(2016, 11, 1),
                         datetime.date(2016, 12, 25),
                         datetime.date(2016, 12, 26),
                         datetime.date(2017, 1, 1), datetime.date(2017, 1, 2), datetime.date(2017, 2, 8),
                         datetime.date(2017, 4, 17), datetime.date(2017, 4, 27),
                         datetime.date(2017, 5, 1), datetime.date(2017, 5, 2), datetime.date(2017, 6, 25),
                         datetime.date(2017, 8, 15), datetime.date(2017, 10, 31), datetime.date(2017, 11, 1),
                         datetime.date(2017, 12, 25),
                         datetime.date(2017, 12, 26),
                         datetime.date(2018, 1, 1), datetime.date(2018, 1, 2), datetime.date(2018, 2, 8),
                         datetime.date(2018, 4, 2), datetime.date(2018, 4, 27), datetime.date(2018, 5, 1),
                         datetime.date(2018, 5, 2), datetime.date(2018, 6, 25), datetime.date(2018, 8, 15),
                         datetime.date(2018, 10, 31), datetime.date(2018, 11, 1), datetime.date(2018, 12, 25),
                         datetime.date(2018, 12, 26)]

    @staticmethod
    def create_dummies(data, column_name):
        dummies = pd.get_dummies(data[column_name], prefix=column_name, drop_first=True)

        # always drop the column, because either the values are dummified
        # or there is only one value, which makes column useless
        data.drop(column_name, axis=1, inplace=True)
        if not dummies.empty:
            dummies = dummies.astype(np.int8)
            data = pd.concat([data, dummies], axis=1)
        return data

    @staticmethod
    def is_invalid_value(value):
        return value == 'ED6' and value == 'ED8'

    def is_slovenian_holiday(self, now):
        if isinstance(now, datetime.datetime):
            now = now.date()
        for holiday in self.holidays:
            if holiday == now:
                return True
        return False

    def get_season(self, now):
        if isinstance(now, datetime.datetime):
            now = now.date()
        now = now.replace(year=self.Y)
        return next(season for season, (start, end) in self.seasons
                    if start <= now <= end)

    def build_features(self, values_by_date, existing_data):
        features = []

        for dt, value in values_by_date.items():
            yesterday = dt - datetime.timedelta(days=1)
            yesterday_value = values_by_date[yesterday] if yesterday in values_by_date else None or existing_data[
                yesterday] if yesterday in existing_data else None

            day_before_yesterday = dt - datetime.timedelta(days=2)
            day_before_yesterday_value = values_by_date[
                day_before_yesterday] if day_before_yesterday in values_by_date else None or existing_data[
                day_before_yesterday] if day_before_yesterday in existing_data else None

            last_known_value = yesterday_value or day_before_yesterday_value

            if last_known_value is None:
                continue

            weekday = dt.weekday()
            season = str(self.get_season(dt))
            features.append(
                {'dt': dt,
                 'value': value,
                 'season': season,
                 'season_weekday': season + '_' + str(weekday),
                 'season_weekday_hour': season + '_' + str(weekday) + '_' + str(dt.hour),
                 'season_hour': season + '_' + str(dt.hour),
                 'month': dt.month,
                 'weekday': weekday,
                 'hour': str(dt.hour),
                 'is_weekend': 1 if weekday == 5 or weekday == 6 else 0,
                 'is_holiday': self.is_slovenian_holiday(dt),
                 'val_minus_96': last_known_value
                 })

        return features

    def one_hot_encoding(self, features):
        # get dummies from categorical features
        features = Preprocessor.create_dummies(features, 'season')
        features = Preprocessor.create_dummies(features, 'season_weekday')
        features = Preprocessor.create_dummies(features, 'season_weekday_hour')
        features = Preprocessor.create_dummies(features, 'season_hour')
        features = Preprocessor.create_dummies(features, 'month')
        features = Preprocessor.create_dummies(features, 'weekday')
        features = Preprocessor.create_dummies(features, 'hour')
        return features

    def get_last_known_value(self, values_by_date, current_dt):
        # get last known value from either from new data or existing data
        yesterday = current_dt - datetime.timedelta(days=1)
        last_known_value = values_by_date[yesterday] if yesterday in values_by_date else None

        if last_known_value is None:
            self.logger.error(
                "last known value is None. this should be impossible as we preprocess the data and fill the gaps!")
        return last_known_value

    def get_value_based_features(self, deque):
        n = len(deque)
        n_string = str(n)

        result = {}
        sum_result = sum(deque)

        # result['min_last_' + n_string] = min(deque)
        # result['max_last_' + n_string] = max(deque)
        # result['sum_last_' + n_string] = sum_result
        result['avg_last_' + n_string] = sum_result / n
        # result['median_last_' + n_string] = reduce(lambda x, y: x + y, deque) / n
        # result['diff_start_end_last_' + n_string] = abs(deque[0] - deque[-1])
        # result['diff_max_min_diff_last_' + n_string] = result['max_last_' + n_string] - result['min_last_' + n_string]

        return result

    def fit_transform(self, X, y=None, **fit_params):

        if len(X) != len(y):
            raise Exception("X and Y dimensions must match")

        if not os.path.isfile('cache/-1_x_data.pickle'):
            print("Transforming data for mm_id=-1")

            # get dummies from categorical features
            # x_values = self.one_hot_encoding(x_values)


            pd.to_pickle(X, 'cache/-1_x_data.pickle')
            pd.to_pickle(y, 'cache/-1_y_data.pickle')
        else:
            print("Found cached data for mm_id=-1")
            x_values = pd.read_pickle('cache/-1_x_data.pickle')
            y = pd.read_pickle('cache/-1_y_data.pickle')

        # run pca to eliminate features
        # pca = PCA(n_components=100)
        # pca.fit(x_values)
        # print(pca.explained_variance_ratio_)
        # x_values = pca.transform(x_values)

        # normalize
        # x_values = VarianceThreshold(threshold=0.01).fit_transform(x_values)

        # x_values = MinMaxScaler().fit_transform(x_values)

        # ridge = Ridge(alpha=10)
        # ridge.fit(x_values, y_values)
        # DataManager.print_cooficients(columns, ridge.coef_)

        return x_values, y
