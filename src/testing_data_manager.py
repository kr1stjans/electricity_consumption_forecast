from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.main_final import FORECAST_SIZE
from src.tools import Connection


class DataManager(object):
    """
    DataManager is used for data retrieval. It returns either already preprocessed cached data or it fetches it from DB 
    and prepossesses it.
    """

    def __init__(self) -> None:
        self.connection = Connection()
        self.conn = self.connection.connection()
        self.curr = self.connection.cursor()

    @staticmethod
    def print_cooficients(columns, coef):
        result = list(zip(columns, [abs(x) for x in coef]))
        result.sort(key=itemgetter(1), reverse=True)
        result = [(x, '{0:.2f}'.format(y)) for (x, y) in result]
        for cooef in result:
            print(cooef)

    def get_data(self, mm_id):
        '''
        Method gets measurements and weather data from 1.1 2016 to 1.5.2017 sorted by date and concatenates them together.
        :param mm_id:
        :return:
        '''
        self.curr.execute("SELECT value FROM MERGED_DATA WHERE mm_id=? ORDER BY dt ASC", mm_id)
        measurements = np.array(self.curr.fetchall())

        self.curr.execute(
            'SELECT Cas, Temperatura, Obsevanje, Veter, smer_vetra, vetrni_potencial, relativna_vlaga '
            'FROM Vreme ORDER BY Cas ASC')
        weather = np.array(self.curr.fetchall())

        return np.hstack((measurements, weather))

    @staticmethod
    def map_datetime_to_circular(data, fn):
        n = len(set(data))
        return [fn(2 * np.pi * x / n) + 1 for x in data]

    @staticmethod
    def get_public_data():
        values = pd.read_csv('../data/data_MAC000002.csv', sep=',', header=None, usecols=['dt', 'value'],
                             names=['id', 'type', 'dt', 'value', 'subgroup', 'group'], index_col='dt',
                             parse_dates=True,
                             infer_datetime_format=True, dtype={'value': np.float64})

        # extract target variable
        y = values['value']

        # create last known value
        values['last_known_value'] = values['value'].shift(FORECAST_SIZE, axis='index')

        # drop target variable from data set
        values.drop('value', axis=1, inplace=True)

        # create cyclic time features
        hours = list(map(lambda x: x.hour, values.index))
        values['hour_sin'] = DataManager.map_datetime_to_circular(hours, np.sin)
        values['hour_cos'] = DataManager.map_datetime_to_circular(hours, np.cos)

        days = list(map(lambda x: x.day, values.index))
        values['days_sin'] = DataManager.map_datetime_to_circular(days, np.sin)
        values['days_cos'] = DataManager.map_datetime_to_circular(days, np.cos)

        weekdays = list(map(lambda x: x.weekday(), values.index))
        values['weekdays_sin'] = DataManager.map_datetime_to_circular(weekdays, np.sin)
        values['weekdays_cos'] = DataManager.map_datetime_to_circular(weekdays, np.cos)

        months = list(map(lambda x: x.month, values.index))
        values['months_sin'] = DataManager.map_datetime_to_circular(months, np.sin)
        values['months_cos'] = DataManager.map_datetime_to_circular(months, np.cos)

        weather = pd.read_csv('../data/weather.csv', sep=",", index_col='time', parse_dates=True,
                              infer_datetime_format=True)

        merged = pd.merge(values, weather, how='left', left_index=True, right_index=True)
        merged.fillna(method='bfill', inplace=True, axis='index')

        # normalize between -1 and 1
        result = MinMaxScaler(feature_range=(-1, 1)).fit_transform(merged)

        return result
