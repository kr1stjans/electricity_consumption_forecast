import datetime
import json
import logging

import pandas as pd
import requests
import sys
from sklearn import linear_model

import numpy as np
import pymssql

from prediction.data_preprocessor import Preprocessor


class PredictionManager(object):
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        credentials = json.load(open('credentials.json'))
        self.conn = pymssql.connect(host=credentials['host'], database=credentials['database'],
                                    user=credentials['user'],
                                    password=credentials['password'])
        self.curr = self.conn.cursor()

    def predict(self, measurement_places, subscriber):
        preprocessor = Preprocessor({})
        for ds_id in measurement_places:
            self.logger.info("Started forecasting for data source %s", ds_id)

            # get all known values for current measurement place
            self.curr.execute(
                "SELECT dt, value, season, season_weekday, season_weekday_hour, season_hour, month, weekday, hour, is_weekend, is_holiday, val_minus_96 FROM MLData WHERE mm_id=%s ORDER BY dt ASC",
                ds_id)
            data = np.array(self.curr.fetchall(), dtype=np.object_)

            if len(data) == 0:
                continue

            df_data = pd.DataFrame(data, columns=['dt', 'value', 'season', 'season_weekday', 'season_weekday_hour',
                                                  'season_hour', 'month', 'weekday', 'hour', 'is_weekend', 'is_holiday',
                                                  'val_minus_96'])

            # get last datetime with value
            self.curr.execute(
                "SELECT dt FROM MLData WHERE mm_id=%s AND value IS NOT NULL ORDER BY dt desc", ds_id)
            last_datetime_with_value = np.array(self.curr.fetchall(), dtype=np.object_)

            if len(last_datetime_with_value) > 0:
                start_testing_date = last_datetime_with_value[0][0]
                start_testing_date = start_testing_date.replace(minute=0, hour=0, second=0) - datetime.timedelta(
                    minutes=15)
            else:
                # if last forecast doesn't exist then start from 1.12.2017
                start_testing_date = datetime.datetime.strptime('2017-12-01 00:00:00', '%Y-%m-%d %H:%M:%S')

            # find start testing index based on last forecast
            start_testing_date_index = list(df_data.index[df_data['dt'] == start_testing_date])[0]

            # extract y
            y_values = df_data['value']

            # memorize datetimes for result and drop from features together with value
            datetime_values = list(df_data['dt'])
            df_data.drop('dt', axis=1, inplace=True)
            df_data.drop('value', axis=1, inplace=True)

            # features are tightly coupled so we have to construct them from train and prediction data together
            df_data = preprocessor.one_hot_encoding(df_data)

            self.logger.info("Features created for data source %s", ds_id)

            predictions_by_date = {}
            potential_by_date = {}

            # create forecasts by training on data before start_testing_date_index and predicting after it
            for test_date_index in range(start_testing_date_index, len(data) - Preprocessor.FORECAST_SIZE,
                                         96):
                x_train = df_data[:test_date_index]
                y_train = y_values[:test_date_index]
                x_test = df_data[test_date_index:test_date_index + Preprocessor.FORECAST_SIZE]

                model = linear_model.LinearRegression()
                model.fit(x_train, y_train)

                # prediction must not be negative
                predicted_values = [max(x, 0) for x in model.predict(X=x_test)]

                # minimum in last two weeks
                minimum_last_week = min(y_values[test_date_index - (96 * 14):test_date_index])

                # save predicted_values and potential vector
                for i in range(len(predicted_values)):
                    current_date = datetime_values[test_date_index + i].strftime('%Y-%m-%d %H:%M:%S')
                    predicted_value = predicted_values[i]

                    predictions_by_date[current_date] = predicted_value

                    # potential is minimum of 4 values time window - smallest value last two weeks
                    minimum = min(predicted_values[i - 2], min(predicted_values[i - 1], min(predicted_values[i], min(
                        predicted_values[i + 1] if i + 1 < len(predicted_values) else sys.maxsize,
                        predicted_values[i + 2] if i + 2 < len(predicted_values) else sys.maxsize))))

                    # potential must not be negative
                    potential_by_date[current_date] = max(minimum - minimum_last_week, 0)

            self.logger.info("Forecasts created for data source %s", ds_id)

            self.curr.executemany("UPDATE MLData SET prediction=%s WHERE dt=%s AND mm_id=%s",
                                  [(predicted_value, current_date, ds_id) for current_date, predicted_value in
                                   predictions_by_date.items()])
            self.conn.commit()

            self.logger.info("Features saved for data source %s", ds_id)

            # propagate ALL results back to Gema
            self.propagate({'dataSourceValuesGroupedByDate': {(ds_id + '_forecast'): predictions_by_date,
                                                              (ds_id + '_potential'): potential_by_date}}, subscriber)

            self.logger.info("Forecasts and potential propagated for data source %s", ds_id)

    @staticmethod
    def propagate(payload, subscriber):
        requests.post(subscriber, json=payload,
                      headers={'content-type': 'application/json'})
