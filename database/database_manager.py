import datetime
import json
import pyodbc

import logging

from prediction.data_preprocessor import Preprocessor


class DatabaseManager(object):
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        credentials = json.load(open('credentials.json'))
        self.conn = pyodbc.connect(DRIVER='{ODBC Driver 13 for SQL Server}',
                                   SERVER=credentials['host'],
                                   DATABASE=credentials['database'], UID=credentials['user'],
                                   PWD=credentials['password'])
        self.curr = self.conn.cursor()
        self.curr.fast_executemany = True

        self.curr.execute(
            "IF NOT EXISTS (select * from sysobjects where name='MLData' and xtype='U') CREATE TABLE MLData (value float(53), mm_id varchar(50), dt datetime, ctrl varchar(10), season varchar(20), season_weekday varchar(20), season_weekday_hour varchar(20), season_hour varchar(20), month varchar(20), weekday varchar(20), hour varchar(20), is_weekend tinyint, is_holiday tinyint, val_minus_96 float(53), prediction float(53))")
        self.conn.commit()

    def update_raw_data(self, data):
        for mm_id, values_by_date in data['dataSourceValuesGroupedByDate'].items():
            self.logger.info("Started processing data source %s with %s values", mm_id, len(values_by_date))

            value_start_dt = None
            value_start = None
            prediction_start_dt = None

            parsed_values_by_date = {}

            # find minimum and maximum datetime in new data
            for dt, value in values_by_date.items():
                parsed_dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
                # update prediction_start_dt. it must always be the maximal datetime so far
                prediction_start_dt = parsed_dt if prediction_start_dt is None else max(prediction_start_dt, parsed_dt)
                # update value_start_dt. it must always be the minimum datetime so far
                if value_start_dt is None or parsed_dt < value_start_dt:
                    value_start_dt = parsed_dt
                    value_start = value

                parsed_values_by_date[parsed_dt] = value

            if value_start_dt is None or prediction_start_dt is None:
                continue

            START_DT_CONST = value_start_dt

            # iterate from first new date to last new date in 15 minute steps and fill missing values with last known value
            while value_start_dt < prediction_start_dt:
                value_start_dt += datetime.timedelta(minutes=15)
                if value_start_dt not in parsed_values_by_date:
                    parsed_values_by_date[value_start_dt] = value_start
                else:
                    value_start = parsed_values_by_date[value_start_dt]

            self.logger.info("Filled missing data for data source %s", mm_id)

            # get existing data for current data source
            new_datetimes = list(parsed_values_by_date.keys())
            existing_datetimes = set()
            batch_size = 1000
            for i in range(0, len(new_datetimes), batch_size):
                size = (len(new_datetimes) - i) if i + batch_size > len(new_datetimes) else batch_size
                query = ("SELECT dt FROM MLData WHERE dt in (%s)" % ', '.join(
                    ['?'] * size)) + " AND mm_id=? ORDER BY dt ASC"
                # select current batch of new datetimes for parameters
                selected_datetimes_params = new_datetimes[i:i + size]
                # append data source id as last param
                selected_datetimes_params.append(mm_id)
                # fetch existing datetimes
                self.curr.execute(query, tuple(selected_datetimes_params))
                existing_datetimes.update(set(dt[0] for dt in self.curr.fetchall()))

            self.logger.info("%s existing datapoints for data source %s", len(existing_datetimes), mm_id)

            # list for preprocessing new nonexisting values
            preprocess_queue = dict()
            # list for simply updating existing values in db
            to_update = []
            for dt, value in parsed_values_by_date.items():
                # only need to update value for all existing data
                if dt in existing_datetimes:
                    to_update.append((value, dt, mm_id))
                else:  # else queue the data for preprocessing
                    preprocess_queue[dt] = value

            if len(to_update) > 0:
                # update values
                self.curr.executemany("UPDATE MLData SET value=? WHERE dt=? AND mm_id=?", to_update)
                self.conn.commit()
                self.logger.info("Updated %s existing data points for data source %s", len(to_update), mm_id)
                # update all relevant val_minus_96
                val_minus_96_to_update = []
                # get all data points since first new data point - 2 days
                existing_data_points_by_dt = {}
                self.curr.execute("SELECT dt, value FROM MLData WHERE mm_id=? AND dt>? ORDER BY dt ASC",
                                  (mm_id, START_DT_CONST - datetime.timedelta(days=2)))
                existing_data_points_list = self.curr.fetchall()
                # map data points by date
                for dt, value in existing_data_points_list:
                    existing_data_points_by_dt[dt] = value
                # calculate val minus 96 for all new datetimes. skip first two days that are used only as reference point
                for dt, _ in existing_data_points_list[Preprocessor.FORECAST_SIZE:]:
                    # get value from one or two days ago
                    yesterday = dt - datetime.timedelta(days=1)
                    yesterday_value = existing_data_points_by_dt[
                        yesterday] if yesterday in existing_data_points_by_dt else None
                    day_before_yesterday = dt - datetime.timedelta(days=2)
                    day_before_yesterday_value = existing_data_points_by_dt[
                        day_before_yesterday] if day_before_yesterday in existing_data_points_by_dt else None

                    if day_before_yesterday_value is None and yesterday_value is None:
                        raise Exception("cant find val_minus_96 for dt: %s and mm_id: %s", dt, mm_id)

                    val_minus_96_to_update.append(
                        (yesterday_value or day_before_yesterday_value, dt, mm_id))

                # update val minus 96 for each dt
                if len(val_minus_96_to_update) > 0:
                    self.curr.executemany("UPDATE MLData SET val_minus_96=? WHERE dt=? AND mm_id=?",
                                          val_minus_96_to_update)
                    self.logger.info("Updated %s val_minus_96 values for mm_id: %s", len(val_minus_96_to_update), mm_id)
                    self.conn.commit()

            # number of non existing prediction holders (intervals) is calculated as 192 -
            # nr_of_intervals_between(last_date_with_prediction, last_date_with_value)
            nr_of_intervals = Preprocessor.FORECAST_SIZE

            # get last row in db
            self.curr.execute("SELECT TOP 1 dt FROM MLData WHERE mm_id=? ORDER BY dt DESC", mm_id)
            last_row_result = self.curr.fetchall()

            if len(last_row_result) > 0:
                # get last row as first element of array and first data of tuple (datetime)
                last_row_dt_from_db = last_row_result[0][0]

                self.curr.execute("SELECT TOP 1 dt FROM MLData WHERE value IS NOT NULL AND mm_id=? ORDER BY dt DESC",
                                  mm_id)
                last_known_value = self.curr.fetchall()
                # get last date as maximum of existing dt from DB or new dt
                last_dt_with_value = max(last_known_value[0][0], prediction_start_dt) if len(
                    last_known_value) > 0 else prediction_start_dt

                # adjust nr of required prediction intervals if last row datetime is greater than last value datetime
                if last_row_dt_from_db > last_dt_with_value:
                    nr_of_intervals = nr_of_intervals - int(
                        (last_row_dt_from_db - last_dt_with_value).total_seconds() / (15 * 60))

                prediction_start_dt = max(prediction_start_dt, last_row_dt_from_db)

            # create missing prediction holders
            for _ in range(nr_of_intervals):
                prediction_start_dt += datetime.timedelta(minutes=15)
                preprocess_queue[prediction_start_dt] = None

            # load last FORECAST_SIZE known values from DB to ensure we always have history data for value_t_minus_96
            self.curr.execute(
                "SELECT TOP 192 dt, value FROM MLData WHERE mm_id=? AND value IS NOT NULL ORDER BY dt DESC", mm_id)
            existing_data = dict((dt, prediction) for dt, prediction in self.curr.fetchall())

            # build features for new data
            features = Preprocessor({}).build_features(preprocess_queue, existing_data)

            if len(features) > 0:
                # batch insert new features
                self.curr.executemany(
                    "INSERT INTO MLData (mm_id, dt, value, season, season_weekday, season_weekday_hour, season_hour, month, weekday, hour, is_weekend, is_holiday, val_minus_96) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(mm_id, f['dt'], f['value'], f['season'], f['season_weekday'], f['season_weekday_hour'],
                      f['season_hour'], f['month'], f['weekday'], f['hour'], f['is_weekend'], f['is_holiday'],
                      f['val_minus_96']) for f in features])
                self.conn.commit()
                self.logger.info("Inserted %s new rows for data source %s", len(features), mm_id)
