import os

import pandas as pd

from keras import Sequential
from keras.layers import CuDNNLSTM, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

from dev.data_util import DataProcessor
from dev.models.abstract_model import AbstractModel
from dev.settings import FORECAST_SIZE
import numpy as np
from keras.models import load_model


class LSTMModel(AbstractModel):

    def __init__(self) -> None:
        super().__init__()
        self.num_days2see = 7
        self.scaler = None
        self.model = None

    def get_prediction(self, x_values, y_values, train_end_index, consumer_name):
        train_end_index_day = int(train_end_index / FORECAST_SIZE)

        file_name = 'dev/trained_models/model_lstm_series_{}.h5'.format(consumer_name)

        if os.path.isfile(file_name):
            if self.model is None:
                print("Found model for %s" % consumer_name)
                self.model = load_model(file_name)
        else:
            print("Training model for %s" % consumer_name)
            train_features = x_values[:train_end_index_day]
            train_labels = y_values[:train_end_index_day]

            self.model = self.lstm_model(num_hidden=64,
                                         num_days=self.num_days2see,
                                         feat_length=x_values.shape[2])
            self.model.fit(x=train_features,
                           y=train_labels,
                           validation_split=0.15,
                           epochs=100,
                           batch_size=64,
                           verbose=2)
            self.model.save(file_name)

        prediction = self.model.predict(x_values[train_end_index_day:train_end_index_day + 1],
                                        batch_size=64)
        return prediction[0] if len(prediction) > 0 else None

    def transform_data(self, df):
        df = df.copy()

        # create last known value
        df['last_known_value'] = df['value'].shift(periods=FORECAST_SIZE, freq=pd.offsets.Minute(30), axis=0)

        # cut first FORECAST_SIZE values, because they have NaN last_known_value
        df = df.drop(df.index[:FORECAST_SIZE], axis=0)

        hours = list(map(lambda x: x.hour, df.index))
        df['hour_sin'] = DataProcessor.map_datetime_to_circular(hours, np.sin)
        df['hour_cos'] = DataProcessor.map_datetime_to_circular(hours, np.cos)

        days = list(map(lambda x: x.day, df.index))
        df['days_sin'] = DataProcessor.map_datetime_to_circular(days, np.sin)
        df['days_cos'] = DataProcessor.map_datetime_to_circular(days, np.cos)

        weekdays = list(map(lambda x: x.weekday(), df.index))
        df['weekdays_sin'] = DataProcessor.map_datetime_to_circular(weekdays, np.sin)
        df['weekdays_cos'] = DataProcessor.map_datetime_to_circular(weekdays, np.cos)

        months = list(map(lambda x: x.month, df.index))
        df['months_sin'] = DataProcessor.map_datetime_to_circular(months, np.sin)
        df['months_cos'] = DataProcessor.map_datetime_to_circular(months, np.cos)

        # create categorical weather variables
        df = DataProcessor.create_dummies(df, "summary")
        df = DataProcessor.create_dummies(df, "precipType")

        df = DataProcessor.normalize_weather_category(df, "summary_Breezy and Mostly Cloudy")
        df = DataProcessor.normalize_weather_category(df, "summary_Breezy and Overcast")
        df = DataProcessor.normalize_weather_category(df, "summary_Breezy and Partly Cloudy")
        df = DataProcessor.normalize_weather_category(df, "summary_Windy and Mostly Cloudy")
        df = DataProcessor.normalize_weather_category(df, "summary_Windy and Overcast")

        values = df['value']
        combined = df.loc[:, df.columns != 'value']

        # The following lines scale the data except the value column.
        self.scaler = MinMaxScaler()
        columns = combined.columns.tolist()
        combined[columns] = self.scaler.fit_transform(combined.loc[:, combined.columns])
        combined = pd.concat([combined, values], axis=1)
        combined.dropna(axis=0, inplace=True)
        unique_dates = np.unique(combined.index.date[:-self.num_days2see])
        unique_dates = np.sort(unique_dates, axis=0).tolist()

        features = []
        target_values = []
        for ind, moment in enumerate(unique_dates):

            is_end = ind + self.num_days2see == len(unique_dates)

            combined_data = []
            for i in range(ind, ind + self.num_days2see):
                combined_data.append(combined.loc[str(unique_dates[i])])

            query_feat = pd.concat(combined_data, axis=0)

            if is_end:
                break

            query_target = combined.loc[str(unique_dates[ind + self.num_days2see])]

            enough = query_feat.shape[0] == FORECAST_SIZE * self.num_days2see and query_target.shape[0] == FORECAST_SIZE

            if not enough:
                print('Date : {}\t query_feat : {}\t query_target : {}'.format(str(moment), query_feat.shape[0],
                                                                               query_target.shape[0]))
                continue
            feat = query_feat.loc[:, query_feat.columns]
            target = query_target.loc[:, 'value']

            # IMPORTANT, DO NOT USE TARGET VARIABLE IN FEATURE SPACE:
            feat.drop('value', axis=1, inplace=True)

            features.append(np.expand_dims(feat.values, axis=0))
            target_values.append(np.expand_dims(target.values, axis=0))

        features = np.concatenate(features, axis=0)
        target_values = np.concatenate(target_values, axis=0)

        return features, target_values

    @staticmethod
    def lstm_model(num_hidden=50, feat_length=26, num_days=30, stateful_lstm=False):
        """
        Defines the LSTM Model for regression. The objective is to minimize MSE which is the same as minimizing RMSE.
        The optimizer used is RMSProp. Please do not change this optimizer.
        """
        model = Sequential()
        model.add(
            LSTM(units=num_hidden, input_shape=(num_days * FORECAST_SIZE, feat_length), stateful=stateful_lstm,
                 return_sequences=False))
        model.add(Dense(48, activation='relu'))
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model

    def get_name(self):
        return "lstm"
