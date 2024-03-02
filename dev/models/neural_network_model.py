import hashlib
import os

import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.utils import multi_gpu_model
from sklearn.preprocessing import MinMaxScaler

from dev.data_util import DataProcessor
from dev.models.abstract_model import AbstractModel
from dev.settings import FORECAST_SIZE
import numpy as np


class NeuralNetworkModel(AbstractModel):
    def __init__(self, num_hidden=32, num_days2_see=1, epoch=512, batch_size=32, use_gpu=True,
                 loss='mean_squared_error', optimizer='rmsprop') -> None:
        super().__init__()
        self.num_days2see = num_days2_see
        self.x_scaler = None
        self.model = None
        self.gpu_model = None
        self.use_gpu = use_gpu
        self.num_hidden = num_hidden
        self.epoch = epoch
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        # self.callback_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        # self.callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-4, patience=0, verbose=1)

    def hash_parameters(self):
        params = (str(self.get_name()) + str(self.num_days2see) + str(self.epoch) + str(self.batch_size) + str(
            self.num_hidden)).encode("utf-8")
        return hashlib.md5(params).hexdigest()

    def build_model(self, num_hidden, feat_length, num_days):
        if self.use_gpu:
            self.gpu_model = multi_gpu_model(self.model, gpus=3)
            self.gpu_model.compile(loss=self.loss, optimizer=self.optimizer)
        else:
            self.model.compile(loss=self.loss, optimizer=self.optimizer)

    def get_prediction(self, x_values, y_values, train_end_index, consumer_name):
        train_end_index_day = int(train_end_index / FORECAST_SIZE)

        file_name = 'dev/trained_models/model_{}_consumer_{}.h5'.format(self.hash_parameters(), consumer_name)

        if os.path.isfile(file_name):
            if self.model is None:
                print("Found model for %s" % file_name)
                self.model = load_model(file_name)
        else:
            print("Training model for %s" % file_name)
            train_features = x_values[:train_end_index_day]
            train_labels = y_values[:train_end_index_day]

            self.build_model(num_hidden=self.num_hidden,
                             num_days=self.num_days2see,
                             feat_length=x_values.shape[2])

            model_to_use = self.gpu_model if self.use_gpu else self.model

            model_to_use.fit(x=train_features,
                             y=train_labels,
                             validation_split=0.15,
                             epochs=self.epoch,
                             batch_size=self.batch_size,
                             verbose=2)

            # always save initial model without GPU
            self.model.save(file_name)

        # make prediction with initial model without GPU
        prediction = self.model.predict(x_values[train_end_index_day:train_end_index_day + 1],
                                        batch_size=self.batch_size)
        return prediction[0] if len(prediction) > 0 else None

    def batch_generator(self, batch_size, sequence_length, x_train, y_train):

        while True:
            x_shape = (batch_size, sequence_length, x_train.shape[1])
            x_batch = np.zeros(shape=x_shape, dtype=np.float16)

            y_shape = (batch_size, sequence_length, y_train.shape[1])
            y_batch = np.zeros(shape=y_shape, dtype=np.float16)

            for i in range(batch_size):
                idx = np.random.randint(x_train.shape[0] - sequence_length)

                x_batch[i] = x_train[idx:idx + sequence_length]
                y_batch[i] = y_train[idx:idx + sequence_length]

            yield (x_batch, y_batch)

    def transform_data(self, df):
        df = df.copy()

        hours = list(map(lambda x: x.hour, df.index))
        df['hour_sin'] = DataProcessor.map_datetime_to_circular(hours, np.sin)
        df['hour_cos'] = DataProcessor.map_datetime_to_circular(hours, np.cos)

        days = list(map(lambda x: x.day, df.index))
        df['days_sin'] = DataProcessor.map_datetime_to_circular(days, np.sin)
        df['days_cos'] = DataProcessor.map_datetime_to_circular(days, np.cos)

        weekdays = list(map(lambda x: x.weekday(), df.index))
        df['weekdays_sin'] = DataProcessor.map_datetime_to_circular(weekdays,
                                                                    np.sin)
        df['weekdays_cos'] = DataProcessor.map_datetime_to_circular(weekdays,
                                                                    np.cos)

        months = list(map(lambda x: x.month, df.index))
        df['months_sin'] = DataProcessor.map_datetime_to_circular(months,
                                                                  np.sin)
        df['months_cos'] = DataProcessor.map_datetime_to_circular(months,
                                                                  np.cos)

        # create categorical weather variables
        df = DataProcessor.create_dummies(df, "summary")
        df = DataProcessor.create_dummies(df, "precipType")

        df = DataProcessor.normalize_weather_category(df,
                                                      "summary_Breezy and Mostly Cloudy")
        df = DataProcessor.normalize_weather_category(df,
                                                      "summary_Breezy and Overcast")
        df = DataProcessor.normalize_weather_category(df,
                                                      "summary_Breezy and Partly Cloudy")
        df = DataProcessor.normalize_weather_category(df,
                                                      "summary_Windy and Mostly Cloudy")
        df = DataProcessor.normalize_weather_category(df,
                                                      "summary_Windy and Overcast")

        # create last known value
        df['target_value'] = df['value'].shift(periods=-FORECAST_SIZE,
                                               freq=pd.offsets.Minute(30),
                                               axis=0)

        # cut last FORECAST_SIZE values, because they have NaN last_known_value
        df = df.drop(df.index[-FORECAST_SIZE:], axis=0)

        target_vector = df['target_value']
        df = df.drop('target_value', axis=1)

        self.x_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.y_scaler = MinMaxScaler(feature_range=(-1, 1))

        df = self.x_scaler.fit_transform(df)
        target_vector = self.y_scaler.fit_transform(target_vector.reshape(-1, 1))

        unique_dates = np.unique(df.index.date[:-self.num_days2see])
        unique_dates = np.sort(unique_dates, axis=0).tolist()

        features = []
        target_values = []
        for ind, moment in enumerate(unique_dates):

            is_end = ind + self.num_days2see == len(unique_dates)

            combined_data = []
            for i in range(ind, ind + self.num_days2see):
                combined_data.append(df.loc[str(unique_dates[i])])

            query_feat = pd.concat(combined_data, axis=0)

            if is_end:
                break

            query_target = df.loc[
                str(unique_dates[ind + self.num_days2see])]

            enough = query_feat.shape[
                         0] == FORECAST_SIZE * self.num_days2see and \
                     query_target.shape[0] == FORECAST_SIZE

            if not enough:
                print('Date : {}\t query_feat : {}\t query_target : {}'.format(
                    str(moment), query_feat.shape[0],
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
