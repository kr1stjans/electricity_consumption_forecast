from data_util import DataProcessor
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
import random

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', None)


def get_data(num_days2see=7):
    result = DataProcessor.load_data_as_separate_dataframes()
    combined = pd.concat(result, axis=0)

    print('stacked data frames BEFORE removing duplicate dates')
    print(combined.loc['2013-01-01 00:00:00'])
    combined.dropna(axis=0, how='any')
    combined = combined[~combined.index.duplicated(keep='first')]
    print('stacked data frames AFTER removing duplicate dates')
    print(combined.loc['2013-01-01 00:00:00'])
    exit(1)

    combined.dropna(axis=0, how='any')
    combined = combined[~combined.index.duplicated(keep='first')]
    # combined = combined.sort_index()
    combined.drop(['summary', 'precipType'], axis=1, inplace=True)
    values = combined['value']
    combined = combined.loc[:, combined.columns != 'value']

    # The following lines scale the data except the value column.
    scaler = MinMaxScaler()
    columns = combined.columns.tolist()
    combined[columns] = scaler.fit_transform(combined.loc[:, combined.columns])
    combined = pd.concat([combined, values], axis=1)
    combined.dropna(axis=0, inplace=True)
    unique_dates = np.unique(combined.index.date)
    unique_dates = np.sort(unique_dates, axis=0).tolist()
    total_dates = len(unique_dates)
    train_length = int(total_dates * 0.80)
    dates_train = unique_dates[:train_length + 1]
    dates_val = unique_dates[train_length + 1:]

    train_features = []
    train_values = []
    val_features = []
    val_values = []
    finished = False
    for ind, moment in enumerate(dates_train):
        is_end = ind + num_days2see == len(dates_train)
        combined_data = []
        for i in range(ind, ind + num_days2see):
            combined_data.append(combined.loc[str(dates_train[i])])
            print(combined.loc[str(dates_train[i])])
            exit(1)
        query_feat = pd.concat(combined_data, axis=0)
        if is_end:
            query_target = combined.loc[str(dates_val[0])]
            finished = True
        else:
            query_target = combined.loc[str(dates_train[ind + num_days2see])]

        enough = query_feat.shape[0] == 48 * num_days2see and query_target.shape[0] == 48

        if not enough:
            print('Date : {}\t query_feat : {}\t query_target : {}'.format(str(moment), query_feat.shape[0],
                                                                           query_target.shape[0]))
            continue
        feat = query_feat.loc[:, query_feat.columns]
        target = query_target.loc[:, 'value']
        train_features.append(np.expand_dims(feat.values, axis=0))
        train_values.append(np.expand_dims(target.values, axis=0))
        if finished:
            break

    train_features = np.concatenate(train_features, axis=0)
    train_values = np.concatenate(train_values, axis=0)

    for ind, moment in enumerate(dates_val):
        is_end = ind + num_days2see == len(dates_val)
        combined_data = []
        for i in range(ind, ind + num_days2see):
            combined_data.append(combined.loc[str(dates_val[i])])
        query_feat = pd.concat(combined_data, axis=0)
        if is_end:
            break

        query_target = combined.loc[str(dates_train[ind + 1])]

        enough = query_feat.shape[0] == 48 * num_days2see and query_target.shape[0] == 48

        if not enough:
            print('Date : {}\t query_feat : {}\t query_target : {}'.format(str(moment), query_feat.shape[0],
                                                                           query_target.shape[0]))
            continue

        feat = query_feat.loc[:, query_feat.columns]
        target = query_target.loc[:, 'value']

        val_features.append(np.expand_dims(feat.values, axis=0))
        val_values.append(np.expand_dims(target.values, axis=0))

    val_features = np.concatenate(val_features, axis=0)
    val_values = np.concatenate(val_values, axis=0)

    return train_features, train_values, val_features, val_values, scaler


if __name__ == "__main__":
    train_features, train_values, val_features, val_values = get_data()
    print(train_features.shape)
    print(val_features.shape)
    train_features, train_values, val_features, val_values = get_data(1)
    print(train_features.shape)
    print(val_features.shape)
