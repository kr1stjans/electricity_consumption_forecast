import os
from keras.models import Sequential
from keras.layers import Dense, Activation, CuDNNLSTM, TimeDistributed, Input, Conv2D, Flatten, AveragePooling2D
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import argparse
from dev.get_data_modified import get_data
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--exppath')
parser.add_argument('--num_days', default=7)
parser.add_argument('--num_hidden', default=256)
parser.add_argument('--numepochs', default=50)


def lstm_model(num_hidden=50, feat_length=9, num_days=30, stateful_lstm=False):
    model = Sequential()
    model.add(Conv2D(36, 3, activation='selu', input_shape=(48 * num_days, feat_length, 1)))
    # model.add(Conv2D(48, 5, activation='selu'))
    model.add(AveragePooling2D(pool_size=(5, 1)))
    model.add(Flatten())
    # model.add(Dense(1024, activation='selu'))
    model.add(Dense(48, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model


if __name__ == "__main__":
    args = parser.parse_args()
    exp_path = args.exppath
    os.makedirs(exp_path, exist_ok=True)
    num_hidden = int(args.num_hidden)
    num_days = int(args.num_days)
    train_features, train_labels, val_features, val_labels, minmaxscalar = get_data(num_days)
    train_features = np.expand_dims(train_features, axis=3)
    val_features = np.expand_dims(val_features, axis=3)
    joblib.dump(minmaxscalar, os.path.join(exp_path, 'minmaxscalar.file'))
    model = lstm_model(num_hidden=num_hidden, num_days=num_days)
    print(model.summary())
    history = model.fit(train_features, train_labels, epochs=100, batch_size=72, verbose=2,
                        validation_data=(val_features, val_labels), shuffle=True)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    plt.close()
    model.save(os.path.join(exp_path, 'model_conv.h5'))
