import os
from keras import Sequential
from keras.layers import Dense, CuDNNLSTM, LSTM
from sklearn.externals import joblib
import argparse
import matplotlib.pyplot as plt

from dev.get_data import get_data

parser = argparse.ArgumentParser()
parser.add_argument('--exppath', default='.')
parser.add_argument('--num_days', default=7)
parser.add_argument('--num_hidden', default=256)
parser.add_argument('--numepochs', default=50)


def lstm_model(num_hidden=50, feat_length=9, num_days=30, stateful_lstm=False):
    model = Sequential()
    model.add(
        LSTM(units=num_hidden, input_shape=(num_days * 48, feat_length), stateful=False, return_sequences=False))
    model.add(Dense(48, activation='relu'))
    # model.add(Dense(48))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model


if __name__ == "__main__":
    args = parser.parse_args()
    exp_path = args.exppath
    os.makedirs(exp_path, exist_ok=True)
    num_hidden = int(args.num_hidden)
    num_days = int(args.num_days)
    train_features, train_labels, val_features, val_labels, minmaxscalar = get_data(num_days)
    joblib.dump(minmaxscalar, os.path.join(exp_path, 'minmaxscalar.file'))
    model = lstm_model(num_hidden=num_hidden, num_days=num_days)
    print(train_features.shape)
    print(train_labels.shape)
    print(val_features.shape)
    print(val_labels.shape)
    print(model.summary())
    history = model.fit(train_features, train_labels, epochs=100, batch_size=72, verbose=2,
                        validation_data=(val_features, val_labels), shuffle=True)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    plt.close()
    model.save(os.path.join(exp_path, 'model_lstm.h5'))
