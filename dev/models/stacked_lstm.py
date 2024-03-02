from keras import Sequential
from keras.layers import CuDNNLSTM, LSTM, Dense

from dev.models.neural_network_model import NeuralNetworkModel
from dev.settings import FORECAST_SIZE


class StackedLSTMModel(NeuralNetworkModel):

    def build_model(self, num_hidden, feat_length, num_days):
        """
        Defines the LSTM Model for regression. The objective is to minimize MSE which is the same as minimizing RMSE.
        The optimizer used is RMSProp. Please do not change this optimizer.
        """
        model = Sequential()
        model.add(
            LSTM(units=num_hidden,
                 input_shape=(num_days * FORECAST_SIZE, feat_length),
                 return_sequences=True))
        model.add(LSTM(units=num_hidden,
                       return_sequences=False))
        model.add(Dense(48, activation='selu'))
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model

    def get_name(self):
        return "lstm"
