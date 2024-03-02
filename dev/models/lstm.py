from keras import Sequential
from keras.layers import CuDNNLSTM, LSTM, Dense
from keras.utils import multi_gpu_model

from dev.models.neural_network_model import NeuralNetworkModel
from dev.settings import FORECAST_SIZE


class LSTMModel(NeuralNetworkModel):

    def build_model(self, num_hidden, feat_length, num_days):
        """
        Defines the LSTM Model for regression. The objective is to minimize MSE which is the same as minimizing RMSE.
        The optimizer used is RMSProp. Please do not change this optimizer.
        """
        self.model = Sequential()

        lstm = LSTM(units=num_hidden, input_shape=(num_days * FORECAST_SIZE, feat_length),
                    return_sequences=False) if not self.use_gpu else CuDNNLSTM(units=num_hidden,
                                                                               input_shape=(num_days * FORECAST_SIZE,
                                                                                            feat_length),
                                                                               return_sequences=False)
        self.model.add(lstm)

        self.model.add(Dense(48, activation='relu'))

        self.build_model(num_hidden, feat_length, num_days)

    def get_name(self):
        return "lstm"
