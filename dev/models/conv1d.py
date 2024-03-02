from dev.models.neural_network_model import NeuralNetworkModel
from keras import Sequential
from keras.layers import Dense, Flatten, AveragePooling1D, Conv1D


class Conv1DModel(NeuralNetworkModel):

    def build_model(self, num_hidden, feat_length, num_days):
        self.model = Sequential()
        self.model.add(Conv1D(48, 5, activation='relu',
                              input_shape=(48 * num_days, feat_length)))
        self.model.add(AveragePooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(num_hidden, activation='relu'))
        self.model.add(Dense(48, activation='selu'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def get_name(self):
        return "conv1d"
