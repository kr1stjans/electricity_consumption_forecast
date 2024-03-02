from dev.models.neural_network_model import NeuralNetworkModel
from keras import Sequential
from keras.layers import Dense, Flatten, AveragePooling2D, Conv2D


class Conv2DModel(NeuralNetworkModel):

    def build_model(self, num_hidden=50, feat_length=26, num_days=30, ):
        self.model = Sequential()
        self.model.add(Conv2D(48, 5, activation='relu',
                              input_shape=(48 * num_days, feat_length)))
        self.model.add(AveragePooling2D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(num_hidden, activation='relu'))
        self.add(Dense(48, activation='selu'))
        self.compile(loss='mean_squared_error', optimizer='adam')

    def get_name(self):
        return "conv2d"
