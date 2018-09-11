from dev.models.neural_network_model import NeuralNetworkModel
from keras import Sequential
from keras.layers import Dense, Flatten, AveragePooling2D, Conv2D


class Conv2DModel(NeuralNetworkModel):

    @staticmethod
    def lstm_model(num_hidden=50, feat_length=26, num_days=30,
                   ):
        model = Sequential()
        model.add(Conv2D(48, 5, activation='relu',
                         input_shape=(48 * num_days, feat_length)))
        # model.add(Conv2D(48, 5, activation='selu'))
        model.add(AveragePooling2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(num_hidden, activation='relu'))
        model.add(Dense(48, activation='selu'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def get_name(self):
        return "conv2d"
