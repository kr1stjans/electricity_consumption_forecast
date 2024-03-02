from keras import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D
from keras.utils import multi_gpu_model

from dev.models.neural_network_model import NeuralNetworkModel


class UFCNNModel(NeuralNetworkModel):

    def build_model(self, num_hidden, feat_length, num_days):
        model = Sequential()
        model.add(Conv1D(nb_filter=36, input_shape=(48 * num_days, feat_length), filter_length=5,
                         border_mode='valid', init="lecun_uniform"))
        model.add(Activation('relu'))
        model.add(Conv1D(nb_filter=48, filter_length=5, border_mode='same', init="lecun_uniform"))
        model.add(Activation('sigmoid'))
        model.add(Flatten())
        model.add(Dense(128, activation='selu'))
        model.add(Dense(48, activation='selu'))

        if self.use_gpu:
            self.gpu_model = multi_gpu_model(self.model, gpus=3)
            self.gpu_model.compile(loss='mean_squared_error', optimizer='adam')
        else:
            self.model.compile(loss='mean_squared_error', optimizer='adam')

    def get_name(self):
        return "ufcnn"
