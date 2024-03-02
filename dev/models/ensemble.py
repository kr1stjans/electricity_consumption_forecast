from dev.models.abstract_model import AbstractModel
from dev.models.neural_network_model import NeuralNetworkModel
from dev.settings import FORECAST_SIZE
import numpy as np
from keras.models import load_model
from sklearn.neighbors import KernelDensity


class EnsembleModel(AbstractModel):

    def get_prediction(self, x_values, y_values, train_end_index, consumer_name):
        train_end_index_day = int(train_end_index / FORECAST_SIZE)
        file_names = ['model_lstm_series_{}.h5'.format(consumer_name),
                      'model_conv2d_series_{}.h5'.format(consumer_name),
                      'model_conv1d_series_{}.h5'.format(consumer_name)]

        predictions = [] * len(file_names)
        for ind, file_name in enumerate(file_names):
            model = load_model(file_name)
            if ind == 1:
                sample_data = np.expand_dims(x_values[train_end_index_day:train_end_index_day + 1], axis=3)
            else:
                sample_data = x_values[train_end_index_day:train_end_index_day + 1]
            prediction = model.predict(sample_data, batch_size=64)
            predictions[ind] = prediction

        predictions = np.array(predictions)
        kdes = []
        probs = []
        for col in len(range(predictions.shape[1])):
            silverman_bandwidth = ((4 * np.std(predictions[:, col]) ** 5) / (3 * predictions.shape[0])) ** (1 / 5)
            kde = KernelDensity(kernel='gaussian', bandwidth=silverman_bandwidth).fit(predictions[:, col])
            kdes.append(kde)
            prob = kde.score_samples(predictions[:, col])
            prob = np.exp(prob)
            probs.append(prob)
        probs = np.array(probs)
        prediction = []
        for i in range(len(probs.shape[0])):
            prediction.append(np.argmax(probs[i, :]))
        prediction = np.array(prediction)
        prediction = prediction[np.newaxis, :]
        return prediction[0] if len(prediction) > 0 else None

    def get_name(self):
        return "ensemble"
