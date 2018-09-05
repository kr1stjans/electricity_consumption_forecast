import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense


#################
# Vanilla LSTMs #
#################


# one hot encode sequence
def one_hot_encode(sequence, n_features):
    encoding = np.zeros((len(sequence), n_features))
    for i, value in enumerate(sequence):
        encoding[i, value] = 1
    return encoding


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return np.argmax(encoded_seq, axis=1)


# generate one example for an lstm
def generate_example(length, n_features, out_index):
    sequence = np.random.randint(n_features, size=length)
    encoded = one_hot_encode(sequence, n_features)
    X = encoded.reshape((1, length, n_features))
    y = encoded[out_index].reshape(1, n_features)
    return X, y


# define model
length = 5
n_features = 10
out_index = 2
model = Sequential()
model.add(LSTM(25, input_shape=(length, n_features)))
model.add(Dense(n_features, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(model.summary())


# fit model
for i in range(10000):
    X, y = generate_example(length, n_features, out_index)
    model.fit(X, y, epochs=1, verbose=2)

# evaluate model
correct = 0
for _ in range(100):
    X, y = generate_example(length, n_features, out_index)
    yhat = model.predict(X)
    if one_hot_decode(yhat) == one_hot_decode(y):
        correct += 1
print('Accuracy: %.2f' % ((correct/100)*100.0))

# prediction on new data
X, y = generate_example(length, n_features, out_index)
yhat = model.predict(X)
print('Sequence: %s' % [one_hot_decode(x) for x in X])
print('Expected: %s' % one_hot_decode(y))
print('Predicted: %s' % one_hot_decode(yhat))