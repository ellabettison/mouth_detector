import csv
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout


def define_recurrent_model(X, y):
    model = Sequential()
    model.add(LSTM(100, input_shape=(5, 256)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X, y, epochs=20, validation_split=0.2, verbose=2, batch_size=64)
    scores = model.evaluate(X, y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    # model.save('../rnn_model.h5')
    model.save_weights("../rnn_model_weights.h5")
    model_json = model.to_json()
    with open("../rnn_model_arch.json", 'w') as json_file:
        json_file.write(model_json)


def split_up_data(data):
    mem_size = 5
    X = []
    y = []
    for i in range(mem_size, len(data)):
        X.append([x[:-1] for x in data[i - mem_size:i]])
        y.append(data[i][-1])

    X = np.array(X).reshape((-1, 5, 256))
    X = tf.convert_to_tensor(X)
    y = np.array(y).reshape((-1, 1))
    y = y.astype(np.float)
    return X, y


def get_data_from_csv():
    y = []
    f = open('../calls_labelled_2/call-2_labelled.csv', 'r')
    with f:
        reader = csv.reader(f)
        for row in reader:
            # print(np.array(row[0]))
            y.append([eval(row[0]), int(row[1])])  # TODO: CHANGE!!!
    f = open('../calls_labelled_2/call-3_labelled.csv', 'r')
    with f:
        reader = csv.reader(f)
        for row in reader:
            y.append([eval(row[0]), int(row[1])])  # TODO: CHANGE!!!
    f = open('../call-10_labelled.csv', 'r')
    with f:
        reader = csv.reader(f)
        for row in reader:
            y.append([eval(row[0]), int(row[1])])  # TODO: CHANGE!!!
    # y = np.array(y)
    # print(y)
    # y = y.astype(np.float)
    return y


csv_data = get_data_from_csv()
split_X, split_y = split_up_data(csv_data)
define_recurrent_model(split_X, split_y)
