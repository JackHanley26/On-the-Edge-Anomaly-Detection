import json
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from keras.models import model_from_json
from pandas import DataFrame, concat, Series
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from math import sqrt


def read():
    with open('../resources/us/downtown-west.json') as f:
        content = f.readlines()
        return [json.loads(x.strip()) for x in content]


def get_data(event_name):
    trace = read()

    types = dict()
    count = 0
    for log in trace:
        count += 1
        if count > 300000:
            break
        key = log.get('name')
        tmp = list()
        tmp.append(log.get('timestamp'))
        tmp.append(log.get('value'))
        if types.get(key):
            types[key].append(tmp)
        else:
            types[key] = [tmp]

    return types.get(event_name)


def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def test():
    event_name = "vehicle_speed"

    event_data = get_data(event_name)

    event_target = event_data[1:]

    event_data = event_data[:-1]

    # df = DataFrame(data=event_data, columns=['timestamp', event_data])

    data = np.array([[i[1] for i in event_data]], dtype=float).reshape(1, 1, len(event_data))
    target = np.array([[i[1] for i in event_target]], dtype=float).reshape(1, 1, len(event_target))

    model = generate_model(data, target=target)


def load_model():
    # load json and create model
    json_file = open('../models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


def generate_model(data, target):
    model = Sequential()

    model.add(LSTM(100, input_shape=(1, 100), return_sequences=True))
    model.add(Dense(100))

    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=2)

    model.fit(data, target, epochs=10000, batch_size=1, verbose=2, callbacks=[early_stopping])
    print("Generated ML Model")
    return model


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("../models/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    print("Running Task...")

    test()
    exit(0)

    data = np.array([[i for i in range(100)]], dtype=float).reshape((1, 1, 100))
    target = np.array([[i for i in range(1, 101)]], dtype=float).reshape((1, 1, 100))

    test = np.array([[i for i in range(20, 30)]], dtype=float).reshape(1, 1, 10)

    model = generate_model(data=data, target=target)

    save_model(model)
    model = load_model()

    predict = model.predict(x=test)

    print(predict)
