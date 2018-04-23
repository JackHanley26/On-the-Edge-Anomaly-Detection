import os, re
from keras.layers import LSTM
from keras.models import model_from_json, Sequential

model_dir = "../models/"


def generate_lstm_model(layers, neurons, size, dim, return_sequences=False):
    model = Sequential()
    for i in range(layers):
        model.add(LSTM(neurons, input_shape=(size, dim), return_sequences=return_sequences))
    print("Generated LSTM model with %s hidden layers and %s neurons in each layer" % (layers, neurons))
    return model


def get_most_recent_model_file_name(name):
    models = [f for f in os.listdir(model_dir) if
              os.path.isfile(os.path.join(model_dir, f)) and name.split('_')[1] in f]
    most_recent = max(models)
    split1 = most_recent.split('_')
    split2 = split1[1].split('.')[0]
    return split1[0] + "_" + split2


def load_model(name):
    # load json and create model
    json_file = open(model_dir + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_dir + name + '.h5')
    print("Loaded model from disk")
    return loaded_model


def save_model(model, name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_dir + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_dir + name + ".h5")
    print("Saved model to disk")
