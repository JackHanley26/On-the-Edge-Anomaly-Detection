import datetime
from functools import reduce
from pandas import DataFrame
from keras.callbacks import Callback, EarlyStopping
from keras.layers import Dense
from matplotlib import pyplot as plt, pyplot

# managers
from src.utils import data_manager as dm
from src.utils import model_manager as mm
from src.utils import graph_manager as gm

# global variables
start_time = datetime.datetime.now()
#
normal_path = '../resources/csv/us/downtown-crosstown.csv'
abnormal_path = '../resources/csv/other/aggressive-driving.csv'
#
data_size = 9000
data_index = 10000
batch_size = 100
feature_size = 0
#
normal_shape = None
batch_shape = None
#
hidden_layers = 2
neurons = 10
#
save_figs = False
#
losses = []


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        handle_loss(logs.get('loss'))


def handle_loss(loss):
    global losses
    losses += [loss]
    # print(loss)


def generate_name():
    name = str(start_time) + "_"
    name = name + reduce(lambda x, y: str(x) + "-" + str(y),
                         [data_index, data_size, batch_size, hidden_layers, neurons])
    return name


def load_and_normalise_data(n_path, ab_path, size, index, batch):
    # data
    normal = dm.read_csv_data(n_path)
    abnormal = dm.read_csv_data(ab_path)

    # only use the intersection of columns
    dm.drop_columns(normal, abnormal)

    # values
    normal_values = normal.values
    abnormal_values = abnormal.values

    # no. of dimensions / features
    dim = dm.get_feature_count(normal, abnormal)
    global feature_size
    feature_size = dim

    # get samples from whole data set
    normal_values_sample = dm.get_sample(data=normal_values, index=index, size=size)
    abnormal_values_sample = dm.get_sample(data=abnormal_values, index=index, size=size)

    # reshape samples
    normal_values_sample = normal_values_sample.reshape(size, dim)
    abnormal_values_sample = abnormal_values_sample.reshape(size, dim)
    global normal_shape
    normal_shape = normal_values_sample.shape

    # encode sting properties
    normal_encoded_values_sample = dm.encode_values(DataFrame(normal_values_sample))
    abnormal_encoded_values_sample = dm.encode_values(DataFrame(abnormal_values_sample))

    gm.graph_df(normal, normal_values_sample, save_figs)
    gm.graph_df(abnormal, abnormal_values_sample, save_figs)

    # Normalize the data
    normal_normalised_encoded_values_sample = dm.scale_data(normal_encoded_values_sample)
    abnormal_normalised_encoded_values_sample = dm.scale_data(abnormal_encoded_values_sample)

    global batch_shape
    batch_shape = (int(size / batch), batch, dim)

    return normal_normalised_encoded_values_sample, abnormal_normalised_encoded_values_sample


def generate_model():
    model = mm.generate_lstm_model(layers=hidden_layers, neurons=neurons, size=batch_size, dim=feature_size,
                                   return_sequences=True)
    model.add(Dense(feature_size))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    return model


def run_auto_encoder(iterations, read_model):
    normal, abnormal = load_and_normalise_data(n_path=normal_path,
                                               ab_path=abnormal_path,
                                               size=data_size,
                                               index=data_index,
                                               batch=batch_size)

    def train(data):
        early_stopping_monitor = EarlyStopping(patience=5, monitor='val_loss', min_delta=0)
        data.shape = batch_shape
        model.fit(data, data, epochs=100, batch_size=72, validation_data=(data, data), verbose=2, shuffle=False,
                  callbacks=[LossHistory(), early_stopping_monitor])
        data.shape = normal_shape

    def score(data):
        data.shape = batch_shape
        yhat = model.predict(data)
        yhat.shape = normal_shape
        return yhat

    if not read_model:
        #  Generate LSTM
        model = generate_model()

        loop_start_time = datetime.datetime.now()
        for i in range(iterations):
            current = datetime.datetime.now()
            print("\n- - - - - - - - - - - - - - - -")
            print("Iteration: %s" % str(i))
            print("Time elapsed: %s" % str(loop_start_time - current))
            print("- - - - - - - - - - - - - - - -\n")
            train(normal)
            yhat = score(normal)
            normal.shape = normal_shape

        mm.save_model(model=model, name=generate_name())
    else:
        model_name = mm.get_most_recent_model_file_name(generate_name())
        if model_name:
            model = mm.load_model(model_name)
            model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
        else:
            print("Model not found")
            exit(1)

    # print("Running on abnormal data")
    #     #
    #     # train(abnormal)
    #     # yhat = score(abnormal)
    #     # abnormal.shape = normal_shape
    #     #
    #     # # train(normal)
    #     # # yhat = score(normal)
    #     # # normal.shape = normal_shape
    #     #
    #     # fig, ax = plt.subplots(num=None, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
    #     #
    #     # ax.plot(range(0, len(losses)), losses, '-', color='blue', linewidth=1)
    #     #
    #     # figure_name = '../figures/' + generate_name() + '.png'
    #     # print("Saving figure to %s\n" % figure_name)
    #     # pyplot.savefig(figure_name)
    #     #
    #     # print("Total time elapsed: %s" % str(datetime.datetime.now() - start_time))
    #     #
    #     # pyplot.show()

def prediction():
    normal, abnormal = load_and_normalise_data(n_path=normal_path,
                                               ab_path=abnormal_path,
                                               size=data_size,
                                               index=data_index,
                                               batch=batch_size)





if __name__ == '__main__':
    print("Start Time: %s " % start_time)
    run_auto_encoder(iterations=10, read_model=True)
    # run_auto_encoder(iterations=10, read_model=False)
