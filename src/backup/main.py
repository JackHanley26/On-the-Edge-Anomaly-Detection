import datetime

from keras import Sequential
from keras.layers import LSTM, Dense
from matplotlib import pyplot
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import os

folder = '../resources/us'
file = 'downtown-west.csv'


def dateparse(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def start():
    file_path = os.path.abspath(folder + "/" + file)
    dataset = read_csv(file_path, index_col=0, header=0)
    # dataset = read_csv(file_path, parse_dates=True, date_parser=dateparse, index_col=0, header=0)

    values = dataset.values
    # specify columns to plot
    groups = [i for i in dataset.columns.values]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()
    dataset.drop(dataset.columns[[i for i in range(2, 12)]], axis=1, inplace=True)

    values = dataset.values

    # integer encode direction
    # encoder = LabelEncoder()
    # # brake_status is True/False
    # values[:, 10] = encoder.fit_transform(values[:, 10])
    # # gear position is in string format
    # values[:, 11] = encoder.fit_transform(values[:, 11])
    # # ensure all data is float
    # values = values.astype('float32')

    print(DataFrame(values).head())

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict

    drop_cols = [i for i in range(15, 24)]
    drop_cols.append(12)
    drop_cols.append(13)
    reframed.drop(reframed.columns[drop_cols], axis=1, inplace=True)

    print(reframed.head())

    # split into train and test sets
    values = reframed.values
    n_train = round(len(values) * 0.8)

    train = values[n_train:, :]
    test = values[:n_train, :]
    print("Train Size: %s\nTest Size: %s" % (len(test), len(train)))
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


if __name__ == '__main__':
    start()
