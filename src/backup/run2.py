import datetime
from collections import defaultdict

from keras import Sequential
from keras.layers import LSTM, Dense
from pandas import read_csv, DataFrame, concat
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Small
trace_path0 = '../resources/us/downtown-west.csv'
# Bigger
trace_path1 = '../resources/us/downtown-crosstown.csv'
# Different
trace_path2 = '../resources/PRSA_data_2010.1.1-2014.12.31.csv'

def dateparse(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


def graph_data(dataset):
    # specify columns to plot
    headers = list(dataset.columns.values)
    groups = [v for v in range(len(headers))]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
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


# load dataset
dataset = read_csv(trace_path0, header=0, index_col=0, parse_dates=True, date_parser=dateparse)

values = dataset.values
# Encode all the string to be integers
encoder = LabelEncoder()
values[:, 10] = encoder.fit_transform(values[:, 10])
values[:, 11] = encoder.fit_transform(values[:, 11])


graph_data(dataset=dataset)

# flatten values
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

values = reframed.values
size = len(values)
train_size = round(size * 0.7)
test_size = round(size * 0.3)

train = values[:train_size, :]
test = values[test_size:, :]

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

exit(0)
