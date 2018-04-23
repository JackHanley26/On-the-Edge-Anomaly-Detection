import datetime

from matplotlib import pyplot
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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


def date_parser(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


resources = "../resources/csv/"

file = "us/downtown-crosstown.csv"

dataset = read_csv(resources + file, parse_dates=True, date_parser=date_parser, index_col='timestamp')
# dataset = read_csv(resources + file)

print(dataset.head())

values = dataset.values
num_cols = len(dataset.columns.values)

# specify columns to plot
groups = [i for i in range(0, num_cols)]

# find columns to encode
encoder = LabelEncoder()

for col in dataset.columns:
    print(col + "\t\t\t" + str(dataset[col].dtype))
    if dataset[col].dtype == 'object':
        index = dataset.columns.get_loc(col)
        values[:, index] = encoder.fit_transform(values[:, index].astype(str))

# ensure all data is float
values = values.astype('float64')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)


print("number of columns %s" % len(reframed.columns))
# drop columns we don't want to predict
reframed.drop(reframed.columns[[i for i in range(18, 34)]], axis=1, inplace=True)

print(reframed.head())


# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24




train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)











