import datetime

from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

"""
#  backup
colours = ["#C0C0C0", "#808080", "#000000", "#FF0000", "#800000", "#FFFF00", "#808000", "#00FF00", "#008000", "#00FFFF",
           "#008080", "#0000FF", "#000080", "#FF00FF", "#800080", "#CD5C5C", "#FA8072", "#E9967A", ]
"""


def date_parser(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


def read_csv_data(path, parse_date=True):
    if parse_date:
        df = read_csv(path, parse_dates=True, date_parser=date_parser, index_col='timestamp')
    else:
        df = read_csv(path)
    print("DataFrame size: %s" % str(len(df.values)))
    return df


def drop_columns(normal_df, abnormal_df):
    normal_columns = normal_df.columns.values
    abnormal_columns = abnormal_df.columns.values

    normal_columns_to_drop = [i for i, v in enumerate(normal_columns) if v not in abnormal_columns]
    abnormal_columns_to_drop = [i for i, v in enumerate(abnormal_columns) if v not in normal_columns]

    normal_df.drop(normal_df.columns[normal_columns_to_drop], axis=1, inplace=True)

    abnormal_df.drop(abnormal_df.columns[abnormal_columns_to_drop], axis=1, inplace=True)

    return normal_df, abnormal_df


def get_feature_count(normal, abnormal):
    if len(normal.columns.values) == len(abnormal.columns.values):
        return len(normal.columns.values)
    raise Exception("DataFrames must have the same number of features")


def get_sample(data, index, size):
    sample = data[index: index + size]
    print("Sample size: %s, Sample Shape: %s" % (len(sample), sample.shape))
    return sample


def encode_values(df):
    encoder = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == 'object':
            index = df.columns.get_loc(col)
            df.values[:, index] = encoder.fit_transform(df.values[:, index].astype(str))
    return df.values


def scale_data(data):
    return MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
