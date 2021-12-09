import os
import sys
import time
import copy
import pytz
import numpy as np
import datetime
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from dataclasses import dataclass
from analyze.dataload import DataLoad
from datamodeling.datafeatures import DataFeatures, DSProfile


__version__ = 0.0008



def get_local_timezone_name():
    if time.daylight:
        offset_hour = time.altzone / 3600
    else:
        offset_hour = time.timezone / 3600

    offset_hour_msg = f"{offset_hour:.0f}"
    if offset_hour > 0:
        offset_hour_msg = f"+{offset_hour:.0f}"
    return f'Etc/GMT{offset_hour_msg}'


class TSDataGenerator(TimeseriesGenerator):
    def __init__(self, data, targets, length, sampling_rate=1, stride=1, start_index=0, overlap=0, end_index=None,
                 shuffle=False, reverse=False, batch_size=128):
        super().__init__(data, targets, length, sampling_rate, stride, start_index, end_index, shuffle, reverse,
                         batch_size)

        if len(data) != len(targets):
            raise ValueError('Data and targets have to be' +
                             ' of same length. '
                             'Data length is {}'.format(len(data)) +
                             ' while target length is {}'.format(len(targets)))
        if overlap >= length:
            raise ValueError(f'`overlap={overlap} >= length={length}` is disallowed')
        if overlap > 0:
            start_index += overlap

        self.data = data
        self.targets = targets
        self.length = length
        self.sampling_rate = sampling_rate
        self.stride = stride
        self.start_index = start_index + length
        self.overlap = overlap
        if end_index is None:
            end_index = len(data) - 1
        self.end_index = end_index
        self.shuffle = shuffle
        self.reverse = reverse
        self.batch_size = batch_size
        self.sample_shape = None

        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

        self.sample_shape = self.calc_shape()
        pass

    def calc_shape(self):
        index = 1
        i = (self.start_index + self.batch_size * self.stride * index)
        rows = np.arange(i, min(i + self.batch_size *
                                self.stride, self.end_index + 1), self.stride)
        samples = np.array([self.data[row - self.overlap - self.length:row:self.sampling_rate]
                            for row in rows])
        # self.sample_shape = np.expand_dims(samples, axis=0).shape
        sample_shape = (samples.shape[-2], samples.shape[-1],)
        return sample_shape

    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = (self.start_index + self.batch_size * self.stride * index)
            rows = np.arange(i, min(i + self.batch_size *
                                    self.stride, self.end_index + 1), self.stride)

        samples = np.array([self.data[row - self.overlap - self.length:row:self.sampling_rate]
                            for row in rows])
        self.sample_shape = samples.shape
        targets = np.array([self.targets[row] for row in rows])

        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets


@dataclass
class DataSet:
    def __init__(self):
        self.name: str = ''
        self.dataset_profile = DSProfile()
        self.features_df = None
        self.y_df = None
        self.x_Train = None
        self.y_Train = None
        self.x_Val = None
        self.y_Val = None
        self.x_Test = None
        self.y_Test = None
        self.features_scaler = None
        self.targets_scaler = None
        self.train_gen = None
        self.val_gen = None
        self.test_gen = None
        self.input_shape = None
    pass

    def get_train(self):
        if (self.x_Train is not None) and (self.y_Train is not None):
            return self.x_Train, self.y_Train

    def get_val(self):
        if (self.x_Val is not None) and (self.y_Val is not None):
            return self.x_Val, self.y_Val

    def get_test(self):
        if (self.x_Test is not None) and (self.y_Test is not None):
            return self.x_Test, self.y_Test


class DSCreator:
    """
    Class for dataset creation
    for dataset configuration we are using DSConstants dataclass (profile)
    """

    def __init__(self,
                 loader: DataLoad,
                 dataset_profile: DSProfile):
        """
        Getting object with OHLCV data (symbols and timeframes).
        All data with chosen period loaded to memory

        Args:
            loader (DataLoad):  object with data

        Returns:
            DSCreator (class):  object
        """
        self.features = DataFeatures(loader)
        self.dataset_profile = dataset_profile
        self.dataset = DataSet()

    def split_data_df(self):
        df_rows = self.dataset.features_df.shape[0]
        df_train_len = int(df_rows * self.dataset_profile.train_size)
        df_val_len = df_rows - (df_train_len + self.dataset_profile.gap_timeframes)
        self.dataset.train_df = self.dataset.features_df.iloc[:df_train_len, :]
        if self.dataset_profile.train_size + self.dataset_profile.val_size == 1.0:
            self.dataset.val_df = self.dataset.features_df.iloc[df_train_len + self.dataset_profile.gap_timeframes:, :]
            return df_train_len, df_val_len, None
        else:
            df_val_len = int(df_rows * self.dataset_profile.val_size)
            df_test_len = df_rows - (df_train_len + self.dataset_profile.gap_timeframes) - (df_val_len + self.dataset_profile.gap_timeframes)
            self.dataset.val_df = self.dataset.features_df.iloc[
                                  df_train_len + self.dataset_profile.gap_timeframes: df_val_len + df_train_len + self.dataset_profile.gap_timeframes,
                                  :]
            self.dataset.test_df = self.dataset.features_df.iloc[df_rows - df_test_len:, :]
            return df_train_len, df_val_len, df_test_len

    def get_train_generator(self, x_Train_data, y_Train_data):
        self.dataset.train_gen = TSDataGenerator(data=x_Train_data,
                                                 targets=y_Train_data,
                                                 length=self.dataset_profile.tsg_window_length,
                                                 sampling_rate=self.dataset_profile.tsg_sampling_rate,
                                                 stride=self.dataset_profile.tsg_stride,
                                                 start_index=self.dataset_profile.tsg_start_index,
                                                 overlap=self.dataset_profile.tsg_overlap,
                                                 )
        return self.dataset.train_gen

    def get_val_generator(self, x_Val_data, y_Val_data):
        self.dataset.val_gen = TSDataGenerator(data=x_Val_data,
                                               targets=y_Val_data,
                                               length=self.dataset_profile.tsg_window_length,
                                               sampling_rate=self.dataset_profile.tsg_sampling_rate,
                                               stride=self.dataset_profile.tsg_stride,
                                               start_index=self.dataset_profile.tsg_start_index,
                                               overlap=self.dataset_profile.tsg_overlap,
                                               )
        return self.dataset.val_gen

    def get_test_generator(self, x_Test_data, y_Test_data):
        self.dataset.test_gen = TSDataGenerator(data=x_Test_data,
                                                targets=y_Test_data,
                                                length=self.dataset_profile.tsg_window_length,
                                                sampling_rate=self.dataset_profile.tsg_sampling_rate,
                                                stride=self.dataset_profile.tsg_stride,
                                                start_index=self.dataset_profile.tsg_start_index,
                                                overlap=self.dataset_profile.tsg_overlap,
                                                )
        return self.dataset.val_gen

    def create_dataset(self) -> DataSet:
        self.dataset.dataset_profile = DSProfile()
        self.dataset.features_df = self.features.collect_features(self.dataset_profile)
        self.dataset.y_df = self.features.create_y_close1_close2_sub()
        self.dataset.name = f'{self.dataset_profile.use_symbols_pairs[0]}-{self.dataset_profile.use_symbols_pairs[1]}-{self.dataset_profile.timeframe}'
        y_temp = self.dataset.y_df.values.reshape(-1, 1)
        if self.dataset_profile.scaler == "robust":
            self.dataset.features_scaler = RobustScaler().fit(self.dataset.features_df.values)
            self.dataset.targets_scaler = RobustScaler().fit(y_temp)

        x_arr = self.dataset.features_scaler.transform(self.dataset.features_df.values)
        """ check """
        y_arr = self.dataset.targets_scaler.transform(y_temp)
        train_len, val_len, test_len = self.split_data_df()
        if test_len is None:
            x_Train_data = x_arr[train_len:, :]
            x_Val_data = x_arr[:train_len + self.dataset_profile.gap_timeframes, :]
            y_Train_data = y_arr[train_len:, :]
            y_Val_data = y_arr[:train_len + self.dataset_profile.gap_timeframes, :]
        else:
            x_Train_data = x_arr[train_len:, :]
            x_Val_data = x_arr[train_len + self.dataset_profile.gap_timeframes:train_len + self.dataset_profile.gap_timeframes + val_len, :]
            x_Test_data = x_arr[x_arr.shape[0] - test_len:, :]
            y_Train_data = y_arr[train_len:, :]
            y_Val_data = y_arr[train_len + self.dataset_profile.gap_timeframes:train_len + self.dataset_profile.gap_timeframes + val_len, :]
            y_Test_data = y_arr[x_arr.shape[0] - test_len:, :]
            _ = self.get_test_generator(x_Test_data, y_Test_data)

        _ = self.get_train_generator(x_Train_data, y_Train_data)
        x_Val_gen = self.get_val_generator(x_Val_data, y_Val_data)
        self.dataset.input_shape = x_Val_gen.sample_shape
        return self.dataset

    def save_dataset_arrays(self, path_filename):
        pass


if __name__ == "__main__":
    """
    Usage for DataLoad class
    ------------------------
    pairs_symbol = None ->                    Use all pairs in timeframe directory
    pairs_symbol = ("BTCUSDT", "ETHUSDT") ->  Use only this pairs to load 
    
    time_intervals = None ->                Use all timeframes directories for loading (with pairs_symbols)
    time_intervals = ['15m'] ->             Use timeframes from this list to load
    
    start_period = None ->                  Use from [0:] of historical data
    start_period = '2021-09-01 00:00:00' -> Use from this datetimeindex
    
    end_period = None ->                    Use until [:-1] of historical data
    end_period = '2021-12-05 23:59:59' ->   Use until this datetimeindex
    
    source_directory="../source_root" ->    Use this directory to search timeframes directory
    """

    loaded_crypto_data = DataLoad(pairs_symbols=None,
                                  time_intervals=['15m'],
                                  source_directory="../source_root",
                                  start_period='2021-11-01 00:00:00',
                                  end_period='2021-12-05 23:59:59',
                                  )

    dataset_1_profile = DSProfile()
    dsc = DSCreator(loaded_crypto_data,
                    dataset_1_profile)

    dataset_1_cls = dsc.create_dataset()
