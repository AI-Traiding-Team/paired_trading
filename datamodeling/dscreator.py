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
from datafeatures import DataFeatures


__version__ = 0.0004


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

        if self.start_index > self.end_index:
            raise ValueError('`start_index+length=%i > end_index=%i` '
                             'is disallowed, as no part of the sequence '
                             'would be left to be used as current step.'
                             % (self.start_index, self.end_index))

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
        targets = np.array([self.targets[row] for row in rows])
        # print(samples.shape)
        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, targets


@dataclass(init=True)
class DSProfile:
    features_list = []
    use_symbols_pairs = ("BTCUSDT", "ETHUSDT")


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
        self.x_Train = None
        self.y_Train = None
        self.x_Val = None
        self.y_Val = None
        self.x_Test = None
        self.y_Test = None
        pass

    def create_dataset(self):
        self.features()
        ready_dataset = (self.x_Train, self.y_Train,
                         self.x_Val, self.y_Val,
                         self.x_Test, self.y_Test)

        return ready_dataset

    def save_dataset(self, path_filename):
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
                                  start_period='2021-09-01 00:00:00',
                                  end_period='2021-12-05 23:59:59',

                                  )
    dataset_1_profile = DSProfile()
    dataset_1 = DSCreator(loaded_crypto_data,
                          dataset_1_profile)

    x_Train, y_Train, x_Val, y_Val, x_Test, y_Test = dataset_1.create_dataset()
