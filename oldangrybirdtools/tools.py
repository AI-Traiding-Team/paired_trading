from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import numpy as np
from pandas import DataFrame as df


class AngryFeatureMaker:
    def __init__(self, data):
        self.data_x = data.drop(columns=['Signal'])
        self.data_y = data['Signal'].copy()
        self.xScaler = RobustScaler()
        self.__make_dirty_features()

    def __make_dirty_features(self):
        temp_df = self.data_x.copy(deep=False)
        temp_df['year'] = temp_df.index.year
        temp_df['quarter'] = temp_df.index.quarter
        temp_df['month'] = temp_df.index.month
        temp_df['weeknum'] = temp_df.index.isocalendar().week
        temp_df['weekday'] = temp_df.index.day_of_week
        temp_df['hour'] = temp_df.index.hour
        temp_df['minute'] = temp_df.index.minute
        temp_df['sin'] = temp_df['Close'].apply(lambda x: np.sin(x))
        self.feat_data = self.xScaler.fit_transform(temp_df)
        self.data_y.fillna(1, inplace=True)
        y = self.data_y.copy().to_numpy().reshape(-1, 1)
        self.y_encoder = OneHotEncoder().fit(y)
        self.y_ohot = self.y_encoder.transform(y).toarray()

    def run(self):
        return self.feat_data, self.y_ohot


class BeachBirdSeriesGenerator:
    def __init__(self, dataset, batch_size, sample_x, **kwargs):
        self.source_data = dataset
        self.batch_size = batch_size
        self.sample_x = sample_x
        self.featurizer = AngryFeatureMaker(dataset)
        self.feat_data, self.target_data = self.featurizer.run()
        self.shape = self.feat_data.shape
        self.input_shape = (self.sample_x, self.feat_data.shape[1])

        self.training_start_index = 0 if 'training_start_index' not in kwargs.keys() else kwargs['training_start_index']
        self.val_start_index = 0 if 'val_start_index' not in kwargs.keys() else kwargs['val_start_index']
        self.test_start_index = 0 if 'test_start_index' not in kwargs.keys() else kwargs['test_start_index']
        self.training_end_index = self.shape[0] - 1 if 'training_end_index' not in kwargs.keys() else kwargs[
            'training_end_index']
        self.val_end_index = self.shape[0] - 1 if 'val_end_index' not in kwargs.keys() else kwargs['val_end_index']
        self.test_end_index = self.shape[0] - 1 if 'test_end_index' not in kwargs.keys() else kwargs['test_end_index']

        self.stride = 1
        self.sampling_rate = 1

    def __ts(self, start_index, end_index):
        return TimeseriesGenerator(self.feat_data, self.target_data, length=self.sample_x,
                                   start_index=start_index, end_index=end_index,
                                   batch_size=self.batch_size)

    def train(self, start_index=None, end_index=None):
        start = (self.training_start_index, start_index)[start_index != None]
        end = (self.training_end_index, end_index)[end_index != None]
        return self.__ts(start, end)

    def val(self, start_index=None, end_index=None):
        start = (self.val_start_index, start_index)[start_index != None]
        end = (self.val_end_index, end_index)[end_index != None]
        return self.__ts(start, end)

    def test(self, start_index=None, end_index=None):
        start = (self.test_start_index, start_index)[start_index != None]
        end = (self.test_end_index, end_index)[end_index != None]
        return self.__ts(start, end)

    def test_prep_dec(self, prep):
        assert self.test_end_index + 1 - self.test_start_index - self.sample_x == prep.shape[
            0], 'Length of prediction is not equal length of source data.'
        prep = self.featurizer.y_encoder.inverse_transform(prep)
        res = df()
        res = self.source_data[['Open', 'High', 'Low', 'Close']][
              self.test_start_index + self.sample_x: self.test_end_index + 1].copy()
        res['Signal'] = prep
        return res