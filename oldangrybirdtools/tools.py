from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import numpy as np
from pandas import DataFrame as df


class AngryFeatureMaker:
    def __init__(self, data):
        self.data_x = data.drop(columns=['Signal'])
        self_data_y = data['Signal'].copy()
        self.xScaler = RobustScaler().fit([self.data_x])
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
        self.feat_data = temp_df
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
        self.feat_data, self.target_data = AngryFeatureMaker(dataset).run()
        pass
        # self.input_shape = (self.ensemble, self.feat_data.shape[1])

        # self.training_start_index = 0 if 'training_start_index' not in kwargs.keys() else kwargs['training_start_index']
        # self.val_start_index = 0 if 'val_start_index' not in kwargs.keys() else kwargs['val_start_index']
        # self.test_start_index = 0 if 'test_start_index' not in kwargs.keys() else kwargs['test_start_index']
        # self.training_end_index = self.shape[0] - 1 if 'training_end_index' not in kwargs.keys() else kwargs[
        #     'training_end_index']
        # self.val_end_index = self.shape[0] - 1 if 'val_end_index' not in kwargs.keys() else kwargs['val_end_index']
        # self.test_end_index = self.shape[0] - 1 if 'test_end_index' not in kwargs.keys() else kwargs['test_end_index']
        #
        # self.stride = 1
        # self.sampling_rate = 1

    def featurize(self):
        if self.one_hot_enc:
            self.one_hot_encode_y()

        temp = self.data.copy()

        if self.date_to_feat:
            date = pd.to_datetime(temp['Date'], format='%Y%m%d')
            temp['week_day'] = date.apply(lambda x: x.weekday())
            temp['month'] = date.apply(lambda x: x.month)
            temp['week'] = date.apply(lambda x: x.week)
            temp['day'] = date.apply(lambda x: x.day)

        temp.drop(['Ticker', 'Per', 'Date', 'Time'], axis=1, inplace=True)
        if self.drop_signal:
            temp.drop(['Signal'], axis=1, inplace=True)
        temp['sin'] = temp['Close'].apply(lambda x: np.sin(x))

        if self.dropna:
            temp.dropna(axis=0, inplace=True)

        self.data_before_scaling = temp

        xScaler = RobustScaler()
        self.X = xScaler.fit_transform(temp)

        self.input_shape = (self.ensemble, self.X.shape[1])

        self.featurized = True

    def one_hot_encode_y(self):
        self.y = self.data['Signal'].copy().to_numpy().reshape(-1, 1)
        self.y_encoder = OneHotEncoder().fit(self.y)
        self.y = self.y_encoder.transform(self.y).toarray()

    def __ts(self, start_index, end_index):
        assert self.featurized, 'Dataset is not featuarized yet, perform Dataset.featurize()!'
        return TimeseriesGenerator(self.X, self.y, length=self.ensemble, stride=self.stride,
                                   sampling_rate=self.sampling_rate,
                                   start_index=start_index, end_index=end_index,
                                   batch_size=self.batch_size)

    def train(self, start_index=None, end_index=None):
        start = (self.training_start_index, start_index)[start_index != None]
        end = (self.training_end_index, end_index)[end_index != None]
        assert self.featurized, 'Dataset is not featuarized yet, perform Dataset.featurize()!'
        return self.__ts(start, end)

    def val(self, start_index=None, end_index=None):
        start = (self.val_start_index, start_index)[start_index != None]
        end = (self.val_end_index, end_index)[end_index != None]
        assert self.featurized, 'Dataset is not featuarized yet, perform Dataset.featurize()!'
        return self.__ts(start, end)

    def test(self, start_index=None, end_index=None):
        start = (self.test_start_index, start_index)[start_index != None]
        end = (self.test_end_index, end_index)[end_index != None]
        assert self.featurized, 'Dataset is not featuarized yet, perform Dataset.featurize()!'
        return self.__ts(start, end)

    def test_prep_dec(self, prep):
        assert self.featurized, 'Dataset is not featuarized yet, perform Dataset.featurize()!'
        assert self.test_end_index + 1 - self.test_start_index - self.ensemble == prep.shape[
            0], 'Length of prediction is not equal length of source data.'
        if self.one_hot_enc:
            prep = self.y_encoder.inverse_transform(prep)
        else:
            assert prep.shape[1] == 1, 'Wrong shape dimension of predicted data.'
        res = df()
        res = self.data[['Open', 'High', 'Low', 'Close']][
              self.test_start_index + self.ensemble: self.test_end_index + 1].copy()
        res['Signal'] = prep
        return res