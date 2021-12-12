from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
import numpy as np
from pandas import DataFrame as df


class AngryFeatureMaker:
    def __init__(self, data):
        self.data = data
        self.make_dirty_features()

    def __make_dirty_features(self, data):
        self.feat_data = self.data

    def run(self):
        return self.feat_data


class BeachBirdGenerator:
    def __init__(self, dataset, batch_size, ensemble, one_hot_enc=True, dropna=False, drop_signal=False,
                 date_to_feat=False, **kwargs):
        self.featurized = False
        self.data = dataset
        self.batch_size = batch_size
        self.ensemble = ensemble
        self.dropna = dropna
        self.one_hot_enc = one_hot_enc
        self.shape = self.data.shape
        self.drop_signal = drop_signal
        self.date_to_feat = date_to_feat

        self.training_start_index = 0 if 'training_start_index' not in kwargs.keys() else kwargs['training_start_index']
        self.val_start_index = 0 if 'val_start_index' not in kwargs.keys() else kwargs['val_start_index']
        self.test_start_index = 0 if 'test_start_index' not in kwargs.keys() else kwargs['test_start_index']
        self.training_end_index = self.shape[0] - 1 if 'training_end_index' not in kwargs.keys() else kwargs[
            'training_end_index']
        self.val_end_index = self.shape[0] - 1 if 'val_end_index' not in kwargs.keys() else kwargs['val_end_index']
        self.test_end_index = self.shape[0] - 1 if 'test_end_index' not in kwargs.keys() else kwargs['test_end_index']

        self.stride = 1 if 'strid' not in kwargs.keys() else kwargs['stride']
        self.sampling_rate = 1 if 'sampling_rate' not in kwargs.keys() else kwargs['sampling_rate']

        self.featurize()

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