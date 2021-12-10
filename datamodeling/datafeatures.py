import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple
from analyze.dataload import DataLoad, TradeConstants
from dataclasses import dataclass
# sys.path.insert(1, os.path.join(os.getcwd(), 'analyze'))

__version__ = 0.0008


@dataclass(init=True)
class DSProfile:
    features_list: Tuple = ()
    use_symbols_pairs: Tuple[str, str] = ("BTCUSDT", "ETHUSDT")
    timeframe: str = '15m'
    Y_data: str = "close1-close2"
    scaler: str = "robust"
    """  
    If train_size + val_size < 1.0 
    test_size = 1.0 - train_size + val_size 
    """
    train_size = 0.6
    val_size = 0.2
    """ Warning! Change this qty if using .shift() more then 2 """
    gap_timeframes = 3
    tsg_window_length = 40
    tsg_sampling_rate = 1
    tsg_stride = 1
    tsg_start_index = 0
    tsg_overlap = 0
    pass


class DataFeatures:
    def __init__(self, loader: DataLoad):
        self.drop_idxs: list = []
        self.loader = loader
        self.ohlcv_base = self.loader.ohlcvbase
        self.pairs_symbols = self.loader.pairs_symbols
        self.time_intervals = self.loader.time_intervals
        """ Base is not realized yet """
        # self.features_base: dict = {}
        self.cols_create = ('year',
                            'quarter',
                            'month',
                            'weeknum',
                            'weekday',
                            'hour',
                            'minute'
                            )
        self.clear_na_flag = False
        self.source_df_1 = None
        self.source_df_2 = None
        self.features_df = None
        self.y_df = pd.DataFrame()

    @staticmethod
    def split_datetime_data(datetime_index: pd.DatetimeIndex,
                            cols_defaults: list
                            ) -> pd.DataFrame:
        """
        Args:

            datetime_index (pd.DatetimeIndex):      datetimeindex object
            cols_defaults (list):                   list of names for dividing datetime for encoding

        Returns:
            object:     pd.Dataframe with columns for year, quarter, month, weeknum, weekday, hour(*), minute(*)
                        * only if datetimeindex have data about
        """
        temp_df: pd.DataFrame = pd.DataFrame()
        temp_df.index, temp_df.index.name = datetime_index, "datetimeindex"
        datetime_funct: dict = {'year': temp_df.index.year,
                                'quarter': temp_df.index.quarter,
                                'month': temp_df.index.month,
                                'weeknum': temp_df.index.isocalendar().week,
                                'weekday': temp_df.index.day_of_week,
                                'hour': temp_df.index.hour,
                                'minute': temp_df.index.minute,
                                }
        for col_name in cols_defaults:
            if col_name == 'weeknum':
                temp_df[col_name] = temp_df.index.isocalendar().week
            else:
                if datetime_funct[col_name].nunique() != 1:
                    temp_df[col_name] = datetime_funct[col_name]
        return temp_df

    @staticmethod
    def get_feature_datetime_ohe(datetime_index: pd.DatetimeIndex
                                 ) -> pd.DataFrame:
        """
        Args:
            datetime_index (pd.DatetimeIndex):      datetimeindex object

        Returns:
            object:     pd.Dataframe with dummy encoded datetimeindex columns with prefix 'de_'
        """
        de_df = DataFeatures.split_datetime_data(datetime_index.index)
        cols_names = de_df.columns
        de_df = pd.get_dummies(de_df,
                               columns=cols_names,
                               drop_first=False)
        for col in de_df.columns:
            de_df.rename(columns={col: f'de_{col}'}, inplace=True)
        return de_df

    @staticmethod
    def get_feature_datetime(input_df,
                             cols_create=('year',
                                          'quarter',
                                          'month',
                                          'weeknum',
                                          'weekday',
                                          'hour',
                                          'minute'
                                          )
                             ) -> pd.DataFrame:
        """

        Args:
            input_df (pd.Dataframe):    dataframe
            cols_create:        columns to create from datetime

        Returns:

        """
        date_df = DataFeatures.split_datetime_data(input_df.index, cols_create)
        return date_df

    @staticmethod
    def get_feature_normalized_diff(source_df_1,
                                    source_df_2,
                                    col_use="close"
                                    ) -> pd.DataFrame:

        normalized_df_1 = (source_df_1[col_use] - source_df_1[col_use].min()) / (
                source_df_1[col_use].max() - source_df_1[col_use].min())
        normalized_df_2 = (source_df_2[col_use] - source_df_2[col_use].min()) / (
                source_df_2[col_use].max() - source_df_2[col_use].min())
        diff_df = normalized_df_1 - normalized_df_2
        return diff_df

    def collect_features(self, profile: DSProfile):
        pair_symbol_1 = profile.use_symbols_pairs[0]
        pair_symbol_2 = profile.use_symbols_pairs[1]
        timeframe = profile.timeframe
        features_df = pd.DataFrame()
        self.source_df_1 = self.ohlcv_base[f"{pair_symbol_1}-{timeframe}"].df.copy()
        self.source_df_2 = self.ohlcv_base[f"{pair_symbol_2}-{timeframe}"].df.copy()

        """ Warning! date feature reduced for lowest timeframe """
        if timeframe == '1m':
            cols_create = self.cols_create[-2:]
        else:
            cols_create = self.cols_create
        self.get_feature_datetime(self.source_df_1, cols_create=cols_create)

        features_df["close1"] = self.source_df_1["close"].copy()
        features_df.insert(1, "close2", self.source_df_2["close"].values)
        features_df.insert(2, "volume1", self.source_df_1["volume"].values)
        features_df.insert(3, "volume2", self.source_df_2["volume"].values)
        features_df.insert(4, "close1-close2", self.source_df_1["close"] - self.source_df_2["close"])
        features_df["log_close1"] = np.log(self.source_df_1["close"])
        features_df["log_close2"] = np.log(self.source_df_2["close"])
        features_df["log_volume1"] = np.log(self.source_df_1["volume"])
        features_df["log_volume2"] = np.log(self.source_df_2["volume"])
        features_df["diff_close1"] = self.source_df_1["close"].diff()
        features_df["diff_close2"] = self.source_df_2["close"].diff()

        """ Warning! NA must be cleared in final dataframe after shift """
        self.clear_na_flag = True
        shift_df1 = self.source_df_1["close"].shift(1)
        features_df["log_close1_close_shift1"] = self.source_df_1["close"]/shift_df1
        shift_df2 = self.source_df_2["close"].shift(1)
        features_df["log_close2_close_shift1"] = self.source_df_2["close"]/shift_df2
        features_df["sin_close1"] = np.sin(self.source_df_1['close'])
        features_df["sin_close2"] = np.sin(self.source_df_2['close'])
        if self.clear_na_flag:
            self.drop_idxs = features_df.loc[pd.isnull(features_df).any(1), :].index.values
            features_df = features_df.drop(index=self.drop_idxs)
        self.features_df = features_df.copy()
        return self.features_df

    def create_y_close1_close2_sub(self):
        self.y_df["close"] = self.source_df_1["close"]
        self.y_df = self.y_df["close"] - self.source_df_2["close"]
        self.y_df = self.y_df.drop(index=self.drop_idxs)
        return self.y_df.copy()

    # 0 if Close1 - Close2 < 0 и 1 if Close1 - Close2 >= 0 - в одном столбце
    def create_y_close1_close2_sub_trend(self):
        self.y_df["close"] = self.source_df_1["close"]
        normalized_df_1 = (self.source_df_1["close"] - self.source_df_1["close"].min()) / (
                self.source_df_1["close"].max() - self.source_df_1["close"].min())
        normalized_df_2 = (self.source_df_2["close"] - self.source_df_2["close"].min()) / (
                self.source_df_2["close"].max() - self.source_df_2["close"].min())
        temp_df = pd.DataFrame()
        temp_df["close"] = pd.DataFrame(normalized_df_1 - normalized_df_2)
        temp_df.loc[temp_df["close"] < 0, "close"] = 0
        temp_df.loc[temp_df["close"] > 0, "close"] = 1
        self.y_df = temp_df.drop(index=self.drop_idxs)
        return self.y_df.copy()

    # Предсказание силы движения по модулю (п1 переводим в ohe [1, 2, 3, 4, 5]) - классификация.
    # Идея в том, чтобы разделить предсказание направления [0, 1] и силу этого движения
    def create_y_close1_close2_sub_power(self):
        self.y_df["close"] = self.source_df_1["close"]
        normalized_df_1 = (self.source_df_1["close"] - self.source_df_1["close"].min()) / (
                self.source_df_1["close"].max() - self.source_df_1["close"].min())
        normalized_df_2 = (self.source_df_2["close"] - self.source_df_2["close"].min()) / (
                self.source_df_2["close"].max() - self.source_df_2["close"].min())
        sub_df = pd.DataFrame()
        sub_df["close"] = pd.DataFrame(normalized_df_1 - normalized_df_2).abs()
        # sub_df = self.create_y_close1_close2_sub()
        sub_df_min = sub_df.abs().min().min()
        sub_df_max = sub_df.abs().max().max()
        sub_df_step = (sub_df_max-sub_df_min)/5
        power_list = list(np.arange(sub_df_min, sub_df_max, sub_df_step))
        power_list.insert(5, sub_df_max)
        for idx in range(5):
            sub_df.loc[(sub_df["close"] >= power_list[idx]) & (sub_df["close"] < power_list[idx+1]+0.0001), "close"] = idx
        ohe = pd.get_dummies(sub_df["close"], dtype=float)
        self.y_df = ohe.drop(index=self.drop_idxs)
        return self.y_df.copy()

# if __name__ == "main":
#     loaded_crypto_data = DataLoad(pairs_symbols=None,
#                                   time_intervals=['15m'],
#                                   source_directory="../source_root",
#                                   start_period='2021-09-01 00:00:00',
#                                   end_period='2021-12-05 23:59:59',
#
#                                   )
#
#     fd = DataFeatures(loaded_crypto_data)
#     profile_1 = DSProfile()
#     x_df = fd.collect_features(profile_1)


