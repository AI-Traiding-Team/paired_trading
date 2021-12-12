import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple
from analyze.dataload import DataLoad
from dataclasses import dataclass
# sys.path.insert(1, os.path.join(os.getcwd(), 'analyze'))

__version__ = 0.0010


@dataclass(init=True)
class DSProfile:
    features_list: Tuple = ()
    use_symbols_pairs = ("BTCUSDT", "ETHUSDT")
    timeframe: str = '15m'
    Y_data: str = "close1-close2"
    power_trend = 0.15

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
        self.source_df_3 = None
        self.features_df = None
        self.ds_profile = None
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
    def calculate_trend(input_df: pd.DataFrame,
                        W: float = 0.15
                        ) -> pd.DataFrame:
        """
        Args:
            input_df (pd.DataFrame):    input DataFrame with index and OHCLV data
            W (float):                  weight percentage for calc trend default 0.15
        Returns:
           trend (pd.DataFrame):        output DataFrame
        """

        # volatility = calc_volatility(df)
        # max_vol = volatility.max()
        # min_vol = volatility.min()
        # std_vol = volatility.std()
        # vol_scaling = max_vol*W/min_vol
        # print(f"{min_vol}/{max_vol}= {min_vol/max_vol}, {std_vol} scaling={vol_scaling}")

        # TODO: create autoweight calculation based on current volatility (check hypothesis)
        def weighted_W(idx, W: float = 0.15):
            """
            Args:
                idx (int):  index of the row from pd.Dataframe
                W (float): increase or decrease weight (percentage) of the 'close' bar

            Returns:
            """
            weighted = W
            # if volatility[idx]< 0.9:
            #     weighted = 0.13
            # else:
            #     weighted = W - (np.log(volatility[idx])*0.8)
            #     # print(weighted)
            return weighted

        """
        Setup. Getting first data and initialize variables
        """
        x_bar_0 = [input_df.index[0],  # [0] - datetime
                   input_df["open"][0],  # [1] - open
                   input_df["high"][0],  # [2] - high
                   input_df["low"][0],  # [3] - low
                   input_df["close"][0],  # [4] - CLOSE
                   input_df["volume"][0],
                   # input_df["trades"][0]
                   ]
        FP_first_price = x_bar_0[4]
        xH_highest_price = x_bar_0[2]
        HT_highest_price_timemark = 0
        xL_lowest_price = x_bar_0[3]
        LT_lowest_price_timemark = 0
        Cid = 0
        FPN_first_price_idx = 0
        Cid_array = np.zeros(input_df.shape[0])
        """
        Setup. Getting first data and initialize variables
        """

        for idx in range(input_df.shape[0] - 1):
            x_bar = [input_df.index[idx],
                     input_df["open"][idx],
                     input_df["high"][idx],
                     input_df["low"][idx],
                     input_df["close"][idx],
                     input_df["volume"][idx],
                     # input_df["trades"][idx]
                     ]
            # print(x_bar)
            # print(x_bar[4])
            W = weighted_W(idx, W)
            if x_bar[2] > (FP_first_price + x_bar_0[4] * W):
                xH_highest_price = x_bar[2]
                HT_highest_price_timemark = idx
                FPN_first_price_idx = idx
                Cid = 1
                Cid_array[idx] = 1
                Cid_array[0] = 1
                break
            if x_bar[3] < (FP_first_price - x_bar_0[4] * W):
                xL_lowest_price = x_bar[3]
                LT_lowest_price_timemark = idx
                FPN_first_price_idx = idx
                Cid = -1
                Cid_array[idx] = -1
                Cid_array[0] = -1
                break

        for ix in range(FPN_first_price_idx + 1, input_df.shape[0] - 2):
            x_bar = [input_df.index[ix],
                     input_df["open"][ix],
                     input_df["high"][ix],
                     input_df["low"][ix],
                     input_df["close"][ix],
                     input_df["volume"][ix],
                     # input_df["trades"][ix]
                     ]
            W = weighted_W(ix, W)
            if Cid > 0:
                if x_bar[2] > xH_highest_price:
                    xH_highest_price = x_bar[2]
                    HT_highest_price_timemark = ix
                if x_bar[2] < (
                        xH_highest_price - xH_highest_price * W) and LT_lowest_price_timemark <= HT_highest_price_timemark:
                    for j in range(1, input_df.shape[0] - 1):
                        if LT_lowest_price_timemark < j <= HT_highest_price_timemark:
                            Cid_array[j] = 1
                    xL_lowest_price = x_bar[2]
                    LT_lowest_price_timemark = ix
                    Cid = -1

            if Cid < 0:
                if x_bar[3] < xL_lowest_price:
                    xL_lowest_price = x_bar[3]
                    LT_lowest_price_timemark = ix
                if x_bar[3] > (
                        xL_lowest_price + xL_lowest_price * W) and HT_highest_price_timemark <= LT_lowest_price_timemark:
                    for j in range(1, input_df.shape[0] - 1):
                        if HT_highest_price_timemark < j <= LT_lowest_price_timemark:
                            Cid_array[j] = -1
                    xH_highest_price = x_bar[2]
                    HT_highest_price_timemark = ix
                    Cid = 1

        # TODO: rewrite this block in intelligent way !!! Now is working but code is ugly
        """ Checking last bar in input_df """
        ix = input_df.shape[0] - 1
        x_bar = [input_df.index[ix],
                 input_df["open"][ix],
                 input_df["high"][ix],
                 input_df["low"][ix],
                 input_df["close"][ix],
                 input_df["volume"][ix],
                 # input_df["trades"][ix]
                 ]
        if Cid > 0:
            if x_bar[2] > xH_highest_price:
                xH_highest_price = x_bar[2]
                HT_highest_price_timemark = ix
            if x_bar[2] <= xH_highest_price:
                for j in range(1, input_df.shape[0]):
                    if LT_lowest_price_timemark < j <= HT_highest_price_timemark:
                        Cid_array[j] = 1
                xL_lowest_price = x_bar[3]
                LT_lowest_price_timemark = ix
                Cid = -1
        if Cid < 0:
            if x_bar[3] < xL_lowest_price:
                xL_lowest_price = x_bar[3]
                LT_lowest_price_timemark = ix
                # print(True)
            if x_bar[3] >= xL_lowest_price:
                for j in range(1, input_df.shape[0]):
                    if HT_highest_price_timemark < j <= LT_lowest_price_timemark:
                        Cid_array[j] = -1
                xH_highest_price = x_bar[2]
                HT_highest_price_timemark = ix
                Cid = 1
        if Cid > 0:
            if x_bar[2] > xH_highest_price:
                xH_highest_price = x_bar[2]
                HT_highest_price_timemark = ix
            if x_bar[2] <= xH_highest_price:
                for j in range(1, input_df.shape[0]):
                    if LT_lowest_price_timemark < j <= HT_highest_price_timemark:
                        Cid_array[j] = 1
                xL_lowest_price = x_bar[3]
                LT_lowest_price_timemark = ix
                Cid = -1
        trend = pd.DataFrame(data=Cid_array,
                             index=input_df.index,
                             columns=["trend"])
        return trend

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
        self.ds_profile = profile
        pair_symbol_1 = self.ds_profile.use_symbols_pairs[0]
        pair_symbol_2 = self.ds_profile.use_symbols_pairs[1]
        timeframe = self.ds_profile.timeframe
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
        temp_df = pd.DataFrame()
        temp_df["close1"] = self.source_df_1["close"]
        temp_df.insert(1, "close2", self.source_df_2["close"].values)
        temp_df["close1/close2"] = temp_df["close1"]/temp_df["close2"]
        temp_df["close1/close2_pct"] = temp_df["close1/close2"].pct_change(1)
        temp_df.loc[temp_df["close1/close2_pct"] <= 0, "close1/close2_pct"] = 0
        temp_df.loc[temp_df["close1/close2_pct"] > 0, "close1/close2_pct"] = 1
        temp_df = temp_df["close1/close2_pct"]
        temp_df = temp_df.drop(index=self.drop_idxs)
        ohe = pd.get_dummies(temp_df, dtype=float)
        self.y_df = pd.DataFrame(ohe)
        # unique = np.unique(self.y_df, return_counts=True )
        return self.y_df

    def create_power_trend(self, weight):
        pair_symbol = self.ds_profile.use_symbols_pairs[2]
        timeframe = self.ds_profile.timeframe
        self.source_df_3 = self.ohlcv_base[f"{pair_symbol}-{timeframe}"].df.copy()
        ohlcv_df = self.source_df_3
        trend_df = pd.DataFrame()
        trend_df["trend"] = self.calculate_trend(ohlcv_df, weight)
        trend_df.loc[trend_df["trend"] == -1, "trend"] = 0
        trend_df = trend_df.drop(index=self.drop_idxs)
        ohe_df = pd.get_dummies(trend_df["trend"], dtype=float)
        self.y_df = ohe_df
        return self.y_df

    # # Предсказание силы движения по модулю (п1 переводим в ohe [1, 2, 3, 4, 5]) - классификация.
    # # Идея в том, чтобы разделить предсказание направления [0, 1] и силу этого движения
    def create_y_close1_close2_sub_power(self):
        temp_df = pd.DataFrame()
        temp_df["close1"] = self.source_df_1["close"]
        temp_df.insert(1, "close2", self.source_df_2["close"].values*10)
        temp_df["close1-close2"] = temp_df["close1"]-temp_df["close2"]
        temp_df["close1-close2_pct_change"] = temp_df["close1-close2"].pct_change(1).abs()
        # normalized_df_1 = (self.source_df_1["close"] - self.source_df_1["close"].min()) / (
        #         self.source_df_1["close"].max() - self.source_df_1["close"].min())
        # normalized_df_2 = (self.source_df_2["close"] - self.source_df_2["close"].min()) / (
        #         self.source_df_2["close"].max() - self.source_df_2["close"].min())
        # sub_df = pd.DataFrame()
        # sub_df["close"] = pd.DataFrame(normalized_df_1 - normalized_df_2).abs()
        # sub_df["close"] = sub_df["close"].pct_change(1).abs()

        # temp_df = pd.DataFrame()
        temp_df["close1"] = self.source_df_1["close"]
        temp_df.insert(1, "close2", self.source_df_2["close"].values)
        temp_df["close1/close2"] = temp_df["close1"]/temp_df["close2"]
        temp_df["close1/close2_pct_change"] = (temp_df["close1"]/temp_df["close2"]).pct_change(1)
        temp_df["close1/close2_pct_change"] = temp_df["close1/close2_pct_change"].abs()
        temp_df = temp_df.drop(index=self.drop_idxs)

        # y_list = list(temp_df["close1/close2_pct"].values)
        # classes_idx_dict = {0: [],
        #                     1: [],
        #                     2: [],
        #                     3: [],
        #                     4: [],
        #                     }
        # classes_len_dict = {0: 0,
        #                     1: 0,
        #                     2: 0,
        #                     3: 0,
        #                     4: 0,
        #                     }
        # y_len = len(y_list)
        # for class_num in range(4):
        #     classes_len_dict[class_num] = y_len/5
        # classes_len_dict[5] = y_len - (y_len/5*4)
        # masking = np.zeros(y_len, dtype=np.bool)
        # y_masked = np.ma.MaskedArray(y_list, mask=masking)
        # for class_num in range(5):
        #     print(class_num)
        #     for _ in range(classes_len_dict[class_num]):
        #         idx = y_masked.argmin()
        #         classes_idx_dict[class_num].append(idx)
        #         y_masked.mask[idx] = True
        # y_classes = np.zeros(y_len, dtype=np.int)

        # for class_num in range(5):
        #     for idx in classes_dict[class_num]:
        #         y_classes[idx] = class_num
        # print(y_classes)

        # quantile_transformer = QuantileTransformer(output_distribution='normal',
        #                                            random_state=42)
        # y_arr = temp_df["close1/close2_pct"].values
        # y_arr = y_arr.reshape(-1, 1)
        #
        #
        # normalized_df = (temp_df["close1/close2_pct"] - temp_df["close1/close2_pct"].min()) / (
        #         temp_df["close1/close2_pct"].max() - temp_df["close1/close2_pct"].min())
        # y_arr = normalized_df.values
        # y_arr = y_arr.reshape(-1, 1)
        # y_trans = quantile_transformer.fit_transform(y_arr)
        # temp_df_min = y_trans.min().min()
        # temp_df_max = y_trans.max().max()
        # # temp_df_min = normalized_df.min().min()
        # # temp_df_max = normalized_df.max().max()
        # all_range = temp_df_max-temp_df_min
        # temp_df_step = all_range/5
        # power_list = list(np.arange(temp_df_min, temp_df_max, temp_df_step))
        # power_list.insert(5, temp_df_max)
        # # temp_df = normalized_df
        # y_trans = np.squeeze(y_trans)
        # y_df = pd.DataFrame(data=y_trans, columns=["power"])
        # y_df.index = temp_df.index
        # temp_df = y_df
        # for idx in range(1, 6):
        #     temp_df.loc[(temp_df["power"] > power_list[idx-1]) & (temp_df["power"] <= power_list[idx])] = idx-1
        # # temp_df = temp_df.drop(index=self.drop_idxs)
        unique = np.unique(temp_df.iloc[:,], return_counts=True )
        ohe_df = pd.get_dummies(temp_df, dtype=float)
        self.y_df = ohe_df
        return self.y_df.copy()

    # # Предсказание силы движения по модулю (п1 переводим в ohe [1, 2, 3, 4, 5]) - классификация.
    # # Идея в том, чтобы разделить предсказание направления [0, 1] и силу этого движения
    # def create_y_close1_close2_sub_power(self):
    #     self.y_df["close"] = self.source_df_1["close"]
    #     normalized_df_1 = (self.source_df_1["close"] - self.source_df_1["close"].min()) / (
    #             self.source_df_1["close"].max() - self.source_df_1["close"].min())
    #     normalized_df_2 = (self.source_df_2["close"] - self.source_df_2["close"].min()) / (
    #             self.source_df_2["close"].max() - self.source_df_2["close"].min())
    #     sub_df = pd.DataFrame()
    #     sub_df["close"] = pd.DataFrame(normalized_df_1 - normalized_df_2).abs()
    #     # sub_df = self.create_y_close1_close2_sub()
    #     sub_df_min = sub_df.abs().min().min()
    #     sub_df_max = sub_df.abs().max().max()
    #     sub_df_step = (sub_df_max-sub_df_min)/5
    #     power_list = list(np.arange(sub_df_min, sub_df_max, sub_df_step))
    #     power_list.insert(5, sub_df_max)
    #     for idx in range(5):
    #         sub_df.loc[(sub_df["close"] >= power_list[idx]) & (sub_df["close"] < power_list[idx+1]+0.0001), "close"] = idx
    #     ohe = pd.get_dummies(sub_df["close"], dtype=float)
    #     self.y_df = ohe.drop(index=self.drop_idxs)
    #     return self.y_df.copy()

if __name__ == "main":
    loaded_crypto_data = DataLoad(pairs_symbols=None,
                                  time_intervals=['15m'],
                                  source_directory="../source_root",
                                  start_period='2021-09-01 00:00:00',
                                  end_period='2021-12-05 23:59:59',

                                  )

    fd = DataFeatures(loaded_crypto_data)
    profile_1 = DSProfile()
    x_df = fd.collect_features(profile_1)


