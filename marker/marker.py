import os.path
import random
# from datamodeling.dscreator import DSCreator
from analyze import DataLoad
# from datamodeling.datafeatures import DSProfile
import pandas as pd
import numpy as np
from maketarget.mother import BigFatMommyMakesTargetMarkers

__version__ = 0.0011


class Marker:
    def __init__(self, loader: DataLoad):
        self.loader = loader
        self.source_df = None
        self.cols_create = ('year',
                            'quarter',
                            'month',
                            'weeknum',
                            'weekday',
                            'hour',
                            'minute'
                            )
        self.features_df = None
        self.symbol = None
        self.timeframe = None
        self.y_df = None
        self.clear_na_flag = None
        self.drop_idxs = None
        pass

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

        # TODO: create autoweight calculation based on current volatility (check hypothesis)
        def weighted_W(idx, W: float = 0.15):
            """
            Args:
                idx (int):  index of the row from pd.Dataframe
                W (float): increase or decrease weight (percentage) of the 'close' bar

            Returns:
            """
            weighted = W
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
    def split_datetime_data(datetime_index: pd.DatetimeIndex,
                            cols_defaults: tuple
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
            if col_name == 'year':
                if datetime_funct[col_name].nunique() != 1:
                    temp_df[col_name] = datetime_funct[col_name]
            else:
                temp_df[col_name] = datetime_funct[col_name]
        return temp_df

    @staticmethod
    def get_feature_datetime(input_df,
                             cols_create: tuple = ('year',
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
            object:     pd.Dataframe with columns with divided datetime
        """
        date_df = Marker.split_datetime_data(input_df.index, cols_create)
        return date_df

    @staticmethod
    def get_feature_datetime_ohe(input_df,
                                 cols_create: tuple = ('year',
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
            cols_create (list):       default columns names for encoding

        Returns:
            object:     pd.Dataframe with dummy encoded datetimeindex columns with prefix 'de_'
        """
        de_df = Marker.split_datetime_data(input_df.index, cols_create)

        cols_names = de_df.columns
        de_df = pd.get_dummies(de_df, columns=cols_names, drop_first=False)
        for col in de_df.columns:
            de_df.rename(columns={col: f'de_{col}'}, inplace=True)
        return de_df

    def collect_features(self):
        self.source_df = self.loader.ohlcvbase[f"{self.symbol}-{self.timeframe}"].df.copy()

        """ Warning! date feature reduced for lowest timeframe """
        # if self.timeframe == '1m':
        #     cols_create = self.cols_create[-2:]
        # else:
        #     cols_create = self.cols_create
        features_df = self.get_feature_datetime(self.source_df,
                                                self.cols_create
                                                )

        features_df["open"] = self.source_df["open"].copy()
        features_df["high"] = self.source_df["high"].copy()
        features_df["low"] = self.source_df["low"].copy()
        features_df["close"] = self.source_df["close"].copy()
        features_df["volume"] = self.source_df["volume"].copy()
        features_df["log_close"] = np.log(self.source_df["close"])
        features_df["log_volume"] = np.log(self.source_df["volume"])
        features_df["diff_close"] = self.source_df["close"].diff()

        conditions_0 = (features_df['open'] == features_df['low']) | (features_df['close'] == features_df['low'])
        features_df.loc[conditions_0, "open"] += random.uniform(1e-8, 5e-8)
        features_df.loc[conditions_0, "close"] += random.uniform(1e-8, 5e-8)

        conditions_1 = (features_df['open'] == features_df['high']) | (features_df['close'] == features_df['high'])
        features_df.loc[conditions_1, "open"] -= random.uniform(1e-8, 5e-8)
        features_df.loc[conditions_1, "close"] -= random.uniform(1e-8, 5e-8)
        # features_df = features_df.replace([np.inf, -np.inf], np.nan)
        # features_df = features_df.fillna(features_df.rolling(6, min_periods=1).mean())

        # count = np.isinf(features_df).values.sum()
        # print("It contains " + str(count) + " infinite values")

        """ Warning! NA must be cleared in final dataframe after shift """
        self.clear_na_flag = True
        shift_df = self.source_df["close"].shift(1)
        features_df["log_close_close_shift"] = self.source_df["close"]/shift_df
        features_df["sin_close"] = np.sin(self.source_df['close'])
        if self.clear_na_flag:
            self.drop_idxs = features_df.loc[pd.isnull(features_df).any(1), :].index.values
            features_df = features_df.drop(index=self.drop_idxs)
        self.features_df = features_df.copy()
        return self.features_df

    def collect_features_1(self):
        self.source_df = self.loader.ohlcvbase[f"{self.symbol}-{self.timeframe}"].df.copy()

        """ Warning! date feature reduced for lowest timeframe """
        if self.timeframe == '1m':
            cols_create = self.cols_create[-5:]
        else:
            cols_create = self.cols_create
        features_df = self.get_feature_datetime_ohe(self.source_df, cols_create=cols_create)
        features_df["open"] = self.source_df["open"].copy()
        features_df["high"] = self.source_df["high"].copy()
        features_df["low"] = self.source_df["low"].copy()
        features_df["close"] = self.source_df["close"].copy()
        features_df["volume"] = self.source_df["volume"].copy()
        features_df["log_close"] = np.log(self.source_df["close"])
        features_df["log_volume"] = np.log(self.source_df["volume"])
        features_df["diff_close"] = self.source_df["close"].diff()

        conditions_0 = (features_df['open'] == features_df['low']) | (features_df['close'] == features_df['low'])
        features_df.loc[conditions_0, "open"] += random.uniform(1e-8, 5e-8)
        features_df.loc[conditions_0, "close"] += random.uniform(1e-8, 5e-8)

        conditions_1 = (features_df['open'] == features_df['high']) | (features_df['close'] == features_df['high'])
        features_df.loc[conditions_1, "open"] -= random.uniform(1e-8, 5e-8)
        features_df.loc[conditions_1, "close"] -= random.uniform(1e-8, 5e-8)

        # features_df = features_df.replace([np.inf, -np.inf], np.nan)
        # features_df = features_df.fillna(features_df.rolling(6, min_periods=1).mean())
        count = np.isinf(features_df).values.sum()
        print("It contains " + str(count) + " infinite values")

        """ Warning! NA must be cleared in final dataframe after shift """
        self.clear_na_flag = True
        shift_df = self.source_df["close"].shift(1)
        features_df["log_close_close_shift"] = self.source_df["close"]/shift_df
        features_df["sin_close"] = np.sin(self.source_df['close'])
        if self.clear_na_flag:
            self.drop_idxs = features_df.loc[pd.isnull(features_df).any(1), :].index.values
            features_df = features_df.drop(index=self.drop_idxs)
        self.features_df = features_df.copy()
        return self.features_df

    def collect_features_2(self):
        self.source_df = self.loader.ohlcvbase[f"{self.symbol}-{self.timeframe}"].df.copy()

        cols_create = self.cols_create[-5:]
        features_df = self.get_feature_datetime(self.source_df, cols_create=cols_create)
        features_df["open"] = self.source_df["open"].copy()
        features_df["high"] = self.source_df["high"].copy()
        features_df["low"] = self.source_df["low"].copy()
        features_df["close"] = self.source_df["close"].copy()
        features_df["volume"] = self.source_df["volume"].copy()
        features_df["log_close"] = np.log(self.source_df["close"])
        features_df["log_volume"] = np.log(self.source_df["volume"])
        features_df["diff_close"] = self.source_df["close"].diff()

        conditions_0 = (features_df['open'] == features_df['low']) | (features_df['close'] == features_df['low'])
        features_df.loc[conditions_0, "open"] += random.uniform(1e-8, 5e-8)
        features_df.loc[conditions_0, "close"] += random.uniform(1e-8, 5e-8)

        conditions_1 = (features_df['open'] == features_df['high']) | (features_df['close'] == features_df['high'])
        features_df.loc[conditions_1, "open"] -= random.uniform(1e-8, 5e-8)
        features_df.loc[conditions_1, "close"] -= random.uniform(1e-8, 5e-8)

        # features_df = features_df.replace([np.inf, -np.inf], np.nan)
        # features_df = features_df.fillna(features_df.rolling(6, min_periods=1).mean())

        count = np.isinf(features_df).values.sum()
        print("It contains " + str(count) + " infinite values")

        """ Warning! NA must be cleared in final dataframe after shift """
        self.clear_na_flag = True
        shift_df = self.source_df["close"].shift(1)
        features_df["log_close_close_shift"] = self.source_df["close"]/shift_df
        features_df["sin_close"] = np.sin(self.source_df['close'])
        if self.clear_na_flag:
            self.drop_idxs = features_df.loc[pd.isnull(features_df).any(1), :].index.values
            features_df = features_df.drop(index=self.drop_idxs)
        self.features_df = features_df.copy()
        return self.features_df

    def create_power_trend(self, weight):
        ohlcv_df = self.source_df
        trend_df = pd.DataFrame()
        trend_df["trend"] = self.calculate_trend(ohlcv_df, weight)
        trend_df.loc[trend_df["trend"] == -1, "trend"] = 0
        trend_df = trend_df.drop(index=self.drop_idxs)
        self.y_df = trend_df
        return self.y_df

    def create_power_trend_tahn(self, weight):
        ohlcv_df = self.source_df
        trend_df = pd.DataFrame()
        trend_df["trend"] = self.calculate_trend(ohlcv_df, weight)
        trend_df = trend_df.drop(index=self.drop_idxs)
        self.y_df = trend_df
        return self.y_df

    def create_dataset_df_method_0(self, symbol, timeframe, target_directory='', save_file=True, weight=0.055):
        self.symbol = symbol
        self.timeframe = timeframe
        dataset_df = self.collect_features()
        dataset_df['Signal'] = self.create_power_trend(weight=weight)
        uniques, counts = np.unique(dataset_df['Signal'].values, return_counts=True)
        msg_2 = f"Signal type 2\n"
        for unq, cnt in zip(uniques, counts):
            msg_2 += f"Unique: {unq} {cnt}\n"
        msg = f"Pair: {self.symbol} - {self.timeframe}\n" \
              f"Dataframe shape: {dataset_df.shape} \n" \
              f"Trend weight: {weight}\n" \
              f"Start date: {self.loader.ohlcvbase[f'{self.symbol}-{self.timeframe}'].df.index[0]}\n" \
              f"End date: {self.loader.ohlcvbase[f'{self.symbol}-{self.timeframe}'].df.index[-1]}\n" \
              f"{msg_2}"

        print(msg)
        print(dataset_df.head(5).to_string(), f'\n')
        if save_file:
            path_filename = os.path.join(target_directory, self.timeframe, f'{self.symbol}-{self.timeframe}.csv')
            dataset_df.to_csv(path_filename)
        pass

    def create_dataset_df_method_1(self, symbol, timeframe, target_directory='', save_file=True, window_size=5):
        big_mommy = BigFatMommyMakesTargetMarkers(window_size=window_size)
        _ = self.collect_features()
        self.source_df = self.loader.ohlcvbase[f"{self.symbol}-{self.timeframe}"].df.copy()
        self.symbol = symbol
        self.timeframe = timeframe
        # dataset_df.columns = [item.lower().capitalize() for item in dataset_df.columns]
        dataset_df = self.source_df.copy()
        dataset_df = dataset_df.drop(index=self.drop_idxs)
        dataset_df = big_mommy.mark_y(dataset_df)
        uniques, counts = np.unique(dataset_df['Signal'].values, return_counts=True)
        msg_2 = f"Signal type 1\n"
        for unq, cnt in zip(uniques, counts):
            msg_2 += f"Unique: {unq} {cnt}\n"

        msg = f"Pair: {self.symbol} - {self.timeframe}\n" \
              f"Dataframe shape: {dataset_df.shape} \n" \
              f"Window size: {window_size}\n" \
              f"Start date: {self.loader.ohlcvbase[f'{self.symbol}-{self.timeframe}'].df.index[0]}\n" \
              f"End date: {self.loader.ohlcvbase[f'{self.symbol}-{self.timeframe}'].df.index[-1]}\n" \
              f"{msg_2}"

        print(msg)
        print(dataset_df.head(5).to_string(), f'\n')
        if save_file:
            path_filename = os.path.join(target_directory, self.timeframe, f'{self.symbol}-{self.timeframe}.csv')
            dataset_df.to_csv(path_filename)
        pass

    def create_dataset_df_method_2(self, symbol, timeframe, target_directory='', save_file=True, weight=0.055):
        self.symbol = symbol
        self.timeframe = timeframe
        dataset_df = self.collect_features()
        dataset_df['Signal'] = self.create_power_trend_tahn(weight=weight)
        current_trend = dataset_df.iloc[-1, -1]
        trend_length_list: list = []
        trend_counter = 1
        for idx in range(dataset_df.shape[0]-1, -1, -1):
            if dataset_df.iloc[idx, -1] == current_trend:
                trend_length_list.append(trend_counter)
                trend_counter += 1
            else:
                current_trend = dataset_df.iloc[idx, -1]
                trend_counter = 1
                trend_length_list.append(trend_counter)
        trend_length_list.reverse()
        dataset_df.insert(len(dataset_df.columns)-1, column="Trend_length", value=trend_length_list )

        uniques, counts = np.unique(dataset_df['Signal'].values, return_counts=True)
        msg_2 = f"Signal type 2\n"
        for unq, cnt in zip(uniques, counts):
            msg_2 += f"Unique: {unq} {cnt}\n"
        msg = f"Pair: {self.symbol} - {self.timeframe}\n" \
              f"Dataframe shape: {dataset_df.shape} \n" \
              f"Trend weight: {weight}\n" \
              f"Start date: {self.loader.ohlcvbase[f'{self.symbol}-{self.timeframe}'].df.index[0]}\n" \
              f"End date: {self.loader.ohlcvbase[f'{self.symbol}-{self.timeframe}'].df.index[-1]}\n" \
              f"{msg_2}"

        print(msg)
        print(dataset_df.head(5).to_string(), f'\n')
        if save_file:
            path_filename = os.path.join(target_directory, self.timeframe, f'{self.symbol}-{self.timeframe}.csv')
            dataset_df.to_csv(path_filename)
        pass

    def create_dataset_df_method_3(self, symbol, timeframe, target_directory='', save_file=True, weight=0.055):
        self.symbol = symbol
        self.timeframe = timeframe
        dataset_df = self.collect_features_1()
        dataset_df['Signal'] = self.create_power_trend(weight=weight)
        uniques, counts = np.unique(dataset_df['Signal'].values, return_counts=True)
        msg_2 = f"Signal type 2\n"
        for unq, cnt in zip(uniques, counts):
            msg_2 += f"Unique: {unq} {cnt}\n"
        msg = f"Pair: {self.symbol} - {self.timeframe}\n" \
              f"Dataframe shape: {dataset_df.shape} \n" \
              f"Trend weight: {weight}\n" \
              f"Start date: {self.loader.ohlcvbase[f'{self.symbol}-{self.timeframe}'].df.index[0]}\n" \
              f"End date: {self.loader.ohlcvbase[f'{self.symbol}-{self.timeframe}'].df.index[-1]}\n" \
              f"{msg_2}"

        print(msg)
        print(dataset_df.head(5).to_string(), f'\n')
        if save_file:
            path_filename = os.path.join(target_directory, self.timeframe, f'{self.symbol}-{self.timeframe}.csv')
            dataset_df.to_csv(path_filename)
        pass

    def create_dataset_df_method_4(self, symbol, timeframe, target_directory='', save_file=True, weight=0.055):
        self.symbol = symbol
        self.timeframe = timeframe
        dataset_df = self.collect_features_2()
        dataset_df['Signal'] = self.create_power_trend(weight=weight)
        uniques, counts = np.unique(dataset_df['Signal'].values, return_counts=True)
        msg_2 = f"Signal type 2\n"
        for unq, cnt in zip(uniques, counts):
            msg_2 += f"Unique: {unq} {cnt}\n"
        msg = f"Pair: {self.symbol} - {self.timeframe}\n" \
              f"Dataframe shape: {dataset_df.shape} \n" \
              f"Trend weight: {weight}\n" \
              f"Start date: {self.loader.ohlcvbase[f'{self.symbol}-{self.timeframe}'].df.index[0]}\n" \
              f"End date: {self.loader.ohlcvbase[f'{self.symbol}-{self.timeframe}'].df.index[-1]}\n" \
              f"{msg_2}"

        print(msg)
        print(dataset_df.head(5).to_string(), f'\n')
        if save_file:
            path_filename = os.path.join(target_directory, self.timeframe, f'{self.symbol}-{self.timeframe}.csv')
            dataset_df.to_csv(path_filename)
        pass

    def mark_all_loader_df(self, target_directory='', signal_method=1,  window_size=5, weight=0.0275):
        for idx, (key, ohlcv_obj) in enumerate(self.loader.ohlcvbase.items()):
            self.symbol = ohlcv_obj.symbol_name
            self.timeframe = ohlcv_obj.timeframe
            print(f'Symbol #{idx}')
            if signal_method == 0:
                self.create_dataset_df_method_0(self.symbol,
                                                timeframe=self.timeframe,
                                                target_directory=target_directory,
                                                weight=0.0275)
            elif signal_method == 1:
                self.create_dataset_df_method_1(self.symbol,
                                                timeframe=self.timeframe,
                                                target_directory=target_directory,
                                                window_size=window_size)
            elif signal_method == 2:
                self.create_dataset_df_method_2(self.symbol,
                                                timeframe=self.timeframe,
                                                target_directory=target_directory,
                                                weight=0.0275)
            elif signal_method == 3:
                self.create_dataset_df_method_3(self.symbol,
                                                timeframe=self.timeframe,
                                                target_directory=target_directory,
                                                weight=0.0275)
            elif signal_method == 4:
                self.create_dataset_df_method_4(self.symbol,
                                                timeframe=self.timeframe,
                                                target_directory=target_directory,
                                                weight=0.0275)
            else:
                assert signal_method > 2, f"Error! unknown method {signal_method}"

            pass


if __name__ == "__main__":
    loaded_crypto_data = DataLoad(pairs_symbols=None,
                                  time_intervals=['1m'],
                                  source_directory="../source_root",
                                  start_period='2021-09-01 00:00:00',
                                  end_period='2021-10-31 23:59:59',
                                  )
    mr = Marker(loaded_crypto_data)
    mr.mark_all_loader_df(target_directory="../source_ds2", signal_method=2,  weight=0.0275)
    # mr.mark_all_loader_df(target_directory="/Users/chekh/Development/Python/paired_trading/source_ds1", signal_method=1, window_size=5)
    # mr.create_dataset_df_method_0("ETHUSDT", timeframe="1m", target_directory="../source_ds", weight=0.0275)
    # mr.create_dataset_df_method_1("ETHUSDT", timeframe="1m", target_directory="../source_ds1", window_size = 5)
