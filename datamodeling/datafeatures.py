import os
import sys
import numpy as np
import pandas as pd
from analyze.dataload import DataLoad, TradeConstants
# sys.path.insert(1, os.path.join(os.getcwd(), 'analyze'))

__version__ = 0.0004


class DataFeatures:
    def __init__(self, loader: DataLoad):
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
        pass

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

    def collect_features(self,
                         pair_symbol_1,
                         pair_symbol_2,
                         timeframe):

        features_df = pd.DataFrame()
        source_df_1 = self.ohlcv_base[f"{pair_symbol_1}-{timeframe}"].copy()
        source_df_2 = self.ohlcv_base[f"{pair_symbol_2}-{timeframe}"].copy()

        """ Warning! date feature reduced for lowest timeframe """
        if timeframe == '1m':
            cols_create = self.cols_create[:-2]
        else:
            cols_create = self.cols_create[:-2]
        self.get_feature_datetime(source_df_1.index, cols_create=cols_create)

        features_df["close1"] = source_df_1["close"]
        features_df.insert(1, "close2", source_df_2["close"].values())
        features_df.insert(2, "volume1", source_df_1["volume"].values())
        features_df.insert(3, "volume2", source_df_2["volume"].values())
        features_df.insert(4, "close1-close2", source_df_1["close"] - source_df_2["close"])
        features_df["log_close1"] = np.log(source_df_1["close"])
        features_df["log_close2"] = np.log(source_df_2["close"])
        features_df["log_volume1"] = np.log(source_df_1["volume"])
        features_df["log_volume2"] = np.log(source_df_2["volume"])
        features_df["diff_close1"] = source_df_1["close"].diff()
        features_df["diff_close2"] = source_df_2["close"].diff()
        shift_df1 = source_df_1["close"].shift(1)
        features_df["log_close1_close_shift1"] = source_df_1["close"]/shift_df1
        shift_df2 = source_df_2["close"].shift(1)
        features_df["log_close1_close_shift2"] = source_df_2["close"]/shift_df2
        features_df["sin_close1"] = np.sin(source_df_1['close'])
        features_df["sin_close2"] = np.sin(source_df_2['close'])
        return features_df

    def create_y(self):
        y_data = None
        return y_data



if __name__ == "main":
    fd = DataFeatures("/source_root")
    x_df = fd.collect_features("ETHUSDT", "BTCUSDT", "1m")


