import os
import sys
import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import itertools
import matplotlib.pyplot as plt

from typing import Tuple
from dataclasses import dataclass

__version__ = 0.0008

@dataclass
class TradeConstants:
    time_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    binsizes = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
                '1d': 1440, '3d': 4320, '1w': 10080}
    default_datetime_format: str = "%Y-%m-%d %H:%M:%S"
    ohlcv_cols: Tuple = ('datetime',
                         'open',
                         'high',
                         'low',
                         'close',
                         'volume',
                         )
    ohlcv_dtypes = {'datetime': str,
                    'open': float,
                    'high': float,
                    'low': float,
                    'close': float,
                    'volume': float,
                    }
    pass


class OHLCVData:
    def __init__(self,
                 path_filename: str):
        self.symbol_name: str = ''
        self.timeframe: str = ''
        self.path_filename: str = path_filename
        self.ohlcv_cols = TradeConstants.ohlcv_cols
        self.ohlcv_dtypes = TradeConstants.ohlcv_dtypes
        self.df_starts: datetime = None
        self.df_ends: datetime = None
        self.start_datetime: datetime = None
        self.end_datetime: datetime = None
        self.df = pd.DataFrame()
        self.__set_symbol_interval_names()
        self.load()
        pass

    def __set_symbol_interval_names(self):
        filename = os.path.split(self.path_filename)[-1]
        splited_text = filename.split('-')
        self.symbol_name = splited_text[0]
        self.timeframe = splited_text[1]
        assert self.symbol_name.isalpha(), 'Error: name of pair symbol is not alphabet'
        assert self.timeframe[:-1].isnumeric(), 'Error: timeframe is not correct'
        assert self.timeframe in TradeConstants.time_intervals, 'Error: timeframe is not correct'
        pass

    def set_period(self, start_period, end_period):
        """
        Checking data for start & end period
        If dataframe have data for this period of time, dataframe will be cuted and saved to self.df
        If dataframe doesn't have this period iof data and dataframe? data wil be loaded from original file
        (if period doesn't exist in original file  -> Error occurred)
        Setting start_period and end_period

        Returns:
            None
        """

        self.start_datetime = datetime.datetime.strptime(start_period, TradeConstants.default_datetime_format)
        self.end_datetime = datetime.datetime.strptime(end_period, TradeConstants.default_datetime_format)

        if self.df.index[0] > self.start_datetime:
            if self.df.index[0] > self.df_starts:
                msg = f"Error: Start period is early then data in file starts"
                sys.exit(msg)
            else:
                self.load()
        if self.df.index[-1] < self.end_datetime:
            if self.df.index[-1] < self.df_ends:
                msg = f"Error: End period is greater hen data ends"
                sys.exit(msg)
            else:
                self.load()
        self.df = self.df.loc[(self.df.index >= self.start_datetime) & (self.df.index <= self.end_datetime)]
        # print(self.df.head().to_string())
        pass

    def load(self):
        self.df = pd.read_csv(self.path_filename,
                              header=0,
                              names=self.ohlcv_cols,
                              usecols=self.ohlcv_cols,
                              dtype=self.ohlcv_dtypes)
        self.df.index, self.df.index.name = pd.to_datetime(self.df['datetime']), 'datetimeindex'
        self.df = self.df.drop(columns=['datetime'])
        self.df_starts: datetime = self.df.index[0]
        self.df_ends: datetime = self.df.index[-1]
        pass

    def save_current_period(self, path_filename):
        self.df.to_csv(path_filename)
        pass


class DataLoad(object):
    def __init__(self,
                 pairs_symbols: list,
                 time_intervals: list,
                 source_directory,
                 start_period: str = None,
                 end_period: str = None,
                 ):
        self.start_period: str = start_period
        self.end_period: str = end_period
        self.pairs_symbols: list = pairs_symbols
        self.time_intervals: list = time_intervals
        self.source_directory: str = source_directory
        self.ohlcvbase: dict = {}
        self.all_symbols_close = {}
        self._get_all_data()
        pass

    def get_pair(self, pair_symbols, time_intervals):
        return self.ohlcvbase[f'{pair_symbols}-{time_intervals}'].df

    def get_all_close(self):
        for timeframe in self.time_intervals:
            data = [self.ohlcvbase[f"{symbol}-{timeframe}"].df['close'].rename(symbol) for symbol in self.pairs_symbols]
            self.all_symbols_close.update({f'{timeframe}': pd.concat(data, axis=1)})


    def _get_all_data(self):
        for timeframe in self.time_intervals:
            for symbol in self.pairs_symbols:
                source_filename = f'{symbol}-{timeframe}-data.csv'
                source_path_filename = os.path.join(self.source_directory, source_filename)
                ohlcv = OHLCVData(source_path_filename)
                if (self.start_period is not None) and (self.end_period is not None):
                    ohlcv.set_period(self.start_period, self.end_period)
                self.ohlcvbase.update({f"{symbol}-{timeframe}": ohlcv})

    def show_all_data(self, usecol='close'):
        plt.figure(figsize=(45, 18))
        # Don't allow the axis to be on top of your data
        # plt.set_axisbelow(True)
        # Turn on the minor TICKS, which are required for the minor GRID
        plt.minorticks_on()
        # Customize the major grid
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        # Customize the minor grid
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        for timeframe in self.time_intervals:
            for symbol in self.pairs_symbols:
                ohlcv_df = self.ohlcvbase[f"{symbol}-{timeframe}"].df.copy()
                normalized_df = (ohlcv_df[usecol] - ohlcv_df[usecol].min()) / (ohlcv_df[usecol].max() - ohlcv_df[usecol].min())
                plt.plot(normalized_df,
                         label=f"{symbol}-{timeframe}")
        plt.legend()
        plt.show()
        pass

    def show_combinations_diff(self, usecol='close', savepath: str = None):
        symbols_combo_list = [elem for elem in itertools.combinations(self.pairs_symbols, 2)]
        for symbols_combo in symbols_combo_list:
            for timeframe in self.time_intervals:
                plt.figure(figsize=(45, 18))
                # Don't allow the axis to be on top of your data
                # plt.set_axisbelow(True)
                # Turn on the minor TICKS, which are required for the minor GRID
                plt.minorticks_on()
                # Customize the major grid
                plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
                # Customize the minor grid
                plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                for symbol in symbols_combo:
                    ohlcv_df_1 = self.ohlcvbase[f"{symbols_combo[0]}-{timeframe}"].df.copy()
                    ohlcv_df_2 = self.ohlcvbase[f"{symbols_combo[1]}-{timeframe}"].df.copy()
                    normalized_df_1 = (ohlcv_df_1[usecol] - ohlcv_df_1[usecol].min()) / (
                                ohlcv_df_1[usecol].max() - ohlcv_df_1[usecol].min())
                    normalized_df_2 = (ohlcv_df_2[usecol] - ohlcv_df_2[usecol].min()) / (
                                ohlcv_df_2[usecol].max() - ohlcv_df_2[usecol].min())
                    diff_df = normalized_df_1 - normalized_df_2
                    plt.plot(diff_df, label=f"{symbols_combo[0]}-{symbols_combo[1]}-{timeframe}")
                plt.legend()
                if savepath is None:
                    plt.show()
                else:
                    path_filename = os.path.join(savepath, f"{symbols_combo[0]}-{symbols_combo[1]}-{timeframe}.png")
                    plt.savefig(path_filename)
                    plt.show()
        pass

    # def show_histogram_diff(self, usecol='close', savepath: str = None):
    #     symbols_combo_list = [elem for elem in itertools.combinations(self.pairs_symbols, 2)]
    #     for symbols_combo in symbols_combo_list:
    #         for timeframe in self.time_intervals:
    #             plt.figure(figsize=(45, 18))
    #             # Don't allow the axis to be on top of your data
    #             # plt.set_axisbelow(True)
    #             # Turn on the minor TICKS, which are required for the minor GRID
    #             plt.minorticks_on()
    #             # Customize the major grid
    #             plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    #             # Customize the minor grid
    #             plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    #             for symbol in symbols_combo:
    #                 ohlcv_df_1 = self.ohlcvbase[f"{symbols_combo[0]}-{timeframe}"].df.copy()
    #                 ohlcv_df_2 = self.ohlcvbase[f"{symbols_combo[1]}-{timeframe}"].df.copy()
    #                 normalized_df_1 = (ohlcv_df_1[usecol] - ohlcv_df_1[usecol].min()) / (
    #                             ohlcv_df_1[usecol].max() - ohlcv_df_1[usecol].min())
    #                 normalized_df_2 = (ohlcv_df_2[usecol] - ohlcv_df_2[usecol].min()) / (
    #                             ohlcv_df_2[usecol].max() - ohlcv_df_2[usecol].min())
    #                 diff_df = normalized_df_1 - normalized_df_2
    #                 diff_df
    #                 plt.plot(diff_df, label=f"{symbols_combo[0]}-{symbols_combo[1]}-{timeframe}")
    #             plt.legend()
    #             if savepath is None:
    #                 plt.show()
    #             else:
    #                 path_filename = os.path.join(savepath, f"{symbols_combo[0]}-{symbols_combo[1]}-{timeframe}.png")
    #                 plt.savefig(path_filename)
    #                 plt.show()
    #     pass



    @staticmethod
    def create_cuts_from_data(source_directory,
                              target_directory,
                              pairs_symbols,
                              time_intervals,
                              start_period,
                              end_period,
                              suffix='cut'):
        for timeframe in time_intervals:
            for symbol in pairs_symbols:
                source_filename = f'{symbol}-{timeframe}-data.csv'
                source_path_filename = os.path.join(source_directory, source_filename)
                ohlcv = OHLCVData(source_path_filename)
                ohlcv.set_period(start_period, end_period)
                target_filename = f'{symbol}-{timeframe}-{suffix}.csv'
                target_path_filename = os.path.join(target_directory, target_filename)
                ohlcv.save_current_period(target_path_filename)

    pass


if __name__ == '__main__':
    """
    Current list contains list of TOP 12 cryptocurrencies by https://coinmarketcap.com/
    w/o stable crypto coin USDT (TOP #3),  
    The of DOGE TOP #11 and AVAX TOP #12 have very close positions 2021/12/07 
    """
    pairs = ["BTCUSDT",
             "ETHUSDT",
             "BNBUSDT",
             "SOLUSDT",
             "ADAUSDT",
             "USDCUSDT",
             "XRPUSDT",
             "DOTUSDT",
             "LUNAUSDT",
             "DOGEUSDT",
             # "AVAXUSDT"
             ]
    intervals = ['1m']

    database = DataLoad(pairs_symbols=pairs,
                        time_intervals=intervals,
                        source_directory="/home/cubecloud/Python/projects/sunday_data/pairs_data/",
                        start_period='2021-09-01 00:00:00',
                        end_period='2021-12-06 23:59:59'
                        )
    # database.show_all_data()
    database.show_combinations_diff(savepath="/home/cubecloud/Python/projects/paired_trading/analyze/pics")



    # DataLoad.create_cuts_from_data("/home/cubecloud/Python/projects/sunday_data/pairs_data/",
    #                                "/home/cubecloud/Python/projects/paired_trading/source_root",
    #                                pairs,
    #                                intervals,
    #                                '2021-09-01 00:00:00',
    #                                '2021-12-06 23:59:59')

    # pair = OHLCVData('/home/cubecloud/Python/projects/paired_trading/source_root/ADAUSDT-15m-data.csv')
    # pair.set_period('2021-09-01 00:00:00', '2021-12-06 23:59:59')