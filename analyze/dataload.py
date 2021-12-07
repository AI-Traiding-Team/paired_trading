import os
import sys
import datetime
import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass

__version__ = 0.0006

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
        print(self.df.head().to_string())
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
                 time_intervals: list):
        self.pairs_symbols = pairs_symbols
        self.data: dict = {}
        pass


def create_cuts_from_data(source_directory,
                          target_directory,
                          pairs_list,
                          time_intervals,
                          start_period,
                          end_period,
                          suffix='cut'):
    for timeframe in time_intervals:
        for symbol in pairs_list:
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
             "AVAXUSDT"
             ]
    intervals = ['1m']

    create_cuts_from_data("/home/cubecloud/Python/projects/sunday_data/pairs_data/",
                          "/home/cubecloud/Python/projects/paired_trading/source_root",
                          pairs,
                          intervals,
                          '2021-09-01 00:00:00',
                          '2021-12-06 23:59:59')


    # pair = OHLCVData('/home/cubecloud/Python/projects/paired_trading/source_root/ADAUSDT-15m-data.csv')
    # pair.set_period('2021-09-01 00:00:00', '2021-12-06 23:59:59')