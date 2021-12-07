import os
import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass

__version__ = 0.0002

@dataclass
class TradeConstants:
    time_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    binsizes = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
                '1d': 1440, '3d': 4320, '1w': 10080}

class OHLCVData:
    def __init__(self,
                 path_filename: str):
        self.symbol_name: str = ''
        self.timeframe: str = ''
        self.path_filename: str = path_filename
        self.default_columns = ("timestamp",
                                "open",
                                "close",
                                "high",
                                "low",
                                "volume",
                                "trades")
        self.period_start: datetime = None
        self.period_end: datetime = None
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

    def load(self):
        self.df = pd.read_csv(self.path_filename, usecols=self.default_columns)
        pass

class DataLoad(object):
    def __init__(self,
                 pairs_symbols: list,
                 time_intervals: list):
        self.pairs_symbols = pairs_symbols
        self.data: dict = {}
        pass

    def
