import os
import sys
import datetime
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from typing import Tuple
from dataclasses import dataclass

__version__ = 0.0012

sns.set(style='ticks')


@dataclass
class TradeConstants:
    time_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    binsizes = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480,
                '12h': 720,
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

    @staticmethod
    def get_bad_idxs(ctrl_df, df):
        _df = pd.concat([ctrl_df, df], axis=1)
        _mask = _df['datetime'].isna()
        _indexes = _df[_mask].index.to_list()
        _indexes = [idx.to_pydatetime() for idx in _indexes]
        return _indexes

    @staticmethod
    def control_datetime_index(start_datetime, end_datetime, timeframe, ):
        interval = pd.date_range(start_datetime, end_datetime, freq='H')
        control_df = pd.DataFrame()
        control_df.index = interval
        pass

    def load(self):
        self.df = pd.read_csv(self.path_filename,
                              header=0,
                              names=self.ohlcv_cols,
                              usecols=self.ohlcv_cols,
                              dtype=self.ohlcv_dtypes)
        self.df.index, self.df.index.name = pd.to_datetime(self.df['datetime']), 'datetimeindex'
        if not pd.Index(self.df.index).is_monotonic_increasing:
            sys.exit('ERROR DataFrame index is not monotonic!')
        self.df = self.df.drop(columns=['datetime'])
        self.df_starts: datetime = self.df.index[0]
        self.df_ends: datetime = self.df.index[-1]
        pass

    def save_current_period(self, path_filename):
        self.df.to_csv(path_filename)
        pass


class DataLoad(object):
    def __init__(self,
                 source_directory,
                 pairs_symbols: list = None,
                 time_intervals: list = None,
                 start_period: str = None,
                 end_period: str = None,
                 ):
        self.start_period: str = start_period
        self.end_period: str = end_period
        self.pairs_symbols: list = pairs_symbols
        self.time_intervals: list = time_intervals
        self.source_directory: str = source_directory
        self.ohlcvbase: dict = {}
        self.get_all_data()
        pass

    def get_pair(self, pair_symbols, time_intervals):
        return self.ohlcvbase[f'{pair_symbols}-{time_intervals}'].df

    def get_all_data(self):
        dir_list = list()
        file_list = {'interval': list(),
                     'pair': list(),
                     'file_name': list()}
        # if don't have custom interval list check by all interval list
        no_intervals = False
        if self.time_intervals is None:
            no_intervals = True
            self.time_intervals = []
        # if have dirs in root dir check files just in dirs
        for _, dirs, _ in os.walk(self.source_directory, topdown=True):
            for dir_name in dirs:
                if no_intervals:
                    self.time_intervals.append(dir_name)
                    dir_list.append(dir_name)
                else:
                    if dir_name in self.time_intervals:
                        dir_list.append(dir_name)

        # If pairs list is empty will collect pairs in list
        no_pairs_symbols = False
        if self.pairs_symbols is None:
            no_pairs_symbols = True
            self.pairs_symbols = []

        # if dirs exists and in dirs have files work with them or take files from root dir
        if len(dir_list) != 0:
            for dir_name in dir_list:
                full_dir_name = os.path.join(self.source_directory, dir_name)
                for file in os.listdir(full_dir_name):
                    if no_pairs_symbols:          # if don't have custom pairs list take all names from file
                        split_file_name = file.split('-')   # name split by '-'
                        if split_file_name[0] not in self.pairs_symbols:
                            self.pairs_symbols.append(split_file_name[0])
                        file_list['interval'].append(dir_name)
                        file_list['pair'].append(split_file_name[0])
                        file_list['file_name'].append(os.path.join(full_dir_name, file))
                    else:
                        is_in_pairs = False
                        for pair in self.pairs_symbols:
                            is_in_pairs |= file.startswith(pair)
                        if is_in_pairs:
                            file_list['interval'].append(dir_name)
                            file_list['pair'].append(pair)
                            file_list['file_name'].append(os.path.join(full_dir_name, file))
        # or take files from root dir
        else:
            for file in os.listdir(self.source_directory):
                print(file)
                if no_pairs_symbols:
                    split_file_name = file.split('-')
                    if split_file_name[0] not in self.pairs_symbols:
                        self.pairs_symbols.append(split_file_name[0])
                    if no_intervals:
                        if split_file_name[1] not in self.time_intervals:
                            self.time_intervals.append(split_file_name[1])
                        file_list['interval'].append(split_file_name[1])
                        file_list['pair'].append(split_file_name[0])
                        file_list['file_name'].append(os.path.join(self.source_directory, file))
                    else:
                        if split_file_name[1] in self.time_intervals:
                            file_list['interval'].append(split_file_name[1])
                            file_list['pair'].append(split_file_name[0])
                            file_list['file_name'].append(os.path.join(self.source_directory, file))
                else:
                    for pair in self.pairs_symbols:
                        if file.startswith(pair):
                            split_file_name = file.split('-')
                            if no_intervals:
                                if split_file_name[1] not in self.time_intervals:
                                    self.time_intervals.append(split_file_name[1])
                                file_list['interval'].append(split_file_name[1])
                                file_list['pair'].append(pair)
                                file_list['file_name'].append(os.path.join(self.source_directory, file))
                            else:
                                if split_file_name[1] in self.time_intervals:
                                    file_list['interval'].append(split_file_name[1])
                                    file_list['pair'].append(pair)
                                    file_list['file_name'].append(os.path.join(self.source_directory, file))
        print(file_list)

        for index in range(len(file_list['file_name'])):
            source_path_filename = os.path.join(self.source_directory, file_list['file_name'][index])
            ohlcv = OHLCVData(source_path_filename)
            if (self.start_period is not None) and (self.end_period is not None):
                ohlcv.set_period(self.start_period, self.end_period)
            self.ohlcvbase.update({f"{file_list['pair'][index]}-{file_list['interval'][index]}": ohlcv})
        pass

class Analyze(DataLoad):
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
                normalized_df = (ohlcv_df[usecol] - ohlcv_df[usecol].min()) / (
                            ohlcv_df[usecol].max() - ohlcv_df[usecol].min())
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
                # for symbol in symbols_combo:

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

    def show_correlation(self, usecol='close'):
        merge_df = pd.DataFrame()
        for name, ohlcv in self.ohlcvbase.items():
            if merge_df.shape == (0, 0):
                merge_df.index = ohlcv.df.index
                merge_df.index.name = ohlcv.df.index.name
                merge_df[name] = ohlcv.df[usecol]
            else:
                temp = pd.DataFrame()
                temp.index = ohlcv.df.index
                temp.index.name = ohlcv.df.index.name
                temp[name] = ohlcv.df[usecol]
                merge_df = pd.merge(merge_df, temp, on='datetimeindex')
        correlation_matrix = [[0] * len(merge_df.columns) for i in range(len(merge_df.columns))]
        for i, col1 in enumerate(merge_df.columns):
            for j, col2 in enumerate(merge_df.columns):
                correlation_matrix[i][j] = np.corrcoef(merge_df[col1].values, merge_df[col2].values)[0, 1]

        plot = sns.heatmap(correlation_matrix, center=0)
        fig = plot.get_figure()
        fig.savefig('correlation.png')
    """
    0. Берем модули отклонений
    1. Размечает все что по модулю меньше комиссии как 0
    2. Считаем число того, что не 0
    3. Считаем среднюю того что не 0
    4. Строим таблицу
    5. Можно боксплоты построить, но можно и не строить
    """
    def diff_calculation(self, usecol: str = "close", commission: float = 0.1):
        result_df = pd.DataFrame(columns=["#",
                                          "pair_1",
                                          "pair_2",
                                          "commission",
                                          "(diff-comm).abs().sum()",
                                          "(diff-comm).abs().mean()",
                                          "(diff-comm).sum()",
                                          "(diff-comm).mean()"
                                          ])

        symbols_combo_list = [elem for elem in itertools.combinations(self.pairs_symbols, 2)]
        for idx, symbols_combo in enumerate(symbols_combo_list):
            for timeframe in self.time_intervals:
                ohlcv_df_1 = self.ohlcvbase[f"{symbols_combo[0]}-{timeframe}"].df.copy()
                ohlcv_df_2 = self.ohlcvbase[f"{symbols_combo[1]}-{timeframe}"].df.copy()
                normalized_df_1 = (ohlcv_df_1[usecol] - ohlcv_df_1[usecol].min()) / (
                        ohlcv_df_1[usecol].max() - ohlcv_df_1[usecol].min())
                normalized_df_2 = (ohlcv_df_2[usecol] - ohlcv_df_2[usecol].min()) / (
                        ohlcv_df_2[usecol].max() - ohlcv_df_2[usecol].min())
                diff_df = normalized_df_1 - normalized_df_2
                diff_df = pd.DataFrame(diff_df)
                diff_df.loc[(diff_df['close'] <= commission) & (diff_df['close'] >= -commission)] = 0
                diff_abs_sum = diff_df.abs().sum()
                diff_abs_mean = diff_df.abs().mean()
                diff_sum = diff_df.sum()
                diff_mean = diff_df.mean()
                result_df = result_df.append({"#": idx + 1,
                                              "pair_1": f"{symbols_combo[0]}-{timeframe}",
                                              "pair_2": f"{symbols_combo[1]}-{timeframe}",
                                              "commission": commission,
                                              "(diff-comm).abs().sum()": float(diff_abs_sum),
                                              "(diff-comm).abs().mean()": float(diff_abs_mean),
                                              "(diff-comm).sum()": float(diff_sum),
                                              "(diff-comm).mean()": float(diff_mean),
                                              },
                                             ignore_index=True
                                             )
        print(result_df.to_string())
        pass

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
    pairs = None
    # ["BTCUSDT",
    #      "ETHUSDT",
    #      "BNBUSDT",
    #      "SOLUSDT",
    #      "ADAUSDT",
    #      "USDCUSDT",
    #      "XRPUSDT",
    #      "DOTUSDT",
    #      "LUNAUSDT",
    #      "DOGEUSDT",
    #      # "AVAXUSDT"
   # ]

    intervals = ['15m']
    show_data = Analyze(pairs_symbols=pairs,
                       time_intervals=intervals,
                       source_directory="../source_root",
                       start_period='2021-09-01 00:00:00',
                       end_period='2021-12-05 23:59:59'
                       )
    # show_data.show_all_data()
    # show_data.show_combinations_diff(savepath="/home/cubecloud/Python/projects/paired_trading/analyze/pics")
    #show_data.diff_calculation()
    show_data.show_correlation()
    # Analyze.create_cuts_from_data("/home/cubecloud/Python/projects/sunday_data/pairs_data/",
    #                                "/home/cubecloud/Python/projects/paired_trading/source_root",
    #                                pairs,
    #                                intervals,
    #                                '2021-09-01 00:00:00',
    #                                '2021-12-06 23:59:59')

    # pair = OHLCVData('/home/cubecloud/Python/projects/paired_trading/source_root/ADAUSDT-15m-data.csv')
    # pair.set_period('2021-09-01 00:00:00', '2021-12-06 23:59:59')
