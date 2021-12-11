from analyze import DataLoad
import os
from backtester import Back
from backtester.strategies import *
from backtester.strategies.old_fashion import *
import backtrader as bt

import optuna

from maketarget import BigFatMommyMakesTargetMarkers

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    intervals = ['1m']
    root_path = os.getcwd()
    source_root_path = os.path.join(root_path, 'source_root')

    database = DataLoad(pairs_symbols=None,
                        time_intervals=intervals,
                        source_directory=source_root_path,
                        start_period='2021-01-01 00:00:00',
                        end_period='2021-01-10 23:59:59'
                        )

    print(database.pairs_symbols)

    for item in database.pairs_symbols:
        print(item)
        df = database.get_pair(item, intervals[0])
        df = BigFatMommyMakesTargetMarkers(window_size=10).mark_y(df)

        bt = Back(df, SimpleSignalStrategy, cash=100_000, commission=.002, trade_on_close=True)
        stats = bt.run()
        bt.plot(plot_volume=True, relative_equity=True)
