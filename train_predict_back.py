import time

from analyze import DataLoad
import os
from backtester import Back
from backtester.strategies import *
from backtester.strategies.old_fashion import *
import backtrader as bt
from optimizer import Objective

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
                        end_period='2021-03-31 23:59:59'
                        )

    print(database.pairs_symbols)

    window_size =41
    strategy = LongStrategy

    start = time.time()
    for item in database.pairs_symbols:
        print('===' * 30)
        print(item)
        # Загружаем исходные данные OHLCV
        df = database.get_pair(item, intervals[0])
        # Добавляем разметку Signal (значения [1, -1])
        df = BigFatMommyMakesTargetMarkers(window_size=window_size).mark_y(df)
        # Запускаем бэтестинг, на вход подаем DataFrame [['Open','Close','High','Low','Volume','Signal]]
        bt = Back(df, strategy, cash=100_000, commission=.002, trade_on_close=False)
        # Получаем статистику бэктестинга
        stats = bt.run(size=1000)
        # Печатаем статистику
        print(stats)
        # Выводим графиг бэктестинга (копия сохраняется в корень с именем "Название стратегии.html"
        bt.plot(plot_volume=True, relative_equity=True)
    print('===' * 30, '\nBacktesting done by: ', time.time() - start, '\n====' * 30, '\n')
