import time

from analyze import DataLoad
import os
from backtester import Back
from backtester.strategies import *
from backtester.strategies.old_fashion import *
import backtrader as bt
from optimizer import Objective

import optuna
from powerrun.functions import TrainNN, MarkedDataSet
from powerrun.models import get_resnet1d_model, get_angry_bird_model
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

    path_filename = "source_ds/1m/ETHUSDT-1m.csv"

    print(database.pairs_symbols)

    window_size =41
    strategy = LongStrategy

    start = time.time()
    # for item in database.pairs_symbols:
    print('===' * 30)
    # print(item)
    # Загружаем исходные данные OHLCV
    # df = database.get_pair(item, intervals[0])
    dataset = MarkedDataSet(path_filename, df, df_priority=False)
    # Добавляем разметку Signal (значения [1, -1])
    tr = TrainNN(dataset)
    tr.tsg_window_length = 40
    # tr.keras_model = get_angry_bird_model((tr.tsg_window_length))
    tr.epochs = 20

    tr.train()
    df = tr.backtest_test_dataset()
    # df = BigFatMommyMakesTargetMarkers(window_size=window_size).mark_y(df)
    # Запускаем бэтестинг, на вход подаем DataFrame [['Open','Close','High','Low','Volume','Signal]]
    bt = Back(df, strategy, cash=100_000, commission=.002, trade_on_close=False)
    # Получаем статистику бэктестинга
    stats = bt.run(size=1000)
    # Печатаем статистику
    print(stats)
    # Выводим графиг бэктестинга (копия сохраняется в корень с именем "Название стратегии.html"
    bt.plot(plot_volume=True, relative_equity=True)
    print('===' * 30, '\nBacktesting done by: ', time.time() - start, '\n====' * 30, '\n')

path_filename ="../source_ds/1m/ETHUSDT-1m.csv"
dataset = MarkedDataSet(path_filename)
tr = TrainNN(dataset)
tr.epochs = 10
tr.tsg_batch_size = 128
tr.tsg_window_length = 40
tr.train()
test_df = tr.backtest_test_dataset()