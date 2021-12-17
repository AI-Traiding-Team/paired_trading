import time

from analyze import DataLoad
from marker.marker import Marker
import os
from backtester import Back
from backtester.strategies import *
from backtester.strategies.old_fashion import *
import backtrader as bt
from optimizer import Objective

import optuna
from powerrun.functions2 import TrainNN, MarkedDataSet
from powerrun.models import get_resnet1d_model, get_angry_bird_model
from maketarget import BigFatMommyMakesTargetMarkers

import warnings
warnings.filterwarnings('ignore')

__version__ = 0.0004


if __name__ == '__main__':
    intervals = ['1m']
    root_path = os.getcwd()
    source_root_path = os.path.join(root_path, 'source_root')

    loaded_crypto_data = DataLoad(pairs_symbols=None,
                                  time_intervals=['1m'],
                                  source_directory=source_root_path,
                                  start_period='2021-08-01 00:00:00',
                                  end_period='2021-09-30 23:59:59',
                                  )
    mr = Marker(loaded_crypto_data)
    mr.mark_all_loader_df(target_directory="source_ds2", signal_method=2,  weight=0.055)

    path_filename = "source_ds2/1m/ETHUSDT-1m.csv"

    # print(database.pairs_symbols)

    window_size = 82
    strategy = LongStrategy

    start = time.time()
    # for item in database.pairs_symbols:
    print('===' * 30)
    # Загружаем исходные данные OHLCV
    dataset = MarkedDataSet(path_filename, df, df_priority=False)
    # Добавляем разметку Signal (значения [1, -1])
    tr = TrainNN(dataset)
    tr.tsg_window_length = 82
    tr.epochs = 75
    tr.train()
    df = tr.backtest_test_dataset()
    # Запускаем бэтестинг, на вход подаем DataFrame [['Open','Close','High','Low','Volume','Signal]]
    bt = Back(df, strategy, cash=100_000, commission=.002, trade_on_close=False)
    # Получаем статистику бэктестинга
    stats = bt.run(size=1000)
    # Печатаем статистику
    print(stats)
    # Выводим графиг бэктестинга (копия сохраняется в корень с именем "Название стратегии.html"
    bt.plot(plot_volume=True, relative_equity=True)
    print('===' * 30, '\nBacktesting done by: ', time.time() - start, '\n', '====' * 30, '\n')

