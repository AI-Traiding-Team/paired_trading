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
from powerrun.functions import TrainNN, MarkedDataSet
from powerrun.models import get_resnet1d_model, get_angry_bird_model
from maketarget import BigFatMommyMakesTargetMarkers

import warnings
warnings.filterwarnings('ignore')

__version__ = 0.0005


if __name__ == '__main__':
    intervals = ['1m']
    root_path = os.getcwd()
    source_root_path = os.path.join(root_path, 'source_root')

    loaded_crypto_data = DataLoad(pairs_symbols=["ETHUSDT"],
                                  time_intervals=['1m'],
                                  source_directory=source_root_path,
                                  start_period='2021-07-22 00:00:00',
                                  end_period='2021-09-01 23:59:59',
                                  )

    mr = Marker(loaded_crypto_data)
    power_trend = 0.0275
    prepare_dataset_method = 0
    mr.mark_all_loader_df(target_directory=f"source_ds{prepare_dataset_method}",
                          signal_method=prepare_dataset_method,
                          weight=power_trend
                          )

    path_filename = f"source_ds{prepare_dataset_method}/1m/ETHUSDT-1m.csv"

    strategy = LongShortStrategy
    start = time.time()
    # for item in database.pairs_symbols:
    print('===' * 30)
    # Загружаем исходные данные OHLCV
    dataset = MarkedDataSet(path_filename,
                            df,
                            df_priority=False,
                            verbose=False
                            )

    dataset.tsg_window_length = 330
    dataset.tsg_batch_size = 100
    dataset.tsg_overlap = 30
    dataset.gap_timeframes = 70
    dataset.prepare_data()
    # Добавляем разметку Signal (значения [1, -1])
    tr = TrainNN(dataset)
    tr.power_trend = power_trend
    tr.epochs = 20
    # tr.train()
    # tr.figshow_base()
    tr.get_predict()
    tr.show_trend_predict()
    df = tr.backtest_test_dataset()

    window_size = dataset.tsg_window_length
    # Запускаем бэктестинг, на вход подаем DataFrame [['Open','Close','High','Low','Volume','Signal]]
    bt = Back(df, strategy, cash=100_000, commission=0, trade_on_close=False)
    # Получаем статистику бэктестинга
    stats = bt.run(size=1000)
    # Печатаем статистику
    print(stats)
    # Выводим графиг бэктестинга (копия сохраняется в корень с именем "Название стратегии.html"
    bt.plot(plot_volume=True, relative_equity=True)
    print('===' * 30, '\nBacktesting done by:', f'{time.time() - start}', f'\n{"===" * 30}\n')

