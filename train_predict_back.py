import os
import time
from analyze import DataLoad
from marker.marker import Marker
from powerrun.customlosses import *
from backtester import Back
from backtester.strategies import *
from backtester.strategies.old_fashion import *
import tensorflow as tf

import backtrader as bt
from optimizer import Objective

import optuna
from powerrun.functions import TrainNN, MarkedDataSet
from powerrun.unets import *
from powerrun.experimental import *
# from powerrun.models import get_resnet1d_model, get_angry_bird_model
from maketarget import BigFatMommyMakesTargetMarkers

import warnings
warnings.filterwarnings('ignore')

__version__ = 0.0010


if __name__ == '__main__':
    # intervals = ['1m']
    root_path = os.getcwd()
    source_root_path = os.path.join(root_path, 'source_root')
    pair_symbol = 'ETHUSDT'
    timeframe = '1m'
    loaded_crypto_data = DataLoad(pairs_symbols=[pair_symbol],
                                  time_intervals=[timeframe],
                                  source_directory=source_root_path,
                                  start_period='2021-11-01 00:00:00',
                                  end_period='2021-12-01 23:59:59',
                                  # start_period='2021-09-22 00:00:00',
                                  # end_period='2021-11-30 23:59:59',
                                  # start_period='2021-09-01 00:00:00',
                                  # end_period='2021-11-30 23:59:59',
                                  )

    mr = Marker(loaded_crypto_data)
    power_trend = 0.0275
    prepare_dataset_method = 4
    mr.mark_all_loader_df(target_directory=f"source_ds{prepare_dataset_method}",
                          signal_method=prepare_dataset_method,
                          weight=power_trend
                          )

    path_filename = f"source_ds{prepare_dataset_method}/{timeframe}/{pair_symbol}-{timeframe}.csv"

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

    dataset.tsg_window_length = 180
    dataset.tsg_batch_size = 240
    dataset.tsg_overlap = 60
    dataset.gap_timeframes = dataset.tsg_overlap + 20
    dataset.prepare_data()
    # Добавляем разметку Signal (значения [1, -1])
    tr = TrainNN(dataset, power_trend=power_trend)

    """ train UNET start """
    tr.monitor = "val_loss"
    # tr.dice_metric = DiceCoefficient()
    # tr.dice_cce_loss = DiceCCELoss()
    # tr.loss = tr.dice_cce_loss
    # tr.metric = tr.dice_metric
    # if 'dice' in str(tr.loss):
    #     tr.monitor = f'val_{tr.loss}'
    tr.loss = "categorical_crossentropy"
    tr.metric = "categorical_accuracy"

    kernels = 5
    filters = 30
    tr.net_name = f"unet1d_tanh_fltrs_{filters}_krnls_{kernels}_{tr.monitor}"
    tr.keras_model = get_unet1d_tanh(input_shape=tr.mrk_dataset.input_shape,
                                     num_classes=2,
                                     filters=filters,
                                     kernels=kernels
                                     )
    """ train UNET end  """

    # """ train LSTM combo start """
    # tr.monitor = "val_loss"
    # # tr.dice_metric = DiceCoefficient()
    # # tr.dice_cce_loss = DiceCCELoss()
    # # tr.loss = tr.dice_cce_loss
    # # tr.metric = tr.dice_metric
    # # if 'dice' in str(tr.loss):
    # #     tr.monitor = f'val_{tr.loss}'
    # tr.loss = "categorical_crossentropy"
    # tr.metric = "categorical_accuracy"
    #
    # kernels_len = tr.mrk_dataset.input_shape[1]
    # filters = tr.mrk_dataset.input_shape[1]
    # tr.net_name = f"3way_lstm_moded_{tr.monitor}_fltrs_{filters}_krnlslen_{kernels_len}"
    # tr.keras_model = get_3way_lstm_moded(input_shape=tr.mrk_dataset.input_shape,
    #                                      kernels_len=kernels_len,
    #                                      filters=filters,
    #                                      num_classes=2,
    #                                      )
    # """ train LSTM combo end  """

    tr.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    # tr.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, nesterov=True, momentum=0.9)

    tr.epochs = 45
    # tr.train()
    # tr.figshow_base()
    tr.evaluate()
    tr.show_trend_predict(show_data="train")
    tr.show_trend_predict(show_data="val")
    tr.show_trend_predict(show_data="test")

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

