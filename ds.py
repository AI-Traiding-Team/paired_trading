import time

from analyze import DataLoad
import os
from backtester import Back
# from backtester.strategies import *
from backtester.strategies.old_fashion import *
# import backtrader as bt
# from optimizer import Objective
from oldangrybirdtools import BeachBirdSeriesGenerator, get_angry_bird_model, get_old_model

# import optuna

from maketarget import BigFatMommyMakesTargetMarkers

# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, Adamax
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras import utils
#
# from oldangrybirdtools import Super_Dooper, Dataset

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    intervals = ['1m']
    root_path = os.getcwd()
    source_root_path = os.path.join(root_path, 'source_root')

    database = DataLoad(pairs_symbols=['ETHBTC'],
                        time_intervals=intervals,
                        source_directory=source_root_path,
                        start_period='2021-01-01 00:00:00',
                        end_period='2021-01-15 23:59:59'
                        )

    print(database.pairs_symbols)

    window_size = {'ETHBTC': 41, 'ETHUSDT': 123}
    strategy = {'ETHBTC': LongStrategy, 'ETHUSDT': LongShortStrategy}


    start = time.time()
    sample_x = 41
    frame_size = 41
    ensemble = 40
    batch_size = 100
    period = 'hour'

    dataset_test_size = 2400
    dataset_val_size = 2400
    val_test_gap = 1000
    train_val_gap = 10


    for item in database.pairs_symbols:
        print('===' * 30)
        print(item)
        df = database.get_pair(item, intervals[0])
        df = BigFatMommyMakesTargetMarkers(window_size=window_size[item]).mark_y(df)
        df.dropna(axis=0, inplace=True)
        print(df.shape)
        print(df.head(8))
        # ds = Dataset(df, batch_size, ensemble, drop_signal=True, date_to_feat=True)
        ds = BeachBirdSeriesGenerator(df, batch_size=1000, sample_x=sample_x)
        ds.test_start_index = ds.test_end_index - dataset_test_size
        ds.val_end_index = ds.test_start_index - val_test_gap
        ds.val_start_index = ds.val_end_index - dataset_val_size
        ds.train_end_index = ds.val_start_index - train_val_gap
        model = get_old_model(ds.input_shape)
        model.compile(optimizer=Adam(learning_rate=5e-05),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

        history = model.fit(ds.train(), epochs=3, verbose=1)

        pred = model.predict(ds.test())
        ddd = ds.test_prep_dec(pred)
        bt = Back(ddd, strategy[item], cash=100_000, commission=.002, trade_on_close=False)
        stats = bt.run(size=1000)
        print(stats)
        bt.plot(plot_volume=True, relative_equity=True)
        # df = BigFatMommyMakesTargetMarkers(window_size=window_size[item]).mark_y(df)
        # ds = BeachBirdSeriesGenerator(df, 5, 10)
        # ds.test_start_index = ds.test_end_index - dataset_test_size
        # ds.val_end_index = ds.test_start_index - val_test_gap
        # ds.val_start_index = ds.val_end_index - dataset_val_size
        # ds.train_end_index = ds.val_start_index - train_val_gap
        # model = get_angry_bird_model(ds.input_shape)
        # print(ds.input_shape)
        # print(model.summary())
        # model.compile(optimizer=Adam(learning_rate=5e-05),
        #               loss='categorical_crossentropy',
        #               metrics=['categorical_accuracy'])
        # history = model.fit(ds.train(), epochs=5, verbose=1)

        # early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=12, verbose=0,
        #                                restore_best_weights=True, mode='max')
        #
        # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, min_lr=1e-07, verbose=0)
        #
        # model_file_name = 'model.h5'
        # checkpoint = ModelCheckpoint(model_file_name, monitor='val_categorical_accuracy', verbose=0,
        #                              save_best_only=True, mode='max')
        #
        # # fit model
        # history = model.fit(ds.train(), epochs=5, verbose=1,
        #                     validation_data=ds.val(),
        #                     # batch_size = batch_size,
        #                     callbacks=[early_stopping, reduce_lr, checkpoint])
