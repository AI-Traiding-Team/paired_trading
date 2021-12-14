import time

from analyze import DataLoad
import os
from backtester import Back
from backtester.strategies.old_fashion import *
from oldangrybirdtools import BeachBirdSeriesGenerator, get_angry_bird_model, get_old_model

from maketarget import BigFatMommyMakesTargetMarkers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    intervals = ['1m']
    root_path = os.getcwd()
    source_root_path = os.path.join(root_path, 'source_root')

    database = DataLoad(pairs_symbols=['ETHUSDT'],
                        time_intervals=intervals,
                        source_directory=source_root_path,
                        start_period='2021-09-01 00:00:00',
                        end_period='2021-10-30 23:59:59'
                        )

    print(database.pairs_symbols)

    # window_size = {'ETHBTC': 41, 'ETHUSDT': 123}
    window_size = {'ETHBTC': 41, 'ETHUSDT': 123}
    strategy = {'ETHBTC': LongStrategy, 'ETHUSDT': LongShortStrategy}
    # strategy = {'ETHBTC': LongStrategy, 'ETHUSDT': LongStrategy}


    start = time.time()
    sample_x = 60
    frame_size = 41

    dataset_test_size = 12000
    dataset_val_size = 6000
    val_test_gap = 1
    train_val_gap = 10


    for item in database.pairs_symbols:
        print('===' * 30)
        print(item)
        df = database.get_pair(item, intervals[0])
        df = BigFatMommyMakesTargetMarkers(window_size=window_size[item]).mark_y(df)
        # df.dropna(axis=0, inplace=True)
        ds = BeachBirdSeriesGenerator(df, batch_size=256, sample_x=sample_x)
        ds.test_start_index = ds.test_end_index - dataset_test_size
        ds.val_end_index = ds.test_start_index - val_test_gap
        ds.val_start_index = ds.val_end_index - dataset_val_size
        ds.train_end_index = ds.val_start_index - train_val_gap
        model = get_old_model(ds.input_shape)
        optimizer_Adam = Adam(learning_rate=5e-06)
        optimizer_SGD = SGD(learning_rate=03e-03,
                            nesterov=True,
                            momentum=0.89
                            )

        model.compile(optimizer=optimizer_Adam,
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

        history = model.fit(ds.train(),
                            epochs=10,
                            validation_data=ds.val(),
                            verbose=1)

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
