import time

from analyze import DataLoad
import os
from backtester import Back
from backtester.strategies.old_fashion import *
from oldangrybirdtools import BeachBirdSeriesGenerator, get_angry_bird_model, get_old_model

from maketarget import BigFatMommyMakesTargetMarkers
from tensorflow.keras.optimizers import Adam

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
                        end_period='2021-01-31 23:59:59'
                        )

    print(database.pairs_symbols)

    window_size = {'ETHBTC': 41, 'ETHUSDT': 123}
    strategy = {'ETHBTC': LongStrategy, 'ETHUSDT': LongShortStrategy}


    start = time.time()
    sample_x = 600
    frame_size = 41

    dataset_test_size = 6000
    dataset_val_size = 3600
    val_test_gap = 1
    train_val_gap = 10


    for item in database.pairs_symbols:
        print('===' * 30)
        print(item)
        df = database.get_pair(item, intervals[0])
        df = BigFatMommyMakesTargetMarkers(window_size=window_size[item]).mark_y(df)
        # df.dropna(axis=0, inplace=True)
        ds = BeachBirdSeriesGenerator(df, batch_size=512, sample_x=sample_x)
        ds.test_start_index = ds.test_end_index - dataset_test_size
        ds.val_end_index = ds.test_start_index - val_test_gap
        ds.val_start_index = ds.val_end_index - dataset_val_size
        ds.train_end_index = ds.val_start_index - train_val_gap
        model = get_old_model(ds.input_shape)
        model.summary()
        model.compile(optimizer=Adam(learning_rate=5e-06),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

        history = model.fit(ds.train(), epochs=10,  validation_data=ds.val(), verbose=1)

        pred = model.predict(ds.test())
        ddd = ds.test_prep_dec(pred)
        bt = Back(ddd, strategy[item], cash=100_000, commission=.002, trade_on_close=False)
        stats = bt.run(size=1000)
        print(stats)
        bt.plot(plot_volume=True, relative_equity=True)

