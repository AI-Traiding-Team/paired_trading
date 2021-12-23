import os
import time
import random
import numpy as np
import pandas as pd

from networks import *
# from powerrun.models import *
from datamodeling import *
from powerrun.unets import get_unet1d_new
from powerrun.customlosses import *

# from oldangrybirdtools import BeachBirdSeriesGenerator, get_angry_bird_model, get_old_model

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

__version__ = 0.0025


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

import warnings
warnings.filterwarnings("ignore")


def dataset_split_show(close_series, train_df_start_end, val_df_start_end, test_df_start_end, symbol):
    _temp_1 = close_series
    _temp_2 = _temp_1.copy()
    _temp_3 = _temp_1.copy()
    _temp_1[train_df_start_end[1]:] = 0
    _temp_2[:val_df_start_end[0]] = 0
    _temp_2[val_df_start_end[1]:] = 0
    _temp_3[:test_df_start_end[0]] = 0

    plt.figure(figsize=(12, 4))
    ax0 = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    # df['Close'].plot(ax = ax0, label='all')
    _temp_1.plot(ax=ax0, label='train')
    _temp_2.plot(ax=ax0, label='val')
    _temp_3.plot(ax=ax0, label='test')
    plt.title(f'График изменений цены на {symbol}')
    plt.legend()
    plt.grid()
    plt.show()
    pass


class MarkedDataSet:
    def __init__(self, path_filename, all_data_df, df_priority=False, verbose=False):
        self.path_filename = path_filename
        self.timeframe = None
        self.symbol = None
        self.verbose = verbose
        self.tsg_window_length = 40
        self.tsg_sampling_rate = 1
        self.tsg_stride = 1
        self.tsg_start_index = 0
        self.tsg_overlap = 0
        self.tsg_batch_size = 128
        self.gap_timeframes = 10
        self.train_gen = object
        self.test_gen = object
        self.val_gen = object
        self.train_size = 0.6
        self.val_size = 0.2

        if df_priority:
            self.all_data_df = all_data_df
        else:
            self.all_data_df = None
            self.all_data_df = pd.read_csv(self.path_filename,
                                           index_col="datetimeindex")
            self.all_data_df.index = pd.to_datetime(self.all_data_df.index)
        self.features_df = None
        self.y_df = None
        self.train_df_len = None
        self.val_df_len = None
        self.df_test_len = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.features_scaler = None
        self.targets_scaler = None
        self.ohe = None
        self.train_df_start_end: list = [0, 0]
        self.val_df_start_end: list = [0, 0]
        self.test_df_start_end: list = [0, 0]
        self.x_Train = None
        self.y_Train = None
        self.x_Val = None
        self.y_Val = None
        self.y_Test = None
        self.x_Test = None
        self.input_shape = None
        self.x_test_df_backrade = None

    def get_train_generator(self, x_Train_data, y_Train_data):
        self.train_gen = TSDataGenerator(data=x_Train_data,
                                         targets=y_Train_data,
                                         length=self.tsg_window_length,
                                         sampling_rate=self.tsg_sampling_rate,
                                         stride=self.tsg_stride,
                                         start_index=self.tsg_start_index,
                                         overlap=self.tsg_overlap,
                                         batch_size=self.tsg_batch_size,
                                         )
        return self.train_gen

    def get_val_generator(self, x_Val_data, y_Val_data):
        self.val_gen = TSDataGenerator(data=x_Val_data,
                                       targets=y_Val_data,
                                       length=self.tsg_window_length,
                                       sampling_rate=self.tsg_sampling_rate,
                                       stride=self.tsg_stride,
                                       start_index=self.tsg_start_index,
                                       overlap=self.tsg_overlap,
                                       batch_size=self.tsg_batch_size,
                                       )
        return self.val_gen

    def get_test_generator(self, x_Test_data, y_Test_data):
        self.test_gen = TSDataGenerator(data=x_Test_data,
                                        targets=y_Test_data,
                                        length=self.tsg_window_length,
                                        sampling_rate=self.tsg_sampling_rate,
                                        stride=self.tsg_stride,
                                        start_index=self.tsg_start_index,
                                        overlap=self.tsg_overlap,
                                        batch_size=self.tsg_batch_size,
                                        )
        return self.test_gen

    def prepare_data(self):
        print("\nAll dataframe data example (Signal markup with treshhold 0.0275):")
        print(self.all_data_df.head().to_string())
        self.features_df = self.all_data_df.iloc[:, :-1]
        print("\nX (features) dataframe data example:")
        print(self.features_df.head().to_string(), f"\n")
        self.y_df = self.all_data_df.iloc[:, -1:]
        print("\nSignal (true) dataframe data example:")
        print(self.y_df.head().to_string(), f"\n")
        uniques, counts = np.unique(self.y_df.values, return_counts=True)
        for unq, cnt in zip(uniques, counts):
            print("Total:", unq, cnt)

        self.calculate_split_df()
        msg = f"\nSplit dataframe:" \
              f"Train start-end: {self.train_df_start_end[0]} : {self.train_df_start_end[1]}\n" \
              f"Train length: {self.train_df_start_end[1] - self.train_df_start_end[0]}\n" \
              f"Validation start-end: {self.val_df_start_end[0]} : {self.val_df_start_end[1]}\n" \
              f"Validation length: {self.val_df_start_end[1] - self.val_df_start_end[0]}\n" \
              f"Test start-end: {self.test_df_start_end[0]} : {self.test_df_start_end[1]}\n" \
              f"Test length: {self.test_df_start_end[1] - self.test_df_start_end[0]}"
        print(f"{msg}\n")
        self.split_data_df()
        _temp_1 = pd.DataFrame()
        _temp_1 = self.all_data_df["close"].copy()
        self.symbol = self.path_filename.split("-")[0].split("/")[-1]
        self.timeframe = self.path_filename.split("-")[1].split(".")[0]

        if self.verbose:
            dataset_split_show(_temp_1,
                               self.train_df_start_end,
                               self.val_df_start_end,
                               self.test_df_start_end,
                               f"{self.symbol}-{self.timeframe}"
                               )
        datetime_ohe_flag = False
        de_columns_names = list()
        main_columns_names = list()
        for column_name in self.features_df.columns:
            if column_name.startswith("de_"):
                de_columns_names.append(column_name)
                datetime_ohe_flag = True
            else:
                main_columns_names.append(column_name)
        if datetime_ohe_flag:
            de_df = self.features_df[[col_name for col_name in de_columns_names]].copy()
            main_df = self.features_df[[col_name for col_name in main_columns_names]].copy()
            # """ Warning! Now is MinMaxScaler for this dataset """
            # self.features_scaler = MinMaxScaler()
            # self.features_scaler.fit(main_df.values)
            x_train_df = main_df.iloc[self.train_df_start_end[0]: self.train_df_start_end[1], :]
            self.features_scaler = RobustScaler().fit(x_train_df.values)
            temp_arr = de_df.values
            x_arr = self.features_scaler.transform(main_df.values)
            x_arr = np.concatenate((temp_arr, x_arr), axis=1)
        else:
            x_train_df = self.features_df.iloc[self.train_df_start_end[0]: self.train_df_start_end[1], :]
            # x_train_values = x_train_df.values
            # bad_indices = np.where(np.isinf(x_train_values))
            self.features_scaler = RobustScaler().fit(x_train_df.values)
            x_arr = self.features_scaler.transform(self.features_df.values)
        print("\nCreate arrays with X (features)", x_arr.shape)
        y_arr = pd.get_dummies(self.y_df["Signal"], dtype=float).values
        # y_arr = self.y_df.values.reshape(-1, 1)
        print("\nCreate arrays with Signal (True)", y_arr.shape)
        self.prepare_datagens(x_arr, y_arr)
        pass

    def prepare_datagens(self, x_arr, y_arr):
        if self.test_df_start_end == [0, 0]:
            x_Train_data = x_arr[self.train_df_start_end[0]:self.train_df_start_end[1], :]
            x_Val_data = x_arr[self.val_df_start_end[0]:self.val_df_start_end[1], :]
            y_Train_data = y_arr[self.train_df_start_end[0]:self.train_df_start_end[1], :]
            y_Val_data = y_arr[self.val_df_start_end[0]:self.val_df_start_end[1], :]
        else:
            x_Train_data = x_arr[self.train_df_start_end[0]:self.train_df_start_end[1], :]
            x_Val_data = x_arr[self.val_df_start_end[0]:self.val_df_start_end[1], :]
            x_Test_data = x_arr[self.test_df_start_end[0]:self.test_df_start_end[1], :]
            y_Train_data = y_arr[self.train_df_start_end[0]:self.train_df_start_end[1], :]
            y_Val_data = y_arr[self.val_df_start_end[0]:self.val_df_start_end[1], :]
            y_Test_data = y_arr[self.test_df_start_end[0]:self.test_df_start_end[1], :]
            self.test_gen = self.get_test_generator(x_Test_data, y_Test_data)
            """ Using generator 1 time to have solid data """
            self.x_Test, self.y_Test = self.create_data_from_gen(x_Test_data, y_Test_data)
            # x_Test_gen = self.get_test_generator(x_Test_data, y_Test_data)

        msg = f"Created arrays: \nx_Train_data = {x_Train_data.shape}, y_Train_data = {y_Train_data.shape}\n" \
              f"x_Val_data = {x_Val_data.shape}, y_Val_data = {y_Val_data.shape}\n" \
              f"x_Test_data = {x_Test_data.shape}, y_Test_data = {y_Test_data.shape}\n"
        print(f"{msg}\n")
        """" Using generator 1 time to get solid data arrays"""

        self.train_gen = self.get_train_generator(x_Train_data, y_Train_data)
        self.val_gen = self.get_val_generator(x_Val_data, y_Val_data)
        self.x_Train, self.y_Train = self.create_data_from_gen(x_Train_data, y_Train_data)
        self.x_Val, self.y_Val = self.create_data_from_gen(x_Val_data, y_Val_data)
        self.input_shape = self.val_gen.sample_shape
        pass

    def create_data_from_gen(self, x_arr, y_arr):
        gen = TSDataGenerator(data=x_arr,
                              targets=y_arr,
                              length=self.tsg_window_length,
                              sampling_rate=self.tsg_sampling_rate,
                              stride=self.tsg_stride,
                              start_index=self.tsg_start_index,
                              overlap=self.tsg_overlap,
                              batch_size=x_arr.shape[0]
                              )
        for x_data, y_data in gen:
            continue
        return x_data, y_data

    def calculate_split_df(self):
        df_rows = self.features_df.shape[0]
        self.train_df_len = int(df_rows * self.train_size) - self.tsg_start_index
        self.train_df_start_end[0] = self.tsg_start_index
        self.train_df_start_end[1] = self.tsg_start_index + (
                    self.train_df_len // self.tsg_window_length) * self.tsg_window_length
        if self.train_size + self.val_size == 1.0:
            self.val_df_start_end[0] = self.train_df_start_end[1] + self.gap_timeframes
            self.val_df_start_end[1] = self.val_df_start_end[0] + (
                        (df_rows - self.val_df_start_end[0]) // self.tsg_window_length) * self.tsg_window_length
        else:
            self.val_df_len = int(df_rows * self.val_size)
            self.val_df_start_end[0] = self.train_df_start_end[1] + self.gap_timeframes
            self.val_df_start_end[1] = self.val_df_start_end[0] + (
                        self.val_df_len // self.tsg_window_length) * self.tsg_window_length
            self.test_df_start_end[0] = self.val_df_start_end[1] + self.gap_timeframes
            self.test_df_start_end[1] = self.test_df_start_end[0]+(((df_rows - (self.test_df_start_end[
                                                         0] + self.gap_timeframes)) // self.tsg_window_length)-1) * self.tsg_window_length
        pass

    def split_data_df(self):
        self.train_df = self.all_data_df.iloc[self.train_df_start_end[0]: self.train_df_start_end[1], :]
        if self.train_size + self.val_size >= 1.0:
            self.val_df = self.all_data_df.iloc[self.val_df_start_end[0]: self.val_df_start_end[1], :]
        else:
            self.val_df = self.all_data_df.iloc[self.val_df_start_end[0]: self.val_df_start_end[1], :]
            self.test_df = self.all_data_df.iloc[self.test_df_start_end[0]: self.test_df_start_end[1], :]
        pass


class TrainNN:
    def __init__(self,
                 mrk_dataset: MarkedDataSet,
                 power_trend=0.0275
                 ):
        self.dice_cce_loss = None
        self.dice_metric = None
        self.mrk_dataset = mrk_dataset
        self.y_Pred = None
        self.power_trends_list = [0.15, 0.075, 0.055, 0.0275]
        self.power_trend = power_trend
        if self.power_trend not in self.power_trends_list:
            self.power_trends_list.append(self.power_trend)
        self.experiment_name = f"{self.mrk_dataset.symbol}-{self.mrk_dataset.timeframe}"
        self.symbol = self.mrk_dataset.symbol
        self.timeframe = self.mrk_dataset.timeframe

        self.dice_metric = DiceCoefficient()
        self.dice_cce_loss = DiceCCELoss()
        self.history = None
        self.epochs = 15

        """ Use it only if not using TimeSeries Generator"""
        self.batch_size = None
        self.monitor = "val_loss"
        self.loss = "categorical_crossentropy"
        self.metric = "categorical_accuracy"
        self.path_filename: str = ''
        self.model_compiled = False

        self.net_name = "unet1d_new"
        self.keras_model = get_unet1d_new(input_shape=self.mrk_dataset.input_shape, num_classes=2, filters=64)
        # tr.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, nesterov=True, momentum=0.9)
        pass

    def compile(self):
        self.path_filename = os.path.join('outputs', f"{self.experiment_name}_{self.net_name}_NN.png")
        self.keras_model.summary()
        self.keras_model.compile(optimizer=self.optimizer,
                                 loss=self.loss,
                                 metrics=[self.metric],
                                 )
        self.model_compiled = True
        pass

    def train(self):
        if not self.model_compiled:
            self.compile()

        chkp = ModelCheckpoint(os.path.join("outputs", f"{self.experiment_name}_{self.net_name}_{self.power_trend}.h5"),
                               mode='min',
                               monitor=self.monitor,
                               save_best_only=True,
                               )
        rlrs = ReduceLROnPlateau(monitor=self.monitor, factor=0.2, patience=20, min_lr=0.000001)
        es = EarlyStopping(patience=18, restore_best_weights=True)
        callbacks = [rlrs, chkp, es]
        path_filename = os.path.join('outputs', f"{self.experiment_name}_{self.net_name}_{self.power_trend}_NN.png")
        tf.keras.utils.plot_model(self.keras_model,
                                  to_file=path_filename,
                                  show_shapes=True,
                                  show_layer_names=True,
                                  expand_nested=True,
                                  dpi=96,
                                  )
        self.history = self.keras_model.fit(self.mrk_dataset.train_gen,
                                            validation_data=self.mrk_dataset.train_gen,
                                            epochs=self.epochs,
                                            verbose=1,
                                            callbacks=callbacks,
                                            )

    def load_best_weights(self):
        path_filename = os.path.join('outputs', f"{self.experiment_name}_{self.net_name}_{self.power_trend}.h5")
        self.keras_model.load_weights(path_filename)
        pass

    def get_predict(self, x_Data):
        if not self.model_compiled:
            self.compile()
        self.load_best_weights()
        self.y_Pred = self.keras_model.predict(x_Data)
        return self.y_Pred

    def evaluate(self):
        if not self.model_compiled:
            self.compile()
        self.load_best_weights()
        self.keras_model.evaluate(self.mrk_dataset.x_Test, self.mrk_dataset.y_Test)
        pass

    def figshow_base(self):
        sub_plots = 1
        if self.monitor != "val_loss":
            sub_plots = 2
        fig = plt.figure(figsize=(24, 10*sub_plots))
        sns.set_style("white")
        ax1 = fig.add_subplot(1, sub_plots, 1)
        ax1.set_axisbelow(True)
        ax1.minorticks_on()
        # Turn on the minor TICKS, which are required for the minor GRID
        # Customize the major grid
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        # Customize the minor grid
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        N = np.arange(0, len(self.history.history["loss"]))
        plt.plot(N, self.history.history["loss"], label="loss")
        if 'val_loss' in self.history.history:
            plt.plot(N, self.history.history["val_loss"], label="val_loss")
        if 'dice_coef' in self.history.history:
            plt.plot(N, self.history.history["dice_coef"], label="dice_coef")
        if 'val_dice_coef' in self.history.history:
            plt.plot(N, self.history.history["val_dice_coef"], label="val_dice_coef")
        if 'mae' in self.history.history:
            plt.plot(N, self.history.history["mae"], label="mae")
        if 'accuracy' in self.history.history:
            plt.plot(N, self.history.history["accuracy"], label="accuracy")
        if 'val_accuracy' in self.history.history:
            plt.plot(N, self.history.history["val_accuracy"], label="val_accuracy")
        if 'categorical_accuracy' in self.history.history:
            plt.plot(N, self.history.history["categorical_accuracy"], label="categorical_accuracy")
        if 'val_categorical_accuracy' in self.history.history:
            plt.plot(N, self.history.history["val_categorical_accuracy"], label="val_categorical_accuracy")
        plt.title(f"Training Loss and Metric")
        plt.legend()
        if sub_plots == 2:
            ax2 = fig.add_subplot(1, sub_plots, sub_plots)
            ax2.set_axisbelow(True)
            ax2.minorticks_on()
            # Turn on the minor TICKS, which are required for the minor GRID
            # Customize the major grid
            ax2.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
            # Customize the minor grid
            ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
            if 'dice_cce_loss' in self.history.history:
                plt.plot(N, self.history.history["dice_cce_loss"], label="dice_cce_loss")
            if 'val_dice_cce_loss' in self.history.history:
                plt.plot(N, self.history.history["val_dice_cce_loss"], label="val_dice_cce_loss")
            plt.legend()

        path_filename = os.path.join('outputs', f"{self.experiment_name}_{self.net_name}_{self.power_trend}_learning.png")
        plt.savefig(path_filename, dpi=96, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None
                    )
        plt.show()
        pass

    def backtest_test_dataset(self):
        main_columns_names = ["open", "high", "low", "close", "volume", "Signal"]
        print("Creating backtesting set")
        data_df = self.mrk_dataset.features_df[
                  self.mrk_dataset.test_df_start_end[0]: self.mrk_dataset.test_df_start_end[
                                                             1] - self.mrk_dataset.tsg_window_length - self.mrk_dataset.tsg_overlap]

        trend_pred = self.get_predict(self.mrk_dataset.x_Test)
        # data_df['trend'] = trend_pred.flatten()
        data_df['trend'] = np.argmax(trend_pred, axis=1)
        data_df.loc[data_df["trend"] == 0, "trend"] = -1.0
        # data_df.loc[data_df["trend"] == 1, "trend"] = 1.0
        data_df["Signal"] = data_df['trend']
        # self.mrk_dataset.test_df_backtrade = data_df.copy()

        data_df = data_df[[col_name for col_name in main_columns_names]].copy()

        data_df.columns = [item.lower().capitalize() for item in data_df.columns]

        print("\nSignal (pred) dataframe data example for backtesting:")
        print(data_df["Signal"].head().to_string(), f"\n")
        uniques, counts = np.unique(data_df["Signal"].values, return_counts=True)
        for unq, cnt in zip(uniques, counts):
            print(f'Signal: {unq}, total {cnt}')
        print()
        self.mrk_dataset.test_df_backtrade = data_df
        return data_df

    def show_trend_predict(self, show_data='test'):
        print(f"\nВизуализируем результат")
        if show_data == 'test':
            start_end = self.mrk_dataset.test_df_start_end
            x_Data = self.mrk_dataset.x_Test
        elif show_data == 'train':
            start_end = self.mrk_dataset.train_df_start_end
            x_Data = self.mrk_dataset.x_Train
        elif show_data == 'val':
            start_end = self.mrk_dataset.val_df_start_end
            x_Data = self.mrk_dataset.x_Val
        else:
            sys.exit(f"Error! wrong option in show_data {show_data}")
        print(f"Show True and prediction of the *** {show_data} ***")

        # data_df = self.mrk_dataset.features_df[start_end[0]: start_end[1] - self.mrk_dataset.tsg_window_length - self.mrk_dataset.tsg_overlap]
        # y_df = self.mrk_dataset.y_df[start_end[0]: start_end[1] - self.mrk_dataset.tsg_window_length - self.mrk_dataset.tsg_overlap]
        data_df = self.mrk_dataset.features_df[start_end[0]: start_end[1]]
        y_df = self.mrk_dataset.y_df[start_end[0]: start_end[1]]

        max_close = data_df["close"].max()
        min_close = data_df["close"].min()
        mean_close = data_df["close"].mean()

        trend_pred = self.get_predict(x_Data)
        trend_pred = np.argmax(trend_pred,  axis=1)
        zeros_pad = np.zeros([self.mrk_dataset.tsg_window_length + self.mrk_dataset.tsg_overlap])
        zeros_pad[:] = mean_close
        new_trend = np.hstack([zeros_pad, trend_pred])

        trend_pred_df = pd.DataFrame(data=new_trend, columns=["trend"])

        y_df.loc[(y_df["Signal"] == 1), "Signal"] = max_close
        y_df.loc[(y_df["Signal"] == 0), "Signal"] = min_close
        trend_pred_df.loc[(trend_pred_df["trend"] == 1), "trend"] = max_close
        trend_pred_df.loc[(trend_pred_df["trend"] == 0), "trend"] = min_close
        data_df[f"trend_{self.power_trend}"] = y_df["Signal"]

        fig = plt.figure(figsize=(20, 12))
        ax1 = fig.add_subplot(2, 1,  1)
        ax1.set_axisbelow(True)
        ax1.minorticks_on()
        # Turn on the minor TICKS, which are required for the minor GRID
        # Customize the major grid
        ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        # Customize the minor grid
        ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        ax1.plot(
                 data_df.index, data_df[f"trend_{self.power_trend}"],
                 data_df.index, data_df["close"],
                 )
        ax1.set_ylabel(f'True, power = {self.power_trend}', color='r')
        plt.title(f"Trend with power: {self.power_trend}")
        ax2 = fig.add_subplot(2, 1,  2)
        ax2.set_axisbelow(True)
        ax2.minorticks_on()
        # Turn on the minor TICKS, which are required for the minor GRID
        # Customize the major grid
        ax2.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        # Customize the minor grid
        ax2.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        ax2.plot(
                 data_df.index, data_df["close"],
                 data_df.index, trend_pred_df["trend"]
                 )
        ax2.set_ylabel(f'Pred, power = {self.power_trend}', color='b')
        plt.title(f"Trend with power: {self.power_trend}")
        path_filename = os.path.join('outputs',
                                     f"{self.experiment_name}_{self.net_name}_{self.power_trend}_trend_predict_{show_data}.png")
        plt.savefig(path_filename, dpi=96, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None
                    )
        plt.show()
        pass





