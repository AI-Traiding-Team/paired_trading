import pandas as pd
from datamodeling import *
from networks import *
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from powerrun.models import get_resnet1d_and_regression_model

__version__ = 0.0004

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
    def __init__(self, path_filename, all_data_df, df_priority=False):

        self.timeframe = None
        self.symbol = None
        self.path_filename = path_filename
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
        self.y2_df = None
        self.y2_scaler = None
        # self.log_scaler = FunctionTransformer(np.log1p, inverse_func=np.expm1)
        self.train_df_len = None
        self.val_df_len = None
        self.df_test_len = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.features_scaler = None
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

    def prepare_output_y1_y2(self, y1_df, y2_df):
        y1 = y1_df["Signal"].values
        y1 = np.expand_dims(y1, axis=1)
        y2 = y2_df["Trend_length"].values
        y2 = np.expand_dims(y2, axis=1)
        # y2_scaled = self.log_scaler.fit_transform(y2)

        self.y2_scaler = RobustScaler().fit(y2)
        y2_scaled = self.y2_scaler.transform(y2)


        # self.y2_scaler = StandardScaler()
        # self.y2_scaler.fit(y2)
        # y2_scaled = self.y2_scaler.transform(y2)

        # self.y2_scaler = MinMaxScaler()
        # self.y2_scaler.fit(y2)
        # y2_scaled = self.y2_scaler.transform(y2)
        data_y1_y2 = np.hstack((y1, y2_scaled))
        return data_y1_y2

    def prepare_data(self):
        print("\nAll dataframe data example (Signal markup with treshhold 0.0275):")
        print(self.all_data_df.head().to_string())
        self.features_df = self.all_data_df.iloc[:, :-2]
        print("\nX (features) dataframe data example:")
        print(self.features_df.head().to_string(), f"\n")
        self.y_df = self.all_data_df.iloc[:, -1:]
        print("\nSignal (True_1) dataframe data example:")
        print(self.y_df.head().to_string(), f"\n")
        self.y2_df = self.all_data_df.iloc[:, -2:-1]
        print("\nTrend length in ticks to signal change (True_2) dataframe data example:")
        print(self.y2_df.head().to_string(), f"\n")

        uniques, counts = np.unique(self.y_df.values, return_counts=True)
        for unq, cnt in zip(uniques, counts):
            print("Total:", unq, cnt)

        self.calculate_split_df()
        msg = f"\nSplit dataframe:" \
              f"Train start-end and length: {self.train_df_start_end[0]}-{self.train_df_start_end[1]} {self.train_df_start_end[0] - self.train_df_start_end[1]}\n" \
              f"Validation start-end and length: {self.val_df_start_end[0]}-{self.val_df_start_end[1]} {self.val_df_start_end[0] - self.val_df_start_end[1]}\n" \
              f"Test start-end and length: {self.test_df_start_end[0]}-{self.test_df_start_end[1]} {self.test_df_start_end[0] - self.test_df_start_end[1]}"
        print(f"{msg}\n")
        self.split_data_df()
        _temp_1 = pd.DataFrame()
        _temp_1 = self.all_data_df["close"].copy()
        self.symbol = self.path_filename.split("-")[0].split("/")[-1]
        self.timeframe = self.path_filename.split("-")[1].split(".")[0]

        dataset_split_show(_temp_1,
                           self.train_df_start_end,
                           self.val_df_start_end,
                           self.test_df_start_end,
                           f"{self.symbol}-{self.timeframe}"
                           )
        self.features_scaler = RobustScaler().fit(self.features_df.values)
        x_arr = self.features_scaler.transform(self.features_df.values)
        print("\nCreate arrays with X (features)", x_arr.shape)

        y_outs = self.prepare_output_y1_y2(self.y_df, self.y2_df)
        # y_arr = self.y_df.values.reshape(-1, 1)
        print("\nCreate array Signal (True) and Trend length (True)", y_outs.shape)
        # print("\nCreate array with Trend length (True)", y_outs[1].shape)
        self.prepare_datagens(x_arr, y_outs)

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
            x_Test_gen = self.get_test_generator(x_Test_data, y_Test_data)
            """ Using generator 1 time to have solid data """
            self.x_Test, self.y_Test = self.create_data_from_gen(x_Test_data, y_Test_data)
            x_Test_gen = self.get_test_generator(x_Test_data, y_Test_data)
        msg = f"Created arrays: \nx_Train_data = {x_Train_data.shape}, y_Train_data = {y_Train_data.shape}\n" \
              f"x_Val_data = {x_Val_data.shape}, y_Val_data = {y_Val_data.shape}\n" \
              f"x_Test_data = {x_Test_data.shape}, y_Test_data = {y_Test_data.shape}\n"
        print(f"{msg}\n")
        """" Using generator 1 time to get solid data arrays"""
        x_Train_gen = self.get_train_generator(x_Train_data, y_Train_data)
        x_Val_gen = self.get_val_generator(x_Val_data, y_Val_data)
        self.x_Train, self.y_Train = self.create_data_from_gen(x_Train_data, y_Train_data)
        self.x_Val, self.y_Val = self.create_data_from_gen(x_Val_data, y_Val_data)
        self.input_shape = x_Val_gen.sample_shape
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
    def __init__(self, mrk_dataset: MarkedDataSet):

        self.y_Pred = None
        self.mrk_dataset = mrk_dataset
        self.power_trend = 0.0275
        self.net_name = "resnet1d"
        self.experiment_name = f"{self.mrk_dataset.symbol}-{self.mrk_dataset.timeframe}"
        self.symbol = self.mrk_dataset.symbol
        self.timeframe = self.mrk_dataset.timeframe
        self.power_trends_list = (0.15, 0.075, 0.055, 0.0275)
        self.batch_size = 32
        self.history = None
        self.epochs = 15

        self.keras_model = get_resnet1d_and_regression_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.optimizer_sgd = tf.keras.optimizers.SGD(learning_rate=1e-5,
                                                     momentum=0.9,
                                                     nesterov=True
                                                     ),

        self.path_filename = os.path.join('outputs', f"{self.experiment_name}_{self.net_name}_NN.png")
        self.compile()

    def train(self):
        chkp = tf.keras.callbacks.ModelCheckpoint(os.path.join("outputs", f"{self.experiment_name}_{self.net_name}_{self.power_trend}.h5"), monitor='val_trnd_dir_accuracy', save_best_only=True)
        rlrs = ReduceLROnPlateau(monitor='val_trnd_dir_accuracy', factor=0.2, patience=14, min_lr=0.000001)
        callbacks = [chkp, rlrs]
        path_filename = os.path.join('outputs', f"{self.experiment_name}_{self.net_name}_NN.png")
        tf.keras.utils.plot_model(self.keras_model,
                                  to_file=path_filename,
                                  show_shapes=True,
                                  show_layer_names=True,
                                  expand_nested=True,
                                  dpi=96,
                                  )
        print(self.mrk_dataset.y_Train[0, :].shape)
        y_Train_1 = self.mrk_dataset.y_Train[:, 0]
        y_Train_2 = self.mrk_dataset.y_Train[:, 1]
        y_Val_1 = self.mrk_dataset.y_Val[:, 0]
        y_Val_2 = self.mrk_dataset.y_Val[:, 1]
        self.history = self.keras_model.fit(self.mrk_dataset.x_Train,
                                            [y_Train_1, y_Train_2],
                                            epochs=self.epochs,
                                            validation_data=(self.mrk_dataset.x_Val,
                                                             [y_Val_1, y_Val_2]
                                                             ),
                                            batch_size=self.batch_size,
                                            verbose=1,
                                            callbacks=callbacks
                                            )

    def compile(self):
        self.keras_model.compile(optimizer=self.optimizer,
                                 loss={'trnd_dir': 'binary_crossentropy',
                                       'tcks_2chng': 'mae'
                                       },
                                 metrics=['accuracy'],
                                 # loss = {'trnd_dir': 'binary_crossentropy',
                                 #         'tcks_2chng': 'mae'
                                 #         },
                                 # metrics={'trnd_dir': 'accuracy',
                                 #          'tcks_2chng': tf.keras.losses.Huber(),
                                 #          },
                                 )

        pass

    def get_predict(self):
        path_filename = os.path.join('outputs', f"{self.experiment_name}_{self.net_name}.h5")
        tf.keras.models.load_model(path_filename)
        print(self.mrk_dataset.x_Test)
        self.y_Pred = self.keras_model.predict(self.mrk_dataset.x_Test)
        return self.y_Pred

    def load_best_weights(self):
        path_filename = os.path.join('outputs', f"{self.experiment_name}_{self.net_name}_{self.power_trend}.h5")
        self.keras_model.load_weights(path_filename)
        pass

    def figshow_base(self):
        fig = plt.figure(figsize=(12, 7))
        sns.set_style("white")
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_axisbelow(True)
        ax1.minorticks_on()
        N = np.arange(0, len(self.history.history["loss"]))
        plt.plot(N, self.history.history["loss"], label="loss")
        if 'tcks_2chng_huber_loss' in self.history.history:
            plt.plot(N, self.history.history["tcks_2chng_huber_loss"], label="tcks_2chng_huber_loss")
        if 'val_tcks_2chng_huber_loss' in self.history.history:
            plt.plot(N, self.history.history["val_tcks_2chng_huber_loss"], label="val_tcks_2chng_huber_loss")
        if 'mae_loss' in self.history.history:
            plt.plot(N, self.history.history["mae_loss"], label="mae_loss")
        if 'val_mae_loss' in self.history.history:
            plt.plot(N, self.history.history["val_mae_loss"], label="val_mae_loss")
        if 'trnd_dir_accuracy' in self.history.history:
            plt.plot(N, self.history.history["trnd_dir_accuracy"], label="trnd_dir_accuracy")
        if 'val_trnd_dir_accuracy' in self.history.history:
            plt.plot(N, self.history.history["val_trnd_dir_accuracy"], label="val_trnd_dir_accuracy")
        plt.title(f"Training Loss and Accuracy")
        plt.legend()
        plt.show()
        pass

    def backtest_test_dataset(self):
        print("Creating backtesting set")
        data_df = self.mrk_dataset.features_df[
                  self.mrk_dataset.test_df_start_end[0]: self.mrk_dataset.test_df_start_end[
                                                             1] - self.mrk_dataset.tsg_window_length]
        trend_pred = self.keras_model.predict(self.mrk_dataset.x_Test)
        data_df['trend'] = trend_pred.flatten()
        data_df.loc[data_df["trend"] <= 0.5, "trend"] = 0.0
        data_df.loc[data_df["trend"] > 0.5, "trend"] = 1.0
        data_df["Signal"] = data_df['trend']
        # self.mrk_dataset.test_df_backtrade = data_df.copy()
        data_df.drop(columns=[
                             "trend",
                             "quarter", "month", "weeknum", "weekday", "hour",
                             "minute", "log_close", "log_volume", "diff_close",
                             "log_close_close_shift", "sin_close"], inplace=True)

        data_df.columns = [item.lower().capitalize() for item in data_df.columns]

        print("\nSignal (pred) dataframe data example for backtesting:")
        print(data_df["Signal"].head().to_string(), f"\n")
        uniques, counts = np.unique(data_df["Signal"].values, return_counts=True)
        for unq, cnt in zip(uniques, counts):
            print("Total:", unq, cnt)
        self.mrk_dataset.test_df_backtrade = data_df
        return data_df

    def show_trend_predict(self):
        weight = self.power_trend
        print(f"\nВизуализируем результат")
        data_df = self.mrk_dataset.features_df[
                  self.mrk_dataset.test_df_start_end[0]: self.mrk_dataset.test_df_start_end[
                                                             1] - self.mrk_dataset.tsg_window_length - self.mrk_dataset.tsg_overlap]
        y_df = self.mrk_dataset.y_df[self.mrk_dataset.test_df_start_end[0]: self.mrk_dataset.test_df_start_end[
                                                                                1] - self.mrk_dataset.tsg_window_length - self.mrk_dataset.tsg_overlap]
        y2_df = self.mrk_dataset.y2_df[self.mrk_dataset.test_df_start_end[0]: self.mrk_dataset.test_df_start_end[
                                                                                1] - self.mrk_dataset.tsg_window_length - self.mrk_dataset.tsg_overlap]
        trend_pred = self.keras_model.predict(self.mrk_dataset.x_Test)

        y_Test_pred_1 = trend_pred[0]
        y_Test_pred_2_unscaled = np.reshape(trend_pred[1], (-1, 1))
        y_Test_pred_2 = self.mrk_dataset.y2_scaler.inverse_transform(y_Test_pred_2_unscaled)

        # trend_pred = trend_pred.flatten()
        trend_pred_df = pd.DataFrame(data=y_Test_pred_1, columns=["Trend"])
        trend_pred_df["Trend_change"] = pd.Series(data=y_Test_pred_2.flatten())
        trend_pred_df["Trend_change_signal"] = pd.Series(data=y_Test_pred_2.flatten())

        # for visualization we use scaling of trend = 1 to data_df["close"].max()
        max_close = data_df["close"].max()
        min_close = data_df["close"].min()
        mean_close = data_df["close"].mean()
        trend_pred_df.loc[(trend_pred_df["Trend_change_signal"] <= 10), "Trend_change_signal"] = max_close
        trend_pred_df.loc[(trend_pred_df["Trend_change_signal"] > 10), "Trend_change_signal"] = min_close
        trend_pred_df.loc[(trend_pred_df["Trend"] > 0.5), "Trend"] = max_close
        trend_pred_df.loc[(trend_pred_df["Trend"] <= 0.5), "Trend"] = min_close
        y_df.loc[(y_df["Signal"] == 1), "Signal"] = max_close
        y_df.loc[(y_df["Signal"] == 0), "Signal"] = min_close
        data_df[f"Trend_{weight}"] = y_df["Signal"]

        col_list = data_df.columns.to_list()
        try:
           col_list.index("close")
        except ValueError:
           msg = f"Error: 'close' column not found in pd.DataFrame only {col_list}. Can't show figure"
           sys.exit(msg)
        fig = plt.figure(figsize=(20, 12))
        ax1 = fig.add_subplot(2, 1,  1)
        ax1.plot(
                 data_df.index, data_df[f"Trend_{weight}"],
                 data_df.index, data_df["close"],
                 )
        ax1.set_ylabel(f'True, weight = {weight}', color='r')
        plt.title(f"Trend with weight: {weight}")
        ax2 = fig.add_subplot(2, 1,  2)
        ax2.plot(
                 data_df.index, data_df["close"],
                 data_df.index, trend_pred_df["Trend"],
                 data_df.index, trend_pred_df["Trend_change_signal"]
                 )
        ax2.set_ylabel(f'Pred, weight = {weight}', color='b')
        plt.title(f"Trend with weight: {weight}")
        plt.show()
        pass





