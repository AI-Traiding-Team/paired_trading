import pandas as pd
from datamodeling import *
from analyze import DataLoad
from networks import *
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, Conv1D, ReLU, ELU, MaxPool1D, Reshape, Dropout
from backtesting import *

__version__ = 0.0014


class MarkedDataSet:
    def __init__(self, path_filename):

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
        self.all_data_df = None
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
        self.prepare_data()

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
        self.all_data_df = pd.read_csv(path_filename, index_col="datetimeindex")

        print("All dataframe data example:")
        print(self.all_data_df.head().to_string())
        self.features_df = self.all_data_df.iloc[:, :-1]
        print("X (features) dataframe data example:")
        print(self.features_df.head().to_string())
        self.y_df = self.all_data_df.iloc[:, -1:]
        print("Y (true) dataframe data example:")
        print(self.y_df.head().to_string())
        uniques, counts = np.unique(self.y_df.values, return_counts=True)
        for unq, cnt in zip(uniques, counts):
            print("Total:", unq, cnt)

        self.calculate_split_df()
        msg = f"Split dataframe:" \
              f"Train start-end and length: {self.train_df_start_end[0]}-{self.train_df_start_end[1]} {self.train_df_start_end[0] - self.train_df_start_end[1]}\n" \
              f"Validation start-end and length: {self.val_df_start_end[0]}-{self.val_df_start_end[1]} {self.val_df_start_end[0] - self.val_df_start_end[1]}\n" \
              f"Test start-end and length: {self.test_df_start_end[0]}-{self.test_df_start_end[1]} {self.test_df_start_end[0] - self.test_df_start_end[1]}"
        print(msg)

        self.split_data_df()

        self.features_scaler = RobustScaler().fit(self.features_df.values)
        x_arr = self.features_scaler.transform(self.features_df.values)
        print("Create arrays with X (features)", x_arr.shape)
        y_arr = self.y_df.values.reshape(-1, 1)
        print("Create arrays with Y (true)", y_arr.shape)
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
            x_Test_gen = self.get_test_generator(x_Test_data, y_Test_data)
            """ Using generator 1 time to have solid data """
            self.x_Test, self.y_Test = self.create_data_from_gen(x_Test_data, y_Test_data)
            # x_Test_gen = self.get_test_generator(x_Test_data, y_Test_data)
        msg = f"Created arrays: \nx_Train_data = {x_Train_data.shape}, y_Train_data = {y_Train_data.shape}\n" \
              f"x_Val_data = {x_Val_data.shape}, y_Val_data = {y_Val_data.shape}\n" \
              f"x_Test_data = {x_Test_data.shape}, y_Test_data = {y_Test_data.shape}\n"
        print(msg)
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

def get_resnet1d_model(
                       input_shape=(40, 16,),
                       num_classes=2,
                       kernels=32,
                       stride=4,
                       ):
    def residual_block(x, kernels, stride):
        # tf.keras.backend.set_floatx('float64')
        out = Conv1D(kernels, stride, padding='same')(x)
        out = ReLU()(out)
        out = Conv1D(kernels, stride, padding='same')(out)
        out = tf.keras.layers.add([x, out])
        out = ReLU()(out)
        out = MaxPool1D(3, 2)(out)
        return out

    x_in = Input(shape=input_shape)
    x = Dense(16, activation="elu")(x_in)
    x = Conv1D(kernels, stride)(x)
    x = residual_block(x, kernels, stride)
    x = residual_block(x, kernels, stride)
    x = residual_block(x, kernels, stride)
    x = residual_block(x, kernels, stride)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.35)(x)
    x = Dense(32, activation="relu")(x)
    x_out = Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=x_in, outputs=x_out)
    return model

class TrainNN:
    def __init__(self, mrk_dataset: MarkedDataSet):
        self.power_trend = 0.0275
        self.net_name = "resnet1d"
        self.experiment_name = "ETHUSDT-1m"
        self.symbol = self.experiment_name.split('-')[0]
        self.timeframe = self.experiment_name.split('-')[1]
        self.power_trends_list = (0.15, 0.075, 0.055, 0.0275)
        self.mrk_dataset = mrk_dataset
        self.history = None
        self.keras_model = get_resnet1d_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.optimizer_sgd = tf.keras.optimizers.SGD(learning_rate=1e-5,
                                                     momentum=0.9,
                                                     nesterov=True
                                                     ),

        self.path_filename = os.path.join( 'outputs', f"{self.experiment_name}_{self.net_name}_NN.png")
        self.compile()
        pass

    def train(self):
        chkp = tf.keras.callbacks.ModelCheckpoint(os.path.join("outputs", f"{self.experiment_name}_{self.net_name}_{self.power_trend}.h5"), monitor='val_accuracy', save_best_only=True)
        rlrs = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=13, min_lr=0.000001)
        callbacks = [rlrs, chkp]
        path_filename = os.path.join(os.getcwd(), 'outputs', f"{self.experiment_name}_{self.net_name}_NN.png")
        tf.keras.utils.plot_model(self.keras_model,
                                  to_file=path_filename,
                                  show_shapes=True,
                                  show_layer_names=True,
                                  expand_nested=True,
                                  dpi=96,
                                  )
        self.history = self.keras_model.fit(self.mrk_dataset.train_gen,
                                            epochs=100,
                                            validation_data=self.mrk_dataset.train_gen,
                                            verbose=1,
                                            callbacks=callbacks
                                            )
    def compile(self):
        self.keras_model.compile(optimizer=self.optimizer,
                                 loss="binary_crossentropy",
                                 metrics=["accuracy"],
                                 )
        pass
    def get_predict(self):
        path_filename = os.path.join('outputs', f"{self.experiment_name}_{self.net_name}.h5")
        tf.keras.models.load_model(path_filename)
        self.y_Pred = self.keras_model.predict(self.mrk_dataset.x_Test)
        return self.y_Pred

    def load_best_weights(self):
        path_filename = os.path.join('outputs', f"{self.experiment_name}_{self.net_name}_{self.power_trend}.h5")
        tf.keras.models.load_weights(path_filename)
        pass

    def figshow_base(self):
        fig = plt.figure(figsize=(12, 7))
        sns.set_style("white")
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_axisbelow(True)
        ax1.minorticks_on()
        N = np.arange(0, len(self.history.history["loss"]))
        plt.plot(N, self.history.history["loss"], label="loss")
        if 'dice_coef' in self.history.history:
            plt.plot(N, self.history.history["dice_coef"], label="dice_coef")
        if 'val_dice_coef' in self.history.history:
            plt.plot(N, self.history.history["val_dice_coef"], label="val_dice_coef")
        if 'mae' in self.history.history:
            plt.plot(N, self.history.history["mae"], label="mae")
        if 'accuracy' in self.history.history:
            plt.plot(N, self.history.history["accuracy"], label="accuracy")
        if 'accuracy' in self.history.history:
            plt.plot(N, self.history.history["val_accuracy"], label="val_accuracy")
        if 'val_loss' in self.history.history:
            plt.plot(N, self.history.history["val_loss"], label="val_loss")
        if 'lr' in self.history.history:
            lr_list = [x * 1000 for x in self.history.history["lr"]]
            plt.plot(N, lr_list, linestyle=':', label="lr * 1000")
        plt.title(f"Training Loss and Accuracy")
        plt.legend()
        plt.show()
        pass

    def check_binary(self):
        x_test = self.mrk_dataset.x_Test[-600:-500]
        y_test_org = self.mrk_dataset.y_Test[-600:-500]
        conv_test = []
        for i in range(len(x_test)):
            x = x_test[i]
            x = np.expand_dims(x, axis=0)
            prediction = self.keras_model.predict(x)
            if prediction > 0.5:
                prediction = 1
            else:
                prediction = 0
            if prediction == y_test_org[i]:
                conv_test.append('True')
            else:
                conv_test.append('False')

            print(f'Index: {i}, Prediction: {prediction}, Real: {y_test_org[i]},\t====> {y_test_org[i]} {conv_test[i]}')

        uniques, counts = np.unique(conv_test, return_counts=True)
        for unq, cnt in zip(uniques, counts):
            print("Total:", unq, cnt)

    def show_trend_predict(self):
        weight = self.power_trend
        print(f"Считаем тренд с power = {weight}")
        data_df = self.mrk_dataset.features_df[
                  self.mrk_dataset.test_df_start_end[0]: self.mrk_dataset.test_df_start_end[
                                                             1] - self.mrk_dataset.tsg_window_length]
        y_df = self.mrk_dataset.y_df[self.mrk_dataset.test_df_start_end[0]: self.mrk_dataset.test_df_start_end[
                                                                                1] - self.mrk_dataset.tsg_window_length]
        trend_pred = self.keras_model.predict(self.mrk_dataset.x_Test)
        trend_pred = trend_pred.flatten()
        trend_pred_df = pd.DataFrame(data=trend_pred, columns=["trend"])
        # for visualization we use scaling of trend = 1 to data_df["close"].max()
        max_close = data_df["close"].max()
        min_close = data_df["close"].min()
        mean_close = data_df["close"].mean()
        trend_pred_df.loc[(trend_pred_df["trend"] > 0.5), "trend"] = max_close
        y_df.loc[(y_df["y"] == 1), "y"] = max_close
        trend_pred_df.loc[(trend_pred_df["trend"] <= 0.5), "trend"] = min_close
        y_df.loc[(y_df["y"] == 0), "y"] = min_close
        data_df[f"trend_{weight}"] = y_df["y"]

        col_list = data_df.columns.to_list()
        try:
           col_list.index("close")
        except ValueError:
           msg = f"Error: 'close' column not found in pd.DataFrame only {col_list}. Can't show figure"
           sys.exit(msg)
        fig = plt.figure(figsize=(20, 12))
        ax1 = fig.add_subplot(2, 1,  1)
        ax1.plot(
                 data_df.index, data_df[f"trend_{weight}"],
                 data_df.index, data_df["close"],
                 )
        ax1.set_ylabel(f'True, weight = {weight}', color='r')
        plt.title(f"Trend with weight: {weight}")
        ax2 = fig.add_subplot(2, 1,  2)
        ax2.plot(
                 data_df.index, data_df["close"],
                 data_df.index, trend_pred_df["trend"]
                 )
        ax2.set_ylabel(f'Pred, weight = {weight}', color='b')
        plt.title(f"Trend with weight: {weight}")
        plt.show()
        pass

N_TRAIN = 400

path_filename ="../source_ds/1m/ETHUSDT-1m.csv"
dataset = MarkedDataSet(path_filename)
tr = TrainNN(dataset)
tr.train()
# tr.compile()
tr.load_best_weights()
tr.show_trend_predict()
tr.check_binary()

# loaded_crypto_data = DataLoad(pairs_symbols=None,
#                               time_intervals=['1m'],
#                               source_directory="../source_root",
#                               start_period='2021-09-01 00:00:00',
#                               end_period='2021-12-05 23:59:59',
#                               )

# class MLTrainOnceStrategy(Strategy):
#     price_delta = .004  # 0.4%
#
#     def init(self):
#         # Init our model, a kNN classifier
#         self.clf = tr
#
#         # Train the classifier in advance on the first N_TRAIN examples
#         data_df = loaded_crypto_data.ohlcvbase[f"{tr.symbol}-{tr.timeframe}"].df
#         data_df = data_df.iloc[tr.mrk_dataset.test_df_start_end[0]: tr.mrk_dataset.test_df_start_end[1], :]
#
#         # X, y = get_clean_Xy(df)
#         # self.clf.fit(X, y)
#
#         # Plot y for inspection
#         self.I(get_y, self.data.df, name='y_true')
#
#         # Prepare empty, all-NaN forecast indicator
#         self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')
#
#     def next(self):
#         # Skip the training, in-sample data
#         if len(self.data) < N_TRAIN:
#             return
#
#         # Proceed only with out-of-sample data. Prepare some variables
#         high, low, close = self.data.High, self.data.Low, self.data.Close
#         current_time = self.data.index[-1]
#
#         # Forecast the next movement
#         X = get_X(self.data.df.iloc[-1:])
#         forecast = self.clf.predict(X)[0]
#
#         # Update the plotted "forecast" indicator
#         self.forecasts[-1] = forecast
#
#         # If our forecast is upwards and we don't already hold a long position
#         # place a long order for 20% of available account equity. Vice versa for short.
#         # Also set target take-profit and stop-loss prices to be one price_delta
#         # away from the current closing price.
#         upper, lower = close[-1] * (1 + np.r_[1, -1]*self.price_delta)
#
#         if forecast == 1 and not self.position.is_long:
#             self.buy(size=.2, tp=upper, sl=lower)
#         elif forecast == -1 and not self.position.is_short:
#             self.sell(size=.2, tp=lower, sl=upper)
#
#         # Additionally, set aggressive stop-loss on trades that have been open
#         # for more than two days
#         for trade in self.trades:
#             if current_time - trade.entry_time > pd.Timedelta('2 days'):
#                 if trade.is_long:
#                     trade.sl = max(trade.sl, low)
#                 else:
#                     trade.sl = min(trade.sl, high)




# bt = Backtest(tr.mrk_dataset.test_df, LnS, cash=10000,commision=0.004, trade_on_close=True )
# stats = bt.run()
# bt.plot(plot_volume=True, relative_equity=True)


# tr.figshow_base()
# tr.show_trend_predict()
# tr.check_binary()

