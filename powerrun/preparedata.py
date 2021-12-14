from datamodeling import *

__version__ = 0.0001


def dataset_split_show(data1, data2, data3, symbol):
    plt.figure(figsize=(12, 4))
    ax0 = plt.subplot2grid((4, 4), (0, 0), rowspan=3, colspan=4)
    # df['Close'].plot(ax = ax0, label='all')
    data1["close"].plot(ax=ax0, label='train')
    data2["close"].plot(ax=ax0, label='val')
    data3["close"].plot(ax=ax0, label='test')
    plt.title(f'График изменений цены на {symbol}')
    plt.legend()
    plt.grid()
    plt.show()
    pass

class DataModel:
    def __init__(self, data_df, verbose=0):
        self.verbose = verbose
        self.all_data_df = data_df
        self.train_df_start_end: list = [0, 0]
        self.val_df_start_end: list = [0, 0]
        self.test_df_start_end: list = [0, 0]
        pass

    def run(self):
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
              f"Train start-end and length: {self.train_df_start_end[0]}-{self.train_df_start_end[1]} {self.train_df_start_end[0] - self.train_df_start_end[1]}\n" \
              f"Validation start-end and length: {self.val_df_start_end[0]}-{self.val_df_start_end[1]} {self.val_df_start_end[0] - self.val_df_start_end[1]}\n" \
              f"Test start-end and length: {self.test_df_start_end[0]}-{self.test_df_start_end[1]} {self.test_df_start_end[0] - self.test_df_start_end[1]}"
        print(f"{msg}\n")

        self.split_data_df()
        temp_1 = pd.DataFrame()
        temp_1["close"] = self.all_data_df["close"].copy()
        temp_2 = temp_1.copy()
        temp_3 = temp_1.copy()
        temp_1[self.train_df_start_end[1]:] = 0
        temp_2[:self.val_df_start_end[0]] = 0
        temp_2[self.val_df_start_end[1]:] = 0
        temp_3[:self.test_df_start_end[0]] = 0
        symbol = self.path_filename.split("-")[0].split("/")[-1]
        timeframe = self.path_filename.split("-")[1].split(".")[0]
        if not self.verbose:
            dataset_split_show(temp_1, temp_2, temp_3, f"{symbol}-{timeframe}")
        self.features_scaler = RobustScaler().fit(self.features_df.values)
        x_arr = self.features_scaler.transform(self.features_df.values)
        print("\nCreate arrays with X (features)", x_arr.shape)
        y_arr = self.y_df.values.reshape(-1, 1)
        print("\nCreate arrays with Signal (True)", y_arr.shape)
        self.prepare_datagens(x_arr, y_arr)
        pass

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