import pandas as pd

from datamodeling import *
from analyze import DataLoad
from networks import *

__version__ = 0.0005


class TrainNN:
    def __init__(self):
        """
        Model 4,
        Classification, trend with thresholds
        """

        self.power_trends_list = (0.15, 0.075, 0.055, 0.0275)
        self.dataset_profile = DSProfile()
        self.dataset_profile.Y_data = "power_trend"
        self.dataset_profile.timeframe = "1m"
        self.dataset_profile.use_symbols_pairs = ("ETHUSDT", "BTCUSDT", "ETHBTC")


        """ Default options for dataset window"""
        self.dataset_profile.tsg_window_length = 40
        self.dataset_profile.tsg_sampling_rate = 1
        self.dataset_profile.tsg_stride = 1
        self.dataset_profile.tsg_start_index = 0
        """ Warning! Change this qty if using .shift() more then 2 """
        self.dataset_profile.tsg_overlap = 0

        self.dsc = DSCreator(loaded_crypto_data, self.dataset_profile)
        print("Инициализируем подготовку датасета")
        self.dts_power_trend = self.dsc.create_dataset()

        self.nn_profile = NNProfile("categorical_crossentropy")
        self.nn_profile.learning_rate = 1e-4
        self.nn_profile.experiment_name = f"{self.nn_profile.experiment_name}_categorical_trend"
        self.nn_profile.epochs = 5
        self.nn_network = MainNN(self.nn_profile)
        pass

    def train_model(self):
        self.dts_power_trend = self.dsc.create_dataset()
        self.dataset_profile.power_trend = 0.075
        self.nn_profile.num_classes = 2
        self.nn_network.train_model(self.dts_power_trend)
        self.nn_network.show_categorical()
        pass

    def get_dataset(self):
        return self.dts_power_trend

    def check_trends_weights(self, use_col: str = "trend" ) -> None:
        """
        Args:
            use_col (str):          name of column. default "trend"

        Returns:
            None:
        """
        for weight in self.power_trends_list:
            print(f"Считаем тренд с power = {weight}")
            data_df = self.dsc.features.source_df_3
            trend_df = self.dsc.features.calculate_trend(data_df, weight)
            # for visualization we use scaling of trend = 1 to data_df["close"].max()
            max_close = data_df["close"].max()
            min_close = data_df["close"].min()
            mean_close = data_df["close"].mean()
            trend_df.loc[(trend_df["trend"] == 1), "trend"] = max_close
            trend_df.loc[(trend_df["trend"] == -1), "trend"] = min_close
            trend_df.loc[(trend_df["trend"] == 0), "trend"] = mean_close
            data_df[f"trend_{weight}"] = trend_df[use_col]
        col_list = data_df.columns.to_list()
        try:
            col_list.index("close")
        except ValueError:
            msg = f"Error: 'close' column not found in pd.DataFrame only {col_list}. Can't show figure"
            sys.exit(msg)

        weights_list_len = len(self.power_trends_list)
        fig = plt.figure(figsize=(20, 6 * weights_list_len))

        for i, weight in enumerate(self.power_trends_list):
            ax1 = fig.add_subplot(weights_list_len, 1, i + 1)
            ax1.plot(data_df.index, data_df[f"trend_{weight}"], data_df.index, data_df["close"])
            ax1.set_ylabel(f'weight = {weight}', color='r')
            plt.title(f"Trend with weight: {weight}")
        plt.show()
        pass

    def show_trend_predict(self):
        weight = self.dataset_profile.power_trend
        print(f"Считаем тренд с power = {weight}")
        data_df = self.dsc.features.source_df_3.copy()
        data_df.drop(index=self.dsc.features.drop_idxs)

        trend_df = self.dsc.features.calculate_trend(data_df, weight)
        trend_df = trend_df.iloc[-self.dsc.df_test_len:, :]
        test_df = data_df.iloc[-self.dsc.df_test_len:, :]
        trend_pred = self.nn_network.get_predict()
        trend_pred_df = pd.DataFrame(trend_pred[1, :])
        # for visualization we use scaling of trend = 1 to data_df["close"].max()
        max_close = data_df["close"].max()
        min_close = data_df["close"].min()
        mean_close = data_df["close"].mean()
        trend_df.loc[(trend_df["trend"] == 1), "trend"] = max_close
        trend_df.loc[(trend_df["trend"] == -1), "trend"] = min_close
        trend_df.loc[(trend_df["trend"] == 0), "trend"] = mean_close
        data_df[f"trend_{weight}"] = trend_df["trend"]
        col_list = data_df.columns.to_list()
        try:
            col_list.index("close")
        except ValueError:
            msg = f"Error: 'close' column not found in pd.DataFrame only {col_list}. Can't show figure"
            sys.exit(msg)

        fig = plt.figure(figsize=(20, 6))
        ax1 = fig.add_subplot(1, 1,  1)
        ax1.plot(data_df.index, data_df[f"trend_{weight}"], data_df.index, data_df["close"])
        ax1.set_ylabel(f'weight = {weight}', color='r')
        plt.title(f"Trend with weight: {weight}")
        plt.show()
        pass




if __name__ == "__main__":
    """
    Usage for DataLoad class
    ------------------------
    pairs_symbol = None ->                    Use all pairs in timeframe directory
    pairs_symbol = ("BTCUSDT", "ETHUSDT") ->  Use only this pairs to load 

    time_intervals = None ->                Use all timeframes directories for loading (with pairs_symbols)
    time_intervals = ['15m'] ->             Use timeframes from this list to load

    start_period = None ->                  Use from [0:] of historical data
    start_period = '2021-09-01 00:00:00' -> Use from this datetimeindex

    end_period = None ->                    Use until [:-1] of historical data
    end_period = '2021-12-05 23:59:59' ->   Use until this datetimeindex

    source_directory="../source_root" ->    Use this directory to search timeframes directory
    """

    loaded_crypto_data = DataLoad(pairs_symbols=None,
                                  time_intervals=['1m'],
                                  source_directory="../source_root",
                                  start_period='2021-09-01 00:00:00',
                                  end_period='2021-12-05 23:59:59',
                                  )

    tr = TrainNN()
    """
    Model 4, 
    Classification, trend with thresholds
    """
    print("Считаем возможные варианты трендов")
    # tr.check_trends_weights()
    tr.train_model()
    tr.show_trend_predict()





