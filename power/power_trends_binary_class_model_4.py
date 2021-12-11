from datamodeling import *
from analyze import DataLoad
from networks import *

__version__ = 0.0003


class TrainNN:
    def __init__(self):
        """
        Model 4,
        Classification, trend with thresholds
        """
        self.dataset_profile = DSProfile()
        self.dataset_profile.Y_data = "power_trend"
        self.dataset_profile.timeframe = "1m"
        self.dataset_profile.use_symbols_pairs = ("ETHUSDT", "BTCUSDT", "ETHBTC")
        self.dataset_profile.power_trend = 0.075

        """ Default options for dataset window"""
        self.dataset_profile.tsg_window_length = 40
        self.dataset_profile.tsg_sampling_rate = 1
        self.dataset_profile.tsg_stride = 1
        self.dataset_profile.tsg_start_index = 0
        """ Warning! Change this qty if using .shift() more then 2 """
        self.dataset_profile.tsg_overlap = 0

        self.dsc = DSCreator(loaded_crypto_data, self.dataset_profile)
        self.dts_power_trend = self.dsc.create_dataset()
        self.nn_profile = NNProfile("categorical_crossentropy")
        self.nn_profile.learning_rate = 1e-4
        self.nn_profile.experiment_name = f"{self.nn_profile.experiment_name}_categorical_trend"
        self.nn_profile.epochs = 350
        self.nn_network = MainNN(self.nn_profile)
        pass

    def train_model(self):
        self.nn_profile.num_classes = 2
        self.nn_network.train_model(self.dts_power_trend)
        # pred = test_nn.get_predict()
        # # print(pred)
        self.nn_network.show_categorical()
        pass

    def get_dataset(self):
        return self.dts_power_trend


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
                                  source_directory="/home/cubecloud/Python/projects/paired_trading/source_root",
                                  start_period='2021-09-01 00:00:00',
                                  end_period='2021-12-05 23:59:59',
                                  )

    tr = TrainNN()
    """
    Model 4, 
    Classification, trend with thresholds
    """
    tr.train_model()




