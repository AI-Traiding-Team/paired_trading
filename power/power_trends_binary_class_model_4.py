from datamodeling import *
from analyze import DataLoad
from networks import *

__version__ = 0.0001


class TrainNN:
    def __init__(self):
        pass

    def train_model(self):
        """
        Model 4,
        Classification, trend with thresholds
        """
        dataset_4_profile = DSProfile()
        dataset_4_profile.Y_data = "close1-close2_trend"
        dataset_4_profile.timeframe = "1m"

        """ Default options for dataset window"""
        dataset_4_profile.tsg_window_length = 40
        dataset_4_profile.tsg_sampling_rate = 1
        dataset_4_profile.tsg_stride = 1
        dataset_4_profile.tsg_start_index = 0
        dataset_4_profile.tsg_overlap = 0

        """ Warning! Change this qty if using .shift() more then 2 """
        dsc = DSCreator(loaded_crypto_data, dataset_4_profile)
        dts_close1_close2_trend = dsc.create_dataset()

        binary_profile = NNProfile("binary_crossentropy")
        binary_profile.learning_rate = 1e-4
        binary_profile.experiment_name = f"{binary_profile.experiment_name}_close1_close2_trend"
        binary_profile.epochs = 350
        test_nn = MainNN(binary_profile)
        test_nn.train_model(dts_close1_close2_trend)
        # pred = test_nn.get_predict()
        # # print(pred)
        test_nn.show_categorical()
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




