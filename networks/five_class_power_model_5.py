from dataclasses import dataclass
from datamodeling import *
from analyze import DataLoad
from networks import *

__version__ = 0.0002


class TrainNN:
    def __init__(self):
        pass

    def train_model(self):
        """
        Model 3,
        Five class classification, close1-close2 power - 5 classes
        """
        dataset_3_profile = DSProfile()
        dataset_3_profile.Y_data = "close1-close2_power"
        dataset_3_profile.timeframe = "1m"

        """ Default options for dataset window"""
        dataset_3_profile.tsg_window_length = 35
        dataset_3_profile.tsg_sampling_rate = 1
        dataset_3_profile.tsg_stride = 1
        dataset_3_profile.tsg_start_index = 0
        dataset_3_profile.tsg_overlap = 0
        """ Warning! Change this qty if using .shift() more then 2 """
        dataset_3_profile.gap_timeframes = 3

        dsc = DSCreator(loaded_crypto_data, dataset_3_profile)
        dts_close1_close2_power = dsc.create_dataset()
        categorical_profile = NNProfile("categorical_crossentropy")
        categorical_profile.experiment_name = f"{categorical_profile.experiment_name}_close1_close2_power"
        categorical_profile.epochs = 15
        categorical_profile.num_classes = 5
        test_nn = MainNN(categorical_profile)
        test_nn.train_model(dts_close1_close2_power)
        pred = test_nn.get_predict()

        print(pred)
        # test_nn.show_regression()
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
    Model 2, 
    Classification, close1-close2_trend
    """
    tr.train_model()




