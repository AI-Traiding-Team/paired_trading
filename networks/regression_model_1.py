from dataclasses import dataclass
from datamodeling import *
from analyze import DataLoad
from networks import *

__version__ = 0.0008

class TrainNN:
    def __init__(self):
        pass

    def train_model(self):
        """
        Model 1,
        Regression, close1-close2
        """
        dataset_1_profile = DSProfile()
        dataset_1_profile.timeframe = "1m"
        dsc = DSCreator(loaded_crypto_data, dataset_1_profile)
        dts_close1_close2 = dsc.create_dataset()
        regression_profile = NNProfile("regression")
        regression_profile.experiment_name = f"{regression_profile.experiment_name}_close1_close2"
        regression_profile.epochs = 40
        test_nn = MainNN(regression_profile)
        test_nn.train_model(dts_close1_close2)
        test_nn.get_predict()
        test_nn.show_regression()
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
    Model 1, 
    Regression, close1-close2
    """
    tr.train_model()




