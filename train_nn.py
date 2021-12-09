from dataclasses import dataclass
from datamodeling import *
from analyze import DataLoad
from networks import *

__version__ = 0.0003

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
                                  time_intervals=['15m'],
                                  source_directory="/home/cubecloud/Python/projects/paired_trading/source_root",
                                  start_period='2021-11-01 00:00:00',
                                  end_period='2021-12-05 23:59:59',
                                  )

    dataset_1_profile = DSProfile()
    dsc = DSCreator(loaded_crypto_data, dataset_1_profile)
    dataset_1 = dsc.create_dataset()
    test_nn_profile = NNProfile()
    test_nn_profile.experiment_name = "test_NN_regression_ResNetV2"
    test_nn = MainNN(test_nn_profile)
    test_nn.train_model(dataset_1)



