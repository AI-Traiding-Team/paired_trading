import time
from analyze import DataLoad
import os
from backtester import Back
from backtester.strategies import *
from backtester.strategies.old_fashion import *
from optimizer import Objective, BadNegroEvelOptimizer
from maketarget import BigFatMommyMakesTargetMarkers
import json

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    intervals = ['1m']
    root_path = os.getcwd()
    source_root_path = os.path.join(root_path, 'source_root')

    database = DataLoad(pairs_symbols=None,
                        time_intervals=intervals,
                        source_directory=source_root_path,
                        start_period='2021-01-01 00:00:00',
                        end_period='2021-01-31 23:59:59'
                        )

    print(database.pairs_symbols)



    params = {'optimize_subj': 'Win Rate [%]', #Best Trade [%]', #'Return [%]',
              'n_trials': 200,
              'window_size': {'low': 5, 'high': 200, 'step': 2},
              'strategy': [LongStrategy, LongShortStrategy],
              'target_maker': BigFatMommyMakesTargetMarkers,
              'cash': 100_000}

    best = dict()
    for item in database.pairs_symbols:
        print(item)
        df = database.get_pair(item, intervals[0])
        best.update(**BadNegroEvelOptimizer(data=df, direction='maximize', study_name=item, **params).run())
    print(best)
    print(best, file=open('best_results_optimization.json', 'wt'))

