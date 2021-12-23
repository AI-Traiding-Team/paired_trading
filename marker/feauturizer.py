import os
import random
import numpy as np
import pandas as pd
from analyze import DataLoad
import math
import operator

__version__ = 0.0004

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


class Featurizer:
    def __init__(self, ohlcv_df: pd.DataFrame):
        self.ohlcv_df = ohlcv_df
        self.ohlcv_df.columns = [item.lower().capitalize() for item in self.ohlcv_df.columns]
        self.features_df = self.ohlcv_df.copy()
        self.ohlcv_main_columns = ["Open", "High", "Low", "Close"]
        self.temp = pd.DataFrame()
        self.prefix_functions_dict: dict = {}
        self.__dictionary_update()
        self.__preprocess_features()
        pass

    def __dictionary_update(self):
        self.prefix_functions_dict: dict = {'log': np.log,
                                            'exp': np.exp,
                                            'sin': np.sin,
                                            'cos': np.cos,
                                            'sub': operator.sub,
                                            'div': operator.truediv,
                                            'mul': operator.mul,
                                            'add': operator.add,
                                            '.diff': self.temp.diff(),
                                            '.mean': pd.DataFrame(self.temp).mean(axis=1),
                                            '.shift': self.temp.shift(1),
                                            }
        pass

    def __preprocess_features(self):
        """
        (2) When λ (o) or λ (c) is equal to 0, it corresponds to x (o) = x (l) or x (c) = x (l) , respectively.
        We add a random term to x (o) or x (c) and make λ (o) or λ (c) slightly greater than 0.

        (3) When λ (o) or λ (c) is equal to 1, it indicates that x (o) = x (h) or x (c) = x (h) , respectively.
        We subtract a random term from x (o) or x (c) to make λ (o) or λ (c) slightly less than 1.
        """
        conditions_0 = (self.features_df['Open'] == self.features_df['Low']) | (self.features_df['Close'] == self.features_df['Low'])
        self.features_df.loc[conditions_0, "Open"] += random.uniform(1e-8, 3e-8)
        self.features_df.loc[conditions_0, "Close"] += random.uniform(1e-8, 3e-8)

        conditions_1 = (self.features_df['Open'] == self.features_df['High']) | (self.features_df['Close'] == self.features_df['High'])
        self.features_df.loc[conditions_1, "Open"] -= random.uniform(1e-8, 3e-8)
        self.features_df.loc[conditions_1, "Close"] -= random.uniform(1e-8, 3e-8)
        pass

    def __prepare_columns(self, function, cols_list):
        function_name = self.prefix_functions_dict.get(function, None)
        assert function_name is not None, f"Error: function doesn't exist in dictionary"
        for col in cols_list:
            self.features_df[f"{function_name}_{col}"] = function(self.features_df[col])
        pass

    def __do_operation(self, stack):
        if len(stack) >= 3:
            result_df = stack[-1](stack[-3], stack[-2])
            stack = stack[:-3]
            stack.append(result_df)
        elif len(stack) == 2:
            result_df = stack[1](stack[0])
            stack = [result_df]
        return stack

    def __stack_operations(self, operations_list):
        stack: list = []
        for ix, item in enumerate(operations_list):
            if isinstance(item, float) or isinstance(item, int):
                stack.append(item)
            elif isinstance(item, list) or isinstance(item, tuple):
                stack.append(self.features_df[item].copy())
            elif item[0].isupper():
                stack.append(self.features_df[item].copy())
            else:
                operation = self.prefix_functions_dict.get(item, None)
                assert operation is not None, f"Error: function doesn't exist in dictionary"
                if item[0] == '.':
                    self.temp = stack.pop(-1)
                    self.__dictionary_update()
                    operation = self.prefix_functions_dict.get(item, None)
                    stack.append(operation)
                else:
                    stack.append(operation)
                    stack = self.__do_operation(stack)
        return stack[0]

    def create_feature(self, operations_list, col_name=''):
        """
        Args:
            operations_list (list): stack of values and operations for preparing
            col_name (str):         name of the feature

        Example:
            Explanation. Name of df column _must_ have Capitalization, all operands must have the lowercase
            or '.' for pandas class operations

            operations_list = ["Low", "log"]
            operations_list = ["Low", ".diff"]
            operations_list = ["Low", ".shift"]
            operations_list = ["High", "Low", "sub", "log"]
            operations_list = ["High", "Low", "sub", "log"]
            operations_list = ["Open", "Low", "sub", "High", "Low", "sub", "div"]
        """

        if not col_name:
            for item in operations_list:
                col_name += f'{item[0]}_'
            col_name = col_name[:-1]
        self.features_df[col_name] = self.__stack_operations(operations_list)
        pass


if __name__ == "__main__":
    source_root_path = os.path.join('..', 'source_root')
    pair_symbol = 'ETHUSDT'
    timeframe = '1m'
    loaded_crypto_data = DataLoad(pairs_symbols=[pair_symbol],
                                  time_intervals=[timeframe],
                                  source_directory=source_root_path,
                                  start_period='2021-01-01 00:00:00',
                                  end_period='2021-12-01 23:59:59',
                                  # start_period='2021-09-22 00:00:00',
                                  # end_period='2021-11-30 23:59:59',
                                  # start_period='2021-09-01 00:00:00',
                                  # end_period='2021-12-01 23:59:59',
                                  )

    data_df = loaded_crypto_data.get_pair(pair_symbol, timeframe)

    operations_dict = {
                       "Lambda_open": ["Open", "Low", "sub", "High", "Low", "sub", "div"],
                       "Lambda_close": ["Close", "Low", "sub", "High", "Low", "sub", "div"],
                       "Y_1_1": ["Low", "log"],
                       "Y_1_2": ["High", "Low", "sub", "log"],
                       "Y_1_3": ["Lambda_open", 1.0, "Lambda_open", "sub", "div", "log"],
                       "Y_1_4": ["Lambda_close", 1.0, "Lambda_close", "sub", "div", "log"],

                       "Open_log": ["Open", "log"],
                       "High_log": ["High", "log"],
                       "Low_log": ["Low", "log"],
                       "Close_log": ["Close", "log"],
                       "Close_log_diff": ["Close_log", ".diff"],
                       "Close_sin": ["Close", "sin"],
                       "Cl_log_cl_log_shift_div": ["Close_log", "Close_log", ".shift", "div"],
                       "OHLC_mean_log": [["Open", "High", "Low", "Close"], ".mean", "log"],
                       "Volume_log": ["Volume", "log"],
                       }

    ftz = Featurizer(data_df)
    for col_name, operations in operations_dict.items():
        ftz.create_feature(operations, col_name)

    print(ftz.features_df.head().to_string())
