import os
import json
import random
import operator
import numpy as np
import pandas as pd
from analyze import DataLoad
from powertrend import calc_power_trend

__version__ = 0.0009

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


class Featurizer:
    def __init__(self,
                 ohlcv_df: pd.DataFrame,
                 pair_symbol='',
                 timeframe='1m'
                 ):

        self.pair_symbol = pair_symbol
        self.timeframe = timeframe
        self.ohlcv_df = ohlcv_df
        self.ohlcv_df.columns = [item.lower().capitalize() for item in self.ohlcv_df.columns]
        self.features_df = self.ohlcv_df.copy()
        self.ohlcv_main_columns = ["Open", "High", "Low", "Close"]
        self.function_arg: any = 1
        self.temp = pd.Series()
        self.operations_dict: dict = {}

        self.operators_functions_dict: dict = {'log': np.log,
                                               'exp': np.exp,
                                               'sin': np.sin,
                                               'cos': np.cos,
                                               'sub': operator.sub,
                                               'div': operator.truediv,
                                               'mul': operator.mul,
                                               'add': operator.add,
                                               '_power_trend': calc_power_trend
                                               }
        self.pandas_functions_dict: dict = {}
        self.__dictionary_update()
        self.__preprocess_features()
        pass

    def __dictionary_update(self):
        self.pandas_functions_dict: dict = {'.diff': self.temp.diff(self.function_arg),
                                            '.mean': pd.DataFrame(self.temp).mean(axis=1),
                                            '.shift': self.temp.shift(self.function_arg),
                                            }
        pass

    def __preprocess_features(self):
        """
        (2) When λ (o) or λ (c) is equal to 0, it corresponds to x (o) = x (l) or x (c) = x (l) , respectively.
        We add a random term to x (o) or x (c) and make λ (o) or λ (c) slightly greater than 0.

        (3) When λ (o) or λ (c) is equal to 1, it indicates that x (o) = x (h) or x (c) = x (h) , respectively.
        We subtract a random term from x (o) or x (c) to make λ (o) or λ (c) slightly less than 1.
        """
        conditions_0 = (self.features_df['Open'] == self.features_df['Low']) | (
                    self.features_df['Close'] == self.features_df['Low'])

        self.features_df.loc[conditions_0, "Open"] += random.uniform(1e-8, 3e-8)
        self.features_df.loc[conditions_0, "Close"] += random.uniform(1e-8, 3e-8)

        conditions_1 = (self.features_df['Open'] == self.features_df['High']) | (
                    self.features_df['Close'] == self.features_df['High'])

        self.features_df.loc[conditions_1, "Open"] -= random.uniform(1e-8, 3e-8)
        self.features_df.loc[conditions_1, "Close"] -= random.uniform(1e-8, 3e-8)
        pass

    def __prepare_columns(self, function, cols_list):
        function_name = self.operators_functions_dict.get(function, None)
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
        def get_funct_and_arg(funct_arg: str):
            item_split = funct_arg.split('(')
            if len(item_split) == 1:
                _funct = item
                self.function_arg = 1
            else:
                _funct = item_split[0]
                _funct_arg = item_split[1][:-1]
                try:
                    self.function_arg = int(_funct_arg)
                except ValueError:
                    try:
                        self.function_arg = float(_funct_arg)
                    except ValueError:
                        print(f"Error: function argument error {_funct_arg}. Function argument has been set to 1")
                        self.function_arg = 1
            return _funct

        stack: list = []
        for ix, item in enumerate(operations_list):
            if isinstance(item, float) or isinstance(item, int):
                stack.append(item)
            elif isinstance(item, list) or isinstance(item, tuple):
                stack.append(self.features_df[item].copy())
            elif item[0].isupper():
                stack.append(self.features_df[item].copy())
            else:
                if item[0] == '_':
                    self.temp = stack.pop(-1)
                    item_funct = get_funct_and_arg(item)
                    operation = self.operators_functions_dict.get(item_funct, None)
                    assert operation is not None, f"Error: function doesn't exist in dictionary"
                    self.temp = operation(self.temp, self.function_arg)
                    stack.append(self.temp)
                elif item[0] == '.':
                    self.temp = stack.pop(-1)
                    item_funct = get_funct_and_arg(item)
                    self.__dictionary_update()
                    operation = self.pandas_functions_dict.get(item_funct, None)
                    assert operation is not None, f"Error: function doesn't exist in dictionary"
                    stack.append(operation)
                else:
                    operation = self.operators_functions_dict.get(item, None)
                    assert operation is not None, f"Error: function doesn't exist in dictionary"
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

    def run(self):
        assert self.operations_dict != dict(), "Error: operations dictionary not loaded"
        for col_name, operations in self.operations_dict.items():
            ftz.create_feature(operations, col_name)
        return self.features_df

    def load_features_set(self, path_filename, verbose=True):
        features_file = open(path_filename, "r")
        self.operations_dict = json.load(features_file)
        features_file.close()
        if verbose:
            print(self.operations_dict)
        pass

    def save_features_set(self, path_filename):
        features_file = open(path_filename, "w")
        json.dump(self.operations_dict, features_file)
        features_file.close()
        pass

    def save_dataframe(self, target_directory, save_file=True):
        if save_file:
            path_filename = os.path.join(target_directory, self.timeframe, f'{self.pair_symbol}-{self.timeframe}.csv')
            self.features_df.to_csv(path_filename)
        pass


if __name__ == "__main__":
    source_root_path = os.path.join('..', 'source_root')
    symbol = 'ETHUSDT'
    interval = '1m'
    loaded_crypto_data = DataLoad(pairs_symbols=[symbol],
                                  time_intervals=[interval],
                                  source_directory=source_root_path,
                                  start_period='2021-01-01 00:00:00',
                                  end_period='2021-12-01 23:59:59',
                                  # start_period='2021-09-22 00:00:00',
                                  # end_period='2021-11-30 23:59:59',
                                  # start_period='2021-09-01 00:00:00',
                                  # end_period='2021-12-01 23:59:59',
                                  )

    data_df = loaded_crypto_data.get_pair(symbol, interval)
    power_trend = 0.0275
    operations_dict = {
                       # "Cl_power_trend_mul_add": ["Close", "Close", 0.0275, "mul", "add"],
                       # "Cl_power_trend_mul_sub": ["Close", "Close", 0.0275, "mul", "sub"],
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
                       "Close_log_diff": ["Close_log", ".diff(1)"],
                       "Close_sin": ["Close", "sin"],
                       "Cl_log_cl_log_shift_div": ["Close_log", "Close_log", ".shift(1)", "div"],
                       "Volume_log": ["Volume", "log"],
                       "Signal": [["Open", "High", "Low", "Close"], "_power_trend(0.0275)"],
                       "OHLC_mean_log": [["Open", "High", "Low", "Close"], ".mean(1)", "log"],

                       # "L_log_l_log_shift_div": ["Low_log", "Low_log", ".shift", "div"],
                       # "H_log_H_log_shift_div": ["High_log", "High_log", ".shift", "div"],
                       # "1.0-cl_cl_shift_div_sub": [1.0, "Close", "Close", ".shift(1)", "div", "sub"],
                       # "Cl_log_cl_log_diff_div": ["Close_log", "Close_log", ".diff(1)", "sub"],

                       }

    ftz = Featurizer(data_df)
    # filename = os.path.join("sets", "test.json")
    ftz.operations_dict = operations_dict
    # ftz.load_features_set(filename)

    features_df = ftz.run()
    print(features_df.head(10).to_string())
