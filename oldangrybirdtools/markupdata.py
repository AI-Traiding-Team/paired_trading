import numpy as np
from pandas import pandas as pd, DataFrame as df

from tensorflow.keras.activations import sigmoid, linear, relu, selu, tanh, softmax

# Scipy
from scipy.signal import argrelextrema, argrelmin, argrelmax

# System Libs
import re
import os
import io
import math
import datetime


class MarkUpData():
    def __init__(self, data):
        self.data = data
        self.source_columns = self.data.columns
        self.open, self.close, self.high, self.low = data['open'], data['close'], data['high'], data['low']
        self.activation_funcs = {
            'hist': lambda d: d.map(lambda x: ((0, 1)[x > 0], -1)[x < 0]),  # {'none': 0, 'buy': 1, 'sell': -1}
            'line': lambda x: x,
            'sigmoid': sigmoid,
            'linear': linear,
            'relu': relu,
            'selu': selu,
            'tanh': tanh,
            'softmax': softmax
        }

    def __trade_action(self, a, b):
        actions = {'none': 0, 'buy': 1, 'sell': -1}
        action = 'none'
        if a < b:
            action = 'buy'
        elif b < a:
            action = 'sell'
        return actions[action]

    def get_data(self, fill_na=None, drop_na=False):
        '''
        Возвращает DataFrame с данными.
        Заполняет пустые ячейки значением fill_na,
        либо удаляет строки с пустыми значениями, если drop_na=True
        '''
        if drop_na:
            self.data.dropna(inplace=True)
        elif fill_na is not None:
            self.data.fillna(fill_na, inplace=True)

        response = (self.data,)
        if hasattr(self, 'y'):
            response = (*response, self.y)
        if hasattr(self, 'source'):
            response = (*response, self.source)

        return response

    def use_optimizer(self, col_name, optimizer='Adam', **opt_params):
        '''
        Добавляет в датасет значения >>>>>
        '''

        def AdadeltaZeroStart(X, gamma, eps=0.001):
            return Adadelta(X, gamma, lr=0.0, eps=eps)

        def AdadeltaBigStart(X, gamma, eps=0.001):
            return Adadelta(X, gamma, lr=50.0, eps=eps)

        def Adam(X, beta1, beta2=0.999, lr=0.25, eps=0.0000001):
            Y = []
            m = 0
            v = 0
            for i, x in enumerate(X):
                m = beta1 * m + (1 - beta1) * x
                v = beta2 * v + (1 - beta2) * x * x
                m_hat = m / (1 - pow(beta1, i + 1))
                v_hat = v / (1 - pow(beta2, i + 1))
                dthetha = lr / np.sqrt(v_hat + eps) * m_hat
                Y.append(dthetha)
            return np.asarray(Y)

        def RMSProp(X, gamma, lr=0.25, eps=0.00001):
            Y = []
            EG = 0
            for x in X:
                EG = gamma * EG + (1 - gamma) * x * x
                v = lr / np.sqrt(EG + eps) * x
                Y.append(v)
            return np.asarray(Y)

        opt = {'Adam': Adam, 'AdadeltaBigStart': AdadeltaBigStart, 'AdadeltaZeroStart': AdadeltaZeroStart,
               'RMSProp': RMSProp}[optimizer]
        try:
            self.data[f'{optimizer}'] = opt(self.data[col_name].to_numpy(), **opt_params)
            return 200
        except Exception as error:
            return str(error)

    def drop_or_fill_na(self, fill_na=None, drop_na=False):
        '''
        Заполняет пустые ячейки значением fill_na,
        либо удаляет строки с пустыми значениями, если drop_na=True
        '''
        try:
            if drop_na:
                self.data.dropna(inplace=True)
            elif fill_na is not None:
                self.data.fillna(fill_na, inplace=True)
            return 200
        except Exception as error:
            return str(error)

    def drop_unused_cols(self, cols_names):
        '''
        Удаляет колонки из датасета по списку переданных имен
        type(col_names) = list
        '''
        try:
            self.data.drop(cols_names, axis=1, inplace=True)
            return 200
        except Exception as error:
            return str(error)

    def mark_sma(self, steps=[3, 4]):
        '''
        Добавляет в датасет значения Simple Moving Average для каждого из шагов перечисленных в steps
        '''
        try:
            for step in steps:
                self.data[f'sma_{step}'] = talib.SMA(self.close, step)
            return 200
        except Exception as error:
            return str(error)

    def mark_ema(self, steps=[3, 4]):
        '''
        Добавляет в датасет значения Exponential Moving Average для каждого из шагов перечисленных в steps
        '''
        try:
            for step in steps:
                self.data[f'ema_{step}'] = talib.EMA(self.close, step)
            return 200
        except Exception as error:
            return str(error)

    def mark_sar(self, acceleration=0.02, maximum=0.2):
        '''
        Добавляет в датасет значения Parabolic SAR
        '''
        try:
            self.data[f'sar'] = talib.SAR(self.high, self.low, acceleration, maximum)
            return 200
        except Exception as error:
            return str(error)

    def mark_cci(self, steps=[3, 4]):
        '''
        Добавляет в датасет значения Commodity Channel Index для каждого из шагов перечисленных в steps
        '''
        try:
            for step in steps:
                self.data[f'cci_{step}'] = talib.CCI(self.high, self.low, self.close, step)
            return 200
        except Exception as error:
            return str(error)

    def mark_bbands(self, ticks=None, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
        '''
        В tick передавать столбец, по которому нужно построить уровни Боллинджера
        Если ничего не передано, то строится по high
        '''
        if not ticks:
            ticks = self.high
        try:
            self.data['upperband'], self.data['middleband'], self.data['lowerband'] = talib.BBANDS(ticks, timeperiod,
                                                                                                   nbdevup, nbdevdn,
                                                                                                   matype)
            return 200
        except Exception as error:
            return str(error)

    def mark_extrems(self, col_name):
        '''
        В col_name передается название колонки, по которой необходимо найти экстремумы - точки перемены градиента
        '''
        try:
            x = argrelextrema(self.data[col_name].to_numpy(), np.greater)
            self.data.loc[list(*x), [f'{col_name}_extr']] = -1
            x = argrelextrema(self.data[col_name].to_numpy(), np.less)
            self.data.loc[list(*x), [f'{col_name}_extr']] = 1
            return 200
        except Exception as error:
            return str(error)

    def shift_data(self, col_names, steps):
        '''
        Сдвигает вперед (положительне значения steps - из прошлого в будущее) или
        назад (отрицательные значения steps - из будущего в прошлое) каждую колонку из списка col_names,
        на кол-во шагов для каждого значения из списка steps
        '''
        try:
            for col in col_names:
                for step in steps:
                    self.data[f'{col}_{step}'] = self.data[f'{col}'].shift(step)
            return 200
        except Exception as error:
            return str(error)

    def get_available_patterns(self):
        '''
        Возвращает список кодов паттернов для опредедления.
        '''
        return [item for item in talib.__TA_FUNCTION_NAMES__ if 'CDL' in item]

    def mark_patterns(self, patterns='all', bench_mark=0.1):
        '''
        Находит паттерны из списка patterns и создает колонки с указанием значений из диаппазона [-1, 0, 1] для каждого паттерна
        В датасет добавляются только паттерны, которые встречаются чаще, чем указано в bench_mark
        '''
        ext_cdl = []
        cdl = [item for item in talib.__TA_FUNCTION_NAMES__ if 'CDL' in item]
        if patterns != 'all':
            cdl = patterns
        for item in cdl:
            print
            col = getattr(talib, item)(self.open, self.high, self.low, self.close)
            if col[col != 0].count() >= self.data.shape[0] * bench_mark:
                self.data[item.lower()] = col / 100
            else:
                ext_cdl.append(item)

        cdl = [item for item in cdl if item not in ext_cdl]
        return cdl

    def cmp_to_benchmark(self, col_names, bench_marks, activation='tanh', round_to=0):
        '''
        Сравнивает значения каждой колонки из списка col_names с пороговыми значениями из списка bench_mark,
        применяет функцию активации из activation и округляет до round_to знаков
        Доступные функции активации:
          'hist': значения в диаппазоне [-1, 0, 1],
          'line': без изменений,
          'sigmoid': sigmoid,
          'linear': linear,
          'relu': relu,
          'selu': selu,
          'tanh': tanh,
          'softmax': softmax
        '''
        activation_func = self.activation_funcs[activation]
        try:
            for col in col_names:
                for bm in bench_marks:
                    self.data[f'{col}_{bm}'] = np.round(activation_func(self.data[f'{col}'] - bm), round_to)
            return 200
        except Exception as error:
            return str(error)

    def cmp_each_cols(self, col_names, activation='tanh', round_to=0):
        '''
        Сравнивает значения каждой с каждой колонок из списка col_names,
        применяет функцию активации из activation и округляет до round_to знаков
        Доступные функции активации:
          'hist': значения в диаппазоне [-1, 0, 1],
          'line': без изменений,
          'sigmoid': sigmoid,
          'linear': linear,
          'relu': relu,
          'selu': selu,
          'tanh': tanh,
          'softmax': softmax
        '''
        activation_func = self.activation_funcs[activation]
        try:
            for col_1 in col_names:
                for col_2 in col_names:
                    if col_1 == col_2:
                        continue
                    self.data[f'{col_1}_{col_2}'] = np.round(
                        activation_func(self.data[f'{col_1}'] - self.data[f'{col_2}']), round_to)
            return 200
        except Exception as error:
            return str(error)

    def cmp_col_to_cols(self, col_name, col_names, activation='tanh', round_to=0):
        '''
        Сравнивает значения каждой колонки col_name со значениями каждой колонки из списка col_names,
        применяет функцию активации из activation и округляет до round_to знаков
        Доступные функции активации:
          'hist': значения в диаппазоне [-1, 0, 1],
          'line': без изменений,
          'sigmoid': sigmoid,
          'linear': linear,
          'relu': relu,
          'selu': selu,
          'tanh': tanh,
          'softmax': softmax
        '''
        activation_func = self.activation_funcs[activation]
        try:
            for col in col_names:
                self.data[f'{col_name}_{col}'] = np.round(
                    activation_func(self.data[f'{col_name}'] - self.data[f'{col}']), round_to)
            return 200
        except Exception as error:
            return str(error)

    def calc_row_min(self, col_name, col_names, activation='line', absolut=True, round_to=10):
        '''
        Вычисляет минимальное значение построчно для значений из списка колонок col_names,
        применяет функцию активации из activation и округляет до round_to знаков.
        Результат сохраняется в колонке с именем col_name.
        absolut=True - определяет, что значение должно быть взято по модулю
        Доступные функции активации:
          'hist': значения в диаппазоне [-1, 0, 1],
          'line': без изменений,
          'sigmoid': sigmoid,
          'linear': linear,
          'relu': relu,
          'selu': selu,
          'tanh': tanh,
          'softmax': softmax
        '''
        activation_func = self.activation_funcs[activation]
        try:
            self.data[f'{col_name}_min'] = self.data.apply(
                lambda x: np.round(np.abs(activation_func(x[col_names].min())), round_to), axis=1)
            return 200
        except Exception as error:
            return str(error)

    def calc_row_max(self, col_name, col_names, activation='line', absolut=True, round_to=10):
        '''
        Вычисляет максимальное значение построчно для значений из списка колонок col_names,
        применяет функцию активации из activation и округляет до round_to знаков.
        Результат сохраняется в колонке с именем col_name.
        absolut=True - определяет, что значение должно быть взято по модулю
        Доступные функции активации:
          'hist': значения в диаппазоне [-1, 0, 1],
          'line': без изменений,
          'sigmoid': sigmoid,
          'linear': linear,
          'relu': relu,
          'selu': selu,
          'tanh': tanh,
          'softmax': softmax
        '''
        activation_func = self.activation_funcs[activation]
        try:
            self.data[f'{col_name}_max'] = self.data.apply(
                lambda x: np.round(np.abs(activation_func(x[col_names].max())), round_to), axis=1)
            return 200
        except Exception as error:
            return str(error)

    def get_col_names(self, reg=None):
        '''
        Возвращает названия всех колонок датасета, если указано значение reg,
        то возращаются только названия колонок удовлетворяющие купучз выражению в reg.
        '''
        if reg is None:
            return list(self.data.columns)
        else:
            return [item for item in self.data.columns if re.search(reg, item)]

    def separate_data(self, y_col, y_only=False):
        '''
        Функция разделяет датасет на части - отделяет y и отделяет колонки исходного датасета
        '''
        try:
            self.y = df(self.data[y_col].copy())
            self.data.drop(y_col, axis=1, inplace=True)
            if not y_only:
                self.source = self.data[self.source_columns].copy()
                self.data.drop(self.source_columns, axis=1, inplace=True)
        except Exception as error:
            return str(error)

    def save_to_csv(self, path, file_name, as_numpy=False):
        '''
        Сохраняет DataFrame с датасетом в csv файл датасет в папку path с именем file_name.
        При указании параметра as_numpy=True будет сохранен numpy массив.
        Если до сохранения была выполнена функция separate_data (разделение датасета на части),
        то будут сохранены файлы X_file_name, y_file_name и source_file_name&
        Файлы будут содержать размеченные X и н датасеты и колонки исходного датасета
        '''
        try:
            if hasattr(self, 'y'):
                y_file_name = f'y_{file_name}'
            if hasattr(self, 'source'):
                source_file_name = f'source_{file_name}'
            X_file_name = f'X_{file_name}'

            if as_numpy:
                np.savetxt(f'{path}/{X_file_name}', self.data.to_numpy(), delimetr=',')
                if y_file_name:
                    np.savetxt(f'{path}/{y_file_name}', self.y.to_numpy(), delimetr=',')
                if source_file_name:
                    np.savetxt(f'{path}/{source_file_name}', self.source.to_numpy(), delimetr=',')

            else:
                self.data.to_csv(f'{path}/{X_file_name}', sep=',')
                if y_file_name:
                    self.y.to_csv(f'{path}/{y_file_name}', sep=',')
                if source_file_name:
                    self.source.to_csv(f'{path}/{source_file_name}', sep=',')
            return 200
        except Exception as error:
            return str(error)

    def mark_datetime(self):
        '''
        Функция добавляет столбцы с днем недели, номером месяца, номером недели, номером дня в месяце
        '''
        try:
            date = pd.to_datetime(self.data['date'], format='%Y%m%d')
            self.data['week_day'] = date.apply(lambda x: x.weekday())
            self.data['month'] = date.apply(lambda x: x.month)
            self.data['week'] = date.apply(lambda x: x.week)
            self.data['day'] = date.apply(lambda x: x.day)
            return 200
        except Exception as error:
            return str(error)

    def rename_cols(self, cols_to_rename, names):
        '''
        Функция переименовывает столцы из списка cols_to_rename в имена из списка names.
        '''
        try:
            self.data.rename(columns=dict(zip(cols_to_rename, names)), inplace=True)
            return 200
        except Exception as error:
            return str(error)