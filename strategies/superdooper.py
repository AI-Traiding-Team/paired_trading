import pandas as pd


class SuperDooper():
    def __init__(self, window_size=5, remove_additional_data=True, convert_date_time_to_index=True):
        '''
        Если нужны дополнительные параметры, которые вычисляются при определении значений y,
        нужно поставить remove_additional_data=False

        По умолчанию колонки Дата-Время будут преобразованы в индекс DataFrame.
        Если не нужно преобразовывать Дату-Время в индекс, то указать convert_date_time_to_index=False
        '''
        self.prev = 0
        self.remove_additional_data = remove_additional_data
        self.convert_date_time_to_index = convert_date_time_to_index
        self.window_size = window_size

    def __filter_orders(self, item):
        self.prev
        if item == 0:
            return None
        elif item != 0 and item != self.prev:
            self.prev = item
            return item
        elif item != 0 and item == self.prev:
            self.prev = item
            return None

    def mark_y(self, data):
        '''
        На вход подается dataset в виде pandas DataFrame с колонками (минимально):
        'Close'
        Если необходимо конвертировать в индекс дату и время, то должны быть еще колонки:
        'Date', 'Time' в формате %Y%m%d и %H%M%S, пример, 20210228 и 113000
        '''

        data['min_long'] = data['close'].rolling(self.window_size, closed='left').min()
        data['max_long'] = data['close'].rolling(self.window_size, closed='left').max()
        data['min'] = data.apply(lambda x: (0, 1)[int(x['close'] == x['min_long'])], axis=1)
        data['max'] = data.apply(lambda x: (0, 1)[int(x['close'] == x['max_long'])], axis=1)
        data['openinterest'] = data['min'] - data['max']

        self.prev = 0
        data['openinterest'] = data['openinterest'][::-1].apply(self.__filter_orders)[::-1]
        data['openinterest'] = data['openinterest'].fillna(method='ffill')

        if self.remove_additional_data:
            data.drop(['min_long', 'max_long', 'min', 'max'], axis=1, inplace=True)

        return data