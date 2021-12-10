from backtrader import Strategy
import backtrader as bt
import backtrader.indicators as btind
import datetime
import pandas as pd

class SooperDooper(bt.Indicator):
    def __init__(self, window_size=5, remove_additional_data=True, convert_date_time_to_index=True):
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
        data['min_long'] = data['Close'].rolling(self.window_size, closed='left').min()
        data['max_long'] = data['Close'].rolling(self.window_size, closed='left').max()
        data['min'] = data.apply(lambda x: (0, 1)[int(x['Close'] == x['min_long'])], axis=1)
        data['max'] = data.apply(lambda x: (0, 1)[int(x['Close'] == x['max_long'])], axis=1)
        data['Signal'] = data['min'] - data['max']

        self.prev = 0
        data['Signal'] = data['Signal'][::-1].apply(self.__filter_orders)[::-1]
        data['Signal'] = data['Signal'].fillna(method='ffill')

        if self.remove_additional_data:
            data.drop(['min_long', 'max_long', 'min', 'max'], axis=1, inplace=True)

        return data

class Long_n_Short_Strategy(Strategy):

    def __init__(self):
        self.signal = SooperDooper(self.data)

    def next(self):
        # торгуем по крайней цене закрытия
        price = self.data.Close[-1]

        if (self.position.is_long and
                self.signal == -1):
            self.position.close()

        if (self.position.is_short and
                self.signal == 1):
            self.position.close()

        if (self.position.size == 0 and
                self.signal == 1):
            self.buy()
            # self.buy(sl=0.8 * price)
            self.position.entry_price = price

        if (self.position.size == 0 and
                self.signal == -1):
            self.sell()

    def stop(self):
        dtstop = datetime.datetime.now()
        print('End Time:                    {}'.format(dtstop))
        strattime = (dtstop - self.dtstart).total_seconds()
        print('Total Time in Strategy:      {:.2f}'.format(strattime))
        print('Length of data feeds:        {}'.format(len(self.data)))

        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        print('ROI:        {:.2f}%'.format(100.0 * self.roi))