from backtesting import Strategy
from backtesting.lib import SignalStrategy


class BaseStrategy(Strategy):
    def _check_params(self, params):
        for k, v in params.items():
            setattr(self, k, v)
        return params


######################################################################
class SimpleSignalStrategy(SignalStrategy):
    def init(self):
        super().init()
        self.set_signal(self.data['Signal'] == 1, self.data['Signal'] == -1)


######################################################################
class LongStrategy(BaseStrategy):

    def init(self):
        self.signal = self.I(lambda x: x, self.data.Signal, name='Signal')

    def next(self):
        #ToDo добавить торговлю ограниченным лотом
        # торгуем по крайней цене закрытия
        price = self.data.Close[-1]

        if (self.position.is_long and
                self.signal == -1):
            self.position.close()

        if (self.position.size == 0 and
                self.signal == 1):
            self.buy()
            # self.buy(sl=0.8 * price)
            self.position.entry_price = price


######################################################################
class LongShortStrategy(BaseStrategy):

    def init(self):
        self.signal = self.I(lambda x: x, self.data.Signal, name='Signal')

    def next(self):
        # торгуем по крайней цене закрытия
        price = self.data.Close[-1]

        if (self.position.is_long and
                self.signal == -1):
            self.position.close()

        elif (self.position.is_short and
                self.signal == 1):
            self.position.close()

        elif (self.position.size == 0 and
                self.signal == 1):
            self.buy()
            # self.buy(sl=0.8 * price)
            self.position.entry_price = price

        elif (self.position.size == 0 and
                self.signal == -1):
            self.sell()
