import backtrader as bt
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
from backtesting.lib import SignalStrategy
from backtesting import Backtest, Strategy
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import pandas as pd



class Backtester(bt.Cerebro):
    def __init__(self, strategy, cash=100_000, commission=0.002, **kwargs):
        super().__init__()
        self.bokeh = Bokeh(style='bar', plot_mode='single', scheme=Tradimo())
        self.broker.setcash(cash)
        self.broker.setcommission(commission=commission)
        period = kwargs.get('period', 20)
        self.addstrategy(strategy, period=period)

    def run(self, plot=False, plot_type='standard'):
        print('Starting Portfolio Value: %.2f' % self.broker.getvalue())
        super().run()
        print('Final Portfolio Value: %.2f' % self.broker.getvalue())
        if plot:
            if plot_type == 'bokeh':
                self.plot(self.bokeh)
            else:
                self.plot(style='candlestick')


class Back(Backtest):
    def __init__(self,
                 data: pd.DataFrame,
                 strategy: Type[Strategy],
                 *args,
                 cash: float = 10_000,
                 commission: float = .0,
                 margin: float = 1.,
                 trade_on_close=False,
                 hedging=False,
                 exclusive_orders=False
                 ):
        data.columns = [item.lower().capitalize() for item in data.columns]
        super().__init__(
            data, strategy, cash=cash, *args, commission=commission, margin=margin,
            trade_on_close=trade_on_close, hedging=hedging, exclusive_orders=exclusive_orders
        )