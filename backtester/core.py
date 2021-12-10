import backtrader as bt
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo


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
