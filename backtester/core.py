import backtrader as bt
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo


class Backtester(bt.Cerebro):
    def __init__(self, strategy, cash=100_000, commission=0.002):
        super().__init__()
        self.bokeh = Bokeh(style='bar', plot_mode='single', scheme=Tradimo())
        self.broker.setcash(cash)
        self.broker.setcommission(commission=commission)
        self.addstrategy(strategy, period=300)

    def run(self, plot=False, plot_type='standard'):
        print('Starting Portfolio Value: %.2f' % self.broker.getvalue())
        super().run()
        print('Final Portfolio Value: %.2f' % self.broker.getvalue())
        if plot:
            bokeh = self.bokeh if plot_type == 'bokeh' else None
            self.plot(bokeh, style='candlestick')
