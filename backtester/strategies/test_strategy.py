from backtrader import Strategy
import backtrader as bt
import backtrader.indicators as btind
import datetime
from .base import BaseStrategy


####################################################################
class MinMaxScaled(bt.Indicator):
    lines = ('scaled',)
    params = (('plot', False),)
    plotinfo = dict(plot=False)

    def __init__(self):
        self.plotinfo.plot = self.p.plot

    def next(self):
        self.min = min(getattr(self, 'min', self.data[0]), self.data[0])
        self.max = max(getattr(self, 'max', self.data[0]), self.data[0])
        self.lines.scaled[0] = (self.data[0] - self.min) / (self.max - self.min) \
            if self.max - self.min \
            else (self.data[0] - self.min) / self.max


####################################################################
class Diff(bt.Indicator):
    lines = ('diff',)

    def __init__(self):
        self.lines.diff = self.data0 - self.data1


####################################################################
class SMA(bt.Indicator):
    lines = ('source', 'sma',)
    params = (('period', 10),)
    plotinfo = dict(
        plotymargin=0.25,
    )

    def __init__(self):
        self.lines.source = self.data
        self.lines.sma = btind.SmoothedMovingAverage(self.data, period=self.p.period)


####################################################################
class Over(bt.Indicator):
    lines = ('over', )
    plotinfo = dict(
        plotymargin=0.25,
        # plothlines=[1.0, -1.0],
        plotyticks=[1.0, -1.0]
    )

    def __init__(self):
        self.lines.cross = btind.CrossOver(self.data0, self.data1)

    def next(self):
        if self.lines.cross[0] > 0 and self.lines.over[-1] != self.lines.cross[0]:
            self.lines.over[0] = 1
        elif self.lines.cross[0] < 0 and self.lines.over[-1] != self.lines.cross[0]:
            self.lines.over[0] = -1
        else:
            self.lines.over[0] = self.lines.over[-1]


####################################################################
# TestStrategy
####################################################################
class TestStrategy(BaseStrategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        period=100,
        target=0.5,
        coef=100,
        scale=False
    )

    def __init__(self):
        self.scaled_0 = MinMaxScaled(self.data0.close) if self.p.scale else self.data0.close
        self.scaled_1 = MinMaxScaled(self.data1.close) if self.p.scale else self.data1.close
        self.diff = abs(self.scaled_0 - self.scaled_1)
        # self.sma = SMA(self.diff, period=self.p.period)
        self.sma = btind.SmoothedMovingAverage(self.diff, period=self.p.period)
        # self.over_0 = self.data0 - self.sma
        # self.over_1 = self.data1 - self.sma
        # self.crossover = btind.CrossOver(self.diff, self.sma)
        self.cross = Over(self.diff, self.sma)
        # self.over_1 = Over(self.data1, self.sma)
        # self.over = Over(self.diff, self.sma)

    def next(self):
        print('--' * 30)
        print(self.position)
        print('--' * 30)
        print('--' * 30)
        print(self.data0.signal[-1])
        print(self.data0.signal[0])
        print('--' * 30)
        if not self.position:  # not in the market
            if self.data0.signal[0] == 1:
                self.buy(self.data0, size=1)
            if self.data1.signal[0] == 1:
                self.buy(self.data1, size=1)
        elif self.data0.signal[0] == -1:
            self.close(self.data0)
        elif self.data1.signal[0] == -1:
            self.close(self.data1)
        #     if self.cross > 0 and self.diff[0] > self.p.coef:  # if fast crosses slow to the upside
        #         size = 1 #int(self.broker.get_cash() / self.data0)
        #         self.buy(self.data0, size=size)  # enter long
        #     elif self.cross < 0 and self.diff[0] > self.p.coef:
        #         size = 1 #int(self.broker.get_cash() / self.data1)
        #         self.sell(self.data1, size=size)
        #
        # elif self.cross < 0:
        #     self.close(self.data0)
        #
        # elif self.cross > 0:
        #     self.close(self.data1)
