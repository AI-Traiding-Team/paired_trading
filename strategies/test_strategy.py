from backtrader import Strategy
import backtrader as bt
import backtrader.indicators as btind
import datetime


# class MinMaxScaled(bt.Indicator):
#     lines = ('scaled',)
#     plotinfo = dict(plot=False)
#
#     def __init__(self):
#         self.min = min(self.data)
#         self.max = max(self.data)
#         self.lines.scaled = (self.data - self.min) / (self.max - self.min)


class MinMaxScaled(bt.Indicator):
    lines = ('scaled',)
    params = (('plot', False),)
    plotinfo = dict(plot=False)

    def __init__(self):
        self.plotinfo.plot = self.p.plot

    def next(self):
        self.min = min(getattr(self, 'min', self.data[0]), self.data[0])
        self.max = max(getattr(self, 'max', self.data[0]), self.data[0])
        self.lines.scaled[0] = (self.data[0] - self.min) / (self.max - self.min) if self.max - self.min else (self.data[0] - self.min) / self.max


class Diff(bt.Indicator):
    lines = ('diff',)

    def __init__(self):
        self.lines.diff = self.data0 - self.data1


class Signal(bt.Indicator):
    lines = ('signal', 'sma')
    params = (('period', 30),)

    def __init__(self):
        self.lines.sma = btind.SmoothedMovingAverage(self.data1, period=self.p.period)
        self.lines.signal = self.data - self.sma


class TestStrategy(Strategy):
    # list of parameters which are configurable for the strategy
    params = dict(
        period=100,
        target=0.5
    )

    def __init__(self):
        self.scaled_0 = MinMaxScaled(self.data0.close, plotname='MFI Close * 5.0')
        self.scaled_1 = MinMaxScaled(self.data1.close)
        self.diff = self.scaled_0 - self.scaled_1
        self.sma = btind.SmoothedMovingAverage(self.diff, period=self.p.period)
        # self.cross_up = btind.CrossUp(self.diff, self.sma)
        # self.cross_down = btind.CrossDown(self.diff, self.sma)
        self.cross_over = btind.CrossOver(self.diff, self.sma)

    def start(self):
        self.val_start = self.broker.get_cash()
        self.dtstart = datetime.datetime.now()
        print('Strat Start Time:            {}'.format(self.dtstart))

    def next(self):
        if not self.position:  # not in the market
            if self.cross_over > 0:  # if fast crosses slow to the upside
                size = int(self.broker.get_cash() / self.data0)
                self.buy(self.data0, size=size)  # enter long
            elif self.cross_over < 0:
                size = int(self.broker.get_cash() / self.data1)
                self.sell(self.data1, size=size)

        elif self.cross_over < 0:
            self.close(self.data0)
            self.close(self.data1)


    def stop(self):
        dtstop = datetime.datetime.now()
        print('End Time:                    {}'.format(dtstop))
        strattime = (dtstop - self.dtstart).total_seconds()
        print('Total Time in Strategy:      {:.2f}'.format(strattime))
        print('Length of data feeds:        {}'.format(len(self.data)))

        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        print('ROI:        {:.2f}%'.format(100.0 * self.roi))

