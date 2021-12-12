import backtrader as bt
import backtrader.indicators as btind


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