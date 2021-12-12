import backtrader as bt
import backtrader.indicators as btind
from datetime import datetime, timedelta
from backtrader import num2date
from . import BaseStrategy
from backtester.indicators import *
from pandas import DataFrame as df

class TestSizer(bt.Sizer):
    params = dict(stake=1)

    def _getsizing(self, comminfo, cash, data, isbuy):
        dt, i = self.strategy.datetime.date(), data._id
        s = self.p.stake * (1 + (not isbuy))
        print('{} Data {} OType {} Sizing to {}'.format(
            dt, data._name, ('buy' * isbuy) or 'sell', s))

        return s



class SKilStrategy(BaseStrategy):
    params = dict(
        period=20,
        target=0.5,
        coef=2.5,
        scale=False,
        enter=[1, 3, 4],  # data ids are 1 based
        hold=[7, 10, 15],  # data ids are 1 based
        usebracket=True,
        rawbracket=True,
        pentry=0.015,
        plimits=0.03,
        valid=10,
    )

    def notify_order(self, order):
        if order.status == order.Submitted:
            return

        dt, dn = self.datetime.date(), order.data._name
        print('{} {} Order {} Status {}'.format(
            dt, dn, order.ref, order.getstatusname())
        )

        whichord = ['main', 'stop', 'limit', 'close']
        if not order.alive():  # not alive - nullify
            dorders = self.o[order.data]
            idx = dorders.index(order)
            dorders[idx] = None
            print('-- No longer alive {} Ref'.format(whichord[idx]))

            if all(x is None for x in dorders):
                dorders[:] = []  # empty list - New orders allowed

    def __init__(self):
        # self.scaled_0 = MinMaxScaled(self.data0.close) if self.p.scale else self.data0.close
        # self.scaled_1 = MinMaxScaled(self.data1.close) if self.p.scale else self.data1.close
        for i, d in enumerate(self.datas):
            print(i, ': ', d._name)

        self.diff = self.data0 - self.data1
        self.sma = btind.SmoothedMovingAverage(self.diff, period=self.p.period)
        self.signal = self.diff - self.sma
        self.log = df(columns=['datetime', 'close_0', 'close_1', 'diff', 'sma', 'signal'])
        self.o = dict()  # orders per data (main, stop, limit, manual-close)
        self.holding = dict()  # holding periods per data

    def _log_next(self):
        log_data = {
                'datetime': num2date(self.data.datetime[0]),
                'close_0': self.data0[0],
                'close_1': self.data1[0],
                'diff': self.diff[0],
                'sma': self.sma[0],
                'signal': self.signal[0]
        }
        self.log = self.log.append(log_data, ignore_index=True)

    def next(self):
        # print('--' * 30)
        # print(self.position)
        # print('--' * 30)
        # print('--' * 30)
        # print('Diff = ', self.diff[-1], ' - ', self.diff[0])
        # print('sma = ', self.sma[-1], ' - ', self.sma[0])
        # print('--' * 30)
        # self._log_next()
        # if not self.position:  # not in the market
        #     if self.signal > self.p.coef:
        #         # print(self.data0[0], self.data1[0], self.diff[0], self.sma[0], self.signal[0])
        #         self.buy(self.data0, size=1)
        #         self.sell(self.data1, size=1)
        #
        # elif self.signal < 0:
        #     self.close(self.data0)
        #     self.close(self.data1)
         for i, d in enumerate(self.datas):
                    dt, dn = self.datetime.date(), d._name
                    pos = self.getposition(d).size
                    print('{} {} Position {}'.format(dt, dn, pos))

                    if not pos and not self.o.get(d, None):  # no market / no orders
                        if dt.weekday() == self.p.enter[i]:
                            if not self.p.usebracket:
                                self.o[d] = [self.buy(data=d)]
                                print('{} {} Buy {}'.format(dt, dn, self.o[d][0].ref))

                            else:
                                p = d.close[0] * (1.0 - self.p.pentry)
                                pstp = p * (1.0 - self.p.plimits)
                                plmt = p * (1.0 + self.p.plimits)
                                valid = timedelta(self.p.valid)

                                if self.p.rawbracket:
                                    o1 = self.buy(data=d, exectype=bt.Order.Limit,
                                                  price=p, valid=valid, transmit=False)

                                    o2 = self.sell(data=d, exectype=bt.Order.Stop,
                                                   price=pstp, size=o1.size,
                                                   transmit=False, parent=o1)

                                    o3 = self.sell(data=d, exectype=bt.Order.Limit,
                                                   price=plmt, size=o1.size,
                                                   transmit=True, parent=o1)

                                    self.o[d] = [o1, o2, o3]

                                else:
                                    self.o[d] = self.buy_bracket(
                                        data=d, price=p, stopprice=pstp,
                                        limitprice=plmt, oargs=dict(valid=valid))

                                print('{} {} Main {} Stp {} Lmt {}'.format(
                                    dt, dn, *(x.ref for x in self.o[d])))

                            self.holding[d] = 0

                    elif pos:  # exiting can also happen after a number of days
                        self.holding[d] += 1
                        if self.holding[d] >= self.p.hold[i]:
                            o = self.close(data=d)
                            self.o[d].append(o)  # manual order to list of orders
                            print('{} {} Manual Close {}'.format(dt, dn, o.ref))
                            if self.p.usebracket:
                                self.cancel(self.o[d][1])  # cancel stop side
                                print('{} {} Cancel {}'.format(dt, dn, self.o[d][1]))