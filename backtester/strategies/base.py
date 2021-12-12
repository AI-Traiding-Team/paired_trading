from backtrader import Strategy
import datetime
import pandas as pd

####################################################################
# BaseStrategy as a Template
####################################################################

class BaseStrategy(Strategy):
    def start(self):
        self.val_start = self.broker.get_cash()
        self.dtstart = datetime.datetime.now()
        print('Strategy calculation Start Time:            {}'.format(self.dtstart))

    def stop(self):
        dtstop = datetime.datetime.now()
        print('End Time:                    {}'.format(dtstop))

        strattime = (dtstop - self.dtstart).total_seconds()
        print('Total Time in Strategy:      {:.2f}'.format(strattime))
        print('Length of data feeds:        {}'.format(len(self.data)))

        self.roi = (self.broker.get_value() / self.val_start) - 1.0
        print('ROI:        {:.2f}%'.format(100.0 * self.roi))

        if getattr(self, 'log', False) is not False:
            # self.log['datetime'] = pd.to_datetime(self.log['datetime'])
            print(self.log)
