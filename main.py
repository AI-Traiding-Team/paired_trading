from analyze import DataLoad
from datetime import datetime
import os
import backtrader as bt
from strategies import TestStrategy
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo

pairs = ["BTCUSDT",
         "ETHUSDT",
         "BNBUSDT",
         # "SOLUSDT",
         # "ADAUSDT",
         # "USDCUSDT",
         # "XRPUSDT",
         # "DOTUSDT",
         # "LUNAUSDT",
         # "DOGEUSDT",
         # "AVAXUSDT"
         ]

intervals = ['15m']

root_path = os.getcwd()
source_root_path = os.path.join(root_path, 'source_root', '15min')
print(source_root_path)
database = DataLoad(pairs_symbols=pairs,
                    time_intervals=intervals,
                    source_directory=source_root_path,
                    start_period='2021-09-01 00:00:00',
                    end_period='2021-12-06 23:59:59'
                    )


b = Bokeh(style='bar', plot_mode='single', scheme=Tradimo())
for item in database.pairs_symbols:
    cerebro = bt.Cerebro()
    print(item)
    df = database.get_pair(item, intervals[0])
    data = bt.feeds.PandasData(dataname=df)
    data._name = item
    cerebro.adddata(data)
    cerebro.addstrategy(TestStrategy)
    stat = cerebro.run()
    cerebro.plot(b)
