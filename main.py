from analyze import DataLoad
import os
import backtrader as bt
from strategies import TestStrategy
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
from backtester import Backtester

pairs = [
        # "BTCUSDT",
        #  "ETHUSDT",
         "BNBUSDT",
        #  "SOLUSDT",
         "ADAUSDT",
         # "USDCUSDT",
         # "XRPUSDT",
         # "DOTUSDT",
         # "LUNAUSDT",
         # "DOGEUSDT",
         # "AVAXUSDT"
         ]

intervals = ['1m']

root_path = os.getcwd()
source_root_path = os.path.join(root_path, 'source_root')
print(source_root_path)
database = DataLoad(
    pairs_symbols=pairs,
    time_intervals=intervals,
    source_directory=source_root_path,
    start_period='2021-09-01 00:00:00',
    end_period='2021-12-06 23:59:59'
)

trader = Backtester(TestStrategy, cash=100_000)
for item in database.pairs_symbols:
    print(item)
    df = database.get_pair(item, intervals[0])
    data = bt.feeds.PandasData(dataname=df, name=item)
    trader.adddata(data, name=item)
stat = trader.run(plot=True, plot_type='bokeh')
