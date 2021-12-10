from analyze import DataLoad
import os
import backtrader as bt
from strategies import TestStrategy, SuperDooper
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
from backtester import Backtester

pairs = [
        # "BTCUSDT",
        #  "ETHUSDT",
         "BNBUSDT",
         "SOLUSDT",
        #  "ADAUSDT",
         # "USDCUSDT",
         # "XRPUSDT",
         # "DOTUSDT",
         # "LUNAUSDT",
         # "DOGEUSDT",
         # "AVAXUSDT"
         ]

intervals = ['1m']
pairs = ['SOLUSDT', 'BNBUSDT']

root_path = os.getcwd()
source_root_path = os.path.join(root_path, 'source_root')
print(source_root_path)

database = DataLoad(
    pairs_symbols=pairs,
    time_intervals=intervals,
    source_directory=source_root_path,
    start_period='2021-12-01 00:00:00',
    end_period='2021-12-06 23:59:59'
)

trader = Backtester(TestStrategy, cash=100_000, period=100)


for item in pairs:
    print(item)
    df, min_digit_len, max_digit_len = database.get_pair(item, intervals[0])
    df = SuperDooper().mark_y(df)
    print('Max / Min digits = ', max_digit_len, '/', min_digit_len)
    data = bt.feeds.PandasData(dataname=df, name=item)
    trader.adddata(data, name=item)
stat = trader.run(plot=True, plot_type='std')
# stat = trader.run(plot=True, plot_type='bokeh')
