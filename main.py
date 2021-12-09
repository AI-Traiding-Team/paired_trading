from analyze import DataLoad
from datetime import datetime
import os
import backtrader as bt
from strategies import TestStrategy
from backtrader_plotting import Bokeh, OptBrowser
from backtrader_plotting.schemes import Tradimo
import backtrader.analyzers as btanalyzers
import pyfolio as pf
import warnings
warnings.filterwarnings('ignore')

pairs = [
        # "BTCUSDT",
        "ETHUSDT",
         "BNBUSDT",
         "SOLUSDT",
         "ADAUSDT",
         "USDCUSDT",
         "XRPUSDT",
         "DOTUSDT",
         "LUNAUSDT",
         "DOGEUSDT",
         "AVAXUSDT"
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
database.get_all_close()
print(database.all_symbols_close)

# b = Bokeh(style='line', plot_mode='single', scheme=Tradimo())
# for item in database.pairs_symbols:
#     cerebro = bt.Cerebro()
#     print(item)
#     df = database.get_pair(item, intervals[0])
#     data = bt.feeds.PandasData(dataname=df, name=item)
#
#     cerebro.adddata(data)
#     # cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='mysharpe')
#     cerebro.addstrategy(TestStrategy)
#     cerebro.addanalyzer(bt.analyzers.PyFolio)
#     cerebro.addanalyzer(bt.analyzers.SharpeRatio)
#     cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
#     # cerebro.optstrategy(TestStrategy, pfast=range(5, 30, 60), pslow=(10, 20, 30))
#
#     result = cerebro.run(optreturn=True)
#     strat = result[0]
#     pyfoliozer = strat.analyzers.getbyname('pyfolio')
#     returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
#     # res = pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions,
#     #                           live_start_date='2021-09-01', round_trips=True)
#     # print(res)
#     cerebro.plot(b)
