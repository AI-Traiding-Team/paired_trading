from functions import *

path_filename ="../source_ds/1m/ETHUSDT-1m.csv"
dataset = MarkedDataSet(path_filename, None, df_priority=False)
tr = TrainNN(dataset, )
tr.epochs = 10
tr.tsg_batch_size = 128
tr.tsg_window_length = 40
tr.train()
test_df = tr.backtest_test_dataset()
# print(test_df.head().to_string(), f"\n")
# tr.figshow_base()
# tr.compile()
# tr.load_best_weights()
# tr.show_trend_predict()

