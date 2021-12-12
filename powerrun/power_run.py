from functions import *

path_filename ="../source_ds/1m/ETHUSDT-1m.csv"
dataset = MarkedDataSet(path_filename)
tr = TrainNN(dataset)
tr.train()
tr.figshow_base()
# tr.compile()
# tr.load_best_weights()
tr.show_trend_predict()
tr.check_binary()
