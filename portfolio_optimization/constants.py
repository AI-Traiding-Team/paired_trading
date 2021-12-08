import os


path1 = "outputs"
path2 = "outputs/_imgs"
path3 = "outputs/max_sharpe_weights"
path4 = "outputs/opt_portfolio_trades"
try:
    os.mkdir(path1)
except OSError:
    print ("Директория %s уже создана" % path1)
else:
    print ("Успешно создана директория %s " % path1)

try:
    os.makedirs(path2)
    os.makedirs(path3)
    os.makedirs(path4)
except OSError:
    print ("Директории уже созданы")
else:
    print ("Успешно созданы нужные директории")


source_path = '../source_root/1min'
destination_path = 'outputs'