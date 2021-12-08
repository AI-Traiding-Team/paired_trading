# https://vc.ru/u/262921-aleksey-enin/152169-optimizaciya-investicionnogo-portfelya-po-metodu-markovica
# https://fin-plan.org/blog/investitsii/teoriya-portfelya-markovitsa/

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm, trange

np.random.seed(42)
import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)

import warnings
warnings.filterwarnings("ignore")

from functions_for_po import display_calculated_ef_with_random
from constants import destination_path



filename = 'data.txt'
df = pd.read_csv(f"{destination_path}/{filename}", header=0)
df.set_index('datetimeindex', inplace=True)
# df = df[-450:]
del df['USDCUSDT']
print(f'\nИсходные данные (цены Close):\n{df}\n')
stocks = list(df.columns.values)
num_stocks = len(stocks)
closeData = df[stocks]

# plot the heatmap and annotation on it
plt.figure(figsize=(14, 12))
Var_Corr = closeData.corr()
sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=False)
plt.savefig(f'{destination_path}/_imgs/heatmap.png')
plt.show()

#  Вычислим относительные изменения к предыдущему дню
dCloseData = closeData.pct_change()
dCloseData.index = closeData.index
dCloseData = dCloseData[1:]
mean_returns = dCloseData.mean()
cov_matrix = dCloseData.cov()
num_portfolios = 100
risk_free_rate = 0.01136  # The risk-free rate of return is the 10-year US Treasury Bond yield as on 04–02–2021 at 03:12 hours EST is 1.136%

# --------------------------------------------------------------------------------------
# Portfolio optimizaton - реализация форвардной торговлиЮ когда портфель собирается на прошлом,
# а торгуются будущие бары
# --------------------------------------------------------------------------------------

trade_window = 1  # окно торговли
print(f'Идет процесс форвардного анализа:\n'
      f'\tокно торговли = {trade_window} бар(а,ов)')

df_total_sum = pd.DataFrame()
for train_window in tqdm([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 45, 60, 120, 240]):  # окно, на котором создаем оптимальный портфель
    print(f'\n\tокно формирования портфеля = {train_window} бара(ов)')
    max_sharpe_weights = pd.DataFrame()
    df_trades = pd.DataFrame()

    for idx_start in trange(0, dCloseData.shape[0] - 1 - train_window, trade_window):

        idx_end = idx_start + train_window #- trade_window
        # print(f'idx_start: {idx_start},\tidx_end: {idx_end}')

        # Optimizing Portfolios based on Efficient Frontier
        _dCloseData = dCloseData[idx_start:idx_end]
        _mean_returns = _dCloseData.mean()
        _cov_matrix = _dCloseData.cov()
        max_sharpe_allocation, min_vol_allocation = display_calculated_ef_with_random \
            (_dCloseData, stocks, _mean_returns, _cov_matrix, num_portfolios, risk_free_rate)
        result = min_vol_allocation.T / 100.

        # Создание матрицы оптимальных портфелей
        colname = f'{idx_start}-{idx_end}'
        max_sharpe_weights = pd.concat([max_sharpe_weights, result], axis=1)\
            .rename(columns={'allocation': colname})  # Оптимальные веса за период train_window

        # Перераспределение средств между алгоритмами
        solution = pd.concat([result['allocation'], dCloseData[idx_end:idx_end+trade_window].T], axis=1)\
            .rename(columns={'allocation': colname})  #
        df_trades = pd.concat([df_trades, solution.product(axis=1)], axis=1)\
            .rename(columns={0: idx_end+1})  # Умножим веса алгоритмов на торговлю следующего дня
        df_trades.loc['total'] = df_trades.sum()

    df_trades = round(df_trades.T, 3)
    df_trades['total_sum'] = df_trades['total'].cumsum()  # Посчитаем сумму накопленным итогом
    max_sharpe_weights = round(max_sharpe_weights.T, 3)
    df_trades.to_csv(f'{destination_path}/opt_portfolio_trades/opt_portfolio_trades_train{train_window}_trade{trade_window}.txt')
    max_sharpe_weights.to_csv(f'{destination_path}/max_sharpe_weights/max_sharpe_weights_train{train_window}.txt')

    df_total_sum = df_total_sum.join(df_trades['total_sum'], how='outer').rename(
        columns={'total_sum': train_window}).fillna(method='ffill')
    if df_total_sum.shape[0] > dCloseData.shape[0]: df_total_sum = df_total_sum[:dCloseData.shape[0]]
    print(f'\n{df_total_sum[-1:]}')
df_total_sum.index = closeData.index[dCloseData.shape[0] - df_total_sum.shape[0] + 1:]
df_total_sum.to_excel(f'{destination_path}/df_total_sum.xlsx')

# отрисуем графики
df_total_sum[-1:].T.plot.bar(figsize=(10,5))
plt.title(f'Итоговые доходности в зависимости от `train_window`,   trade = {trade_window}')
plt.grid()
plt.savefig(f'{destination_path}/_imgs/Итоговые доходности {dCloseData.shape[1]} оптимизированным портфелем.png')
plt.show()

df_total_sum.plot(figsize=(10, 5))
plt.title(f'Динамика торговли оптимизированным портфелем из {dCloseData.shape[1]} алгоритмов\n'
          f'Легенда = Число дней обучения,   trade = {trade_window} бар')
plt.grid()
plt.savefig(f'{destination_path}/_imgs/Динамика торговли {dCloseData.shape[1]} оптимизированным портфелем.png')
plt.show()

max = df_total_sum[-1:].max(axis=1)
columns = list(df_total_sum.columns.values)
i, j = np.where(df_total_sum[-1:].values == max.iloc[0])
print(f'\nДоходность = {round(max.iloc[0], 2)}%  за период в {df_total_sum.shape[0] - j} бар(а,ов)\n'
      f'Оптимальное окно формирования портфеля = {columns[j[0]]} бара(ов)\n')

# дальше доделать анализ числа прибыльных и убыточных периодов и их средних размеров
# дальше можно смоделировать торговлю со стоп лосами и тейк профитами
