import os
from matplotlib import pyplot as plt
from sklearn import preprocessing
import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)

from portfolio_optimization.constants import source_path, destination_path



filenames = os.listdir(source_path)
print(filenames)

# соберем все Close в один датасет
tickers = []
df = pd.DataFrame()
df = pd.read_csv(f'{source_path}/{filenames[0]}')
# columns = df.columns.tolist()
df = df[['datetimeindex']]
df.set_index('datetimeindex', inplace=True)
df.sort_index(inplace=True)
df.index = pd.to_datetime(df.index)
for i, filename in enumerate(filenames):
    indices = [i for i, x in enumerate(filename) if x == "-"]  # находим индексы вхождения '-'
    ticker = filename[:indices[0]]
    tickers.append(ticker)
    df_temp = pd.read_csv(f'{source_path}/{filename}')
    df_temp.set_index('datetimeindex', inplace=True)
    df_temp.index = pd.to_datetime(df_temp.index)
    df = pd.merge(df, df_temp['close'], on=['datetimeindex'], how="left")
    df.rename({df.columns[-1]: ticker}, axis='columns', inplace=True)
df.dropna(inplace=True)  # начнем торговлю с момента появления всех монет
df.to_csv(f'{destination_path}/data.txt')
print(f'\nИсходные данные (цены Close):\n{df}\n')

# график нормализованной динамики курсов
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_norm = pd.DataFrame(x_scaled)
df_norm.index = df.index
df_norm.columns = tickers
print(f'\nнормализованная динамика курсов:\n{df_norm}\n')
df_norm.plot(figsize=(10, 5))
plt.legend(tickers)
plt.title(f'Сравнение нормализованной динамики курсов {len(filenames)-1} монет')
plt.grid()
plt.savefig(f'{destination_path}/_imgs/'
            f'Сравнение нормализованной динамики курсов {len(filenames)-1} монет.png')
plt.show()


df.pct_change().cumsum().plot(figsize=(10, 5))  # посчитаем прирост депозита
plt.legend(tickers)
plt.title(f'Сравнение доходностей {len(filenames)-1} монет')
plt.grid()
plt.savefig(f'{destination_path}/_imgs/'
            f'Сравнение доходностей {len(filenames)-1} монет.png')
plt.show()



