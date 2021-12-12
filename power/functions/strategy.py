# Библиоетки backtesting
from backtesting import Backtest, Strategy

#######################################################################
# Для покупки на фиксированную сумму в методе run класса Backtest нужно
# добавить два параметра:
# deal_amount = 'fix' - указывает, что покупать надо по фиксированной сумме
# fix_sum = XXXX - сумма покупки
# Если сумма покупки будет больше, чем накопленный капитал, возникнет ошибка
#######################################################################

# bt = Backtest(arr_ddd, LnS, cash=100000, commission=COMMISSION, trade_on_close=True)
# stats = bt.run(deal_amount='fix', fix_sum=1000)

class LongShort(Strategy):
    fix_sum = 0
    deal_amount = 'capital'

    def init(self):
        self.signal = self.I(lambda x: x, self.data.Signal, name='Signal')

    def next(self):
        # торгуем по крайней цене закрытия
        price = self.data.Close[-1]

        if (self.position.is_long and
                self.signal == -1):
            self.position.close()

        if (self.position.is_short and
                self.signal == 1):
            self.position.close()

        if (self.position.size == 0 and
                self.signal == 1):
            # Если backtest.run запущен с параметрами deal_amount='fix' и fix_sum=X
            # покупка будет осуществляться на фиксированную сумму, если сумма не превышает размер всего капитала
            if self.deal_amount == 'fix':
                if self.fix_sum // price:
                    if self.fix_sum // price <= self.equity // price:
                        self.buy(size=self.fix_sum // price)
                    else:
                        raise AttributeError(
                            'Фиксированная цена покупки превышает размер капитала!!! Купить не могу!!!')
            else:
                self.buy()
            self.position.entry_price = price

        if (self.position.size == 0 and
                self.signal == -1):
            self.sell()