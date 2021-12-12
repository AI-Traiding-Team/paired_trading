import optuna
from typing import List
from backtester import Back, Strategy
import pandas as pd
import time


class Objective(object):
    def __init__(self,
                 data: pd.DataFrame,
                 target_maker: object,
                 window_size: dict,
                 strategy: List[Strategy],
                 optimize_subj='Return [%]',
                 cash: float = 100_000.00):
        self.target_maker = target_maker
        self.window_size = window_size
        self.strategy = strategy
        self.data = data
        self.cash = cash
        self.optimize_subj = optimize_subj

    def __call__(self, trial):
        window_size = trial.suggest_int("window_size", **self.window_size)
        strategy = trial.suggest_categorical("strategy", self.strategy)
        data = self.target_maker(window_size).mark_y(self.data.copy(deep=False))
        bt = Back(data, strategy, cash=self.cash, commission=.002, trade_on_close=True)
        stats = bt.run()
        return stats[self.optimize_subj]


class BadNegroEvelOptimizer:
    def __init__(self, study_name, direction='maximize', optimize_subj='Return [%]', logoff=True, **kwargs):
        self.start = time.time()
        for key, val in kwargs.items():
            setattr(self, key, val)
        # Turn off optuna log notes.
        if logoff:
            optuna.logging.set_verbosity(optuna.logging.WARN)
        self.study_name = study_name
        self.direction = direction
        self.optimize_subj = optimize_subj
        self.study = optuna.create_study(study_name=study_name, direction=direction)
        self.best = {}

    def __logging_callback(self, study, frozen_trial):
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            mess = "Trial {} for study_name {} finished with best value: {} and parameters: {}. ".format(
                frozen_trial.number,
                study.study_name,
                frozen_trial.value,
                frozen_trial.params,
            )
            print(mess)
            self.best.update({study.study_name: {'subject': self.optimize_subj,
                                                 'direction': self.direction,
                                                 self.optimize_subj: frozen_trial.value,
                                                 'params': frozen_trial.params}
                              })

    def run(self):
        self.study.optimize(
            Objective(data=self.data, target_maker=self.target_maker,
                      window_size=self.window_size, strategy=self.strategy,
                      optimize_subj=self.optimize_subj, cash=self.cash),
                       n_trials=self.n_trials, callbacks=[self.__logging_callback])
        print('optimization done by: ', time.time() - self.start)
        return self.best
