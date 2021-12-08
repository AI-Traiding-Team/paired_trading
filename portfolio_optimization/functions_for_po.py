import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.optimize as sco


def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns *weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, stocks):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets * [1. / num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]


def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0, 1.0)
    bounds = tuple(bound for asset in range(num_assets))

    result = sco.minimize(portfolio_volatility, num_assets * [1. / num_assets, ], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result


def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights, mean_returns, cov_matrix):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns, cov_matrix) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets * [1. / num_assets, ], args=args, method='SLSQP', bounds=bounds,
                          constraints=constraints)
    return result


def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients


def display_calculated_ef_with_random(closeData, stocks, mean_returns, cov_matrix, num_portfolios, risk_free_rate,
                                      plotting=False):
    results, _ = random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, stocks)

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sdp, rp = portfolio_annualised_performance(max_sharpe['x'], mean_returns, cov_matrix)
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x, index=closeData.columns, columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i * 100, 2) for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T

    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = portfolio_annualised_performance(min_vol['x'], mean_returns, cov_matrix)
    min_vol_allocation = pd.DataFrame(min_vol.x, index=closeData.columns, columns=['allocation'])
    min_vol_allocation.allocation = [round(i * 100, 2) for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T

    if plotting == True:
        print("=" * 80)
        print("Maximum Sharpe Ratio Portfolio Allocation")
        print("Annualised Return:", round(rp, 2))
        print("Annualised Volatility:", round(sdp, 2))
        print(max_sharpe_allocation)
        print("-" * 80)
        print("Minimum Volatility Portfolio Allocation")
        print("Annualised Return:", round(rp_min, 2))
        print("Annualised Volatility:", round(sdp_min, 2))
        print(min_vol_allocation)

        plt.figure(figsize=(10, 7))
        plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
        plt.colorbar()
        plt.scatter(sdp, rp, marker='*', color='r', s=500, label='Maximum Sharpe ratio')
        plt.scatter(sdp_min, rp_min, marker='*', color='g', s=500, label='Minimum volatility')
        target = np.linspace(rp_min, 0.6, 10)
        efficient_portfolios = efficient_frontier(mean_returns, cov_matrix, target)
        # print(efficient_portfolios)
        a = [p['fun'] for p in efficient_portfolios]
        plt.plot(a, target, linestyle='-.', color='black',
                 label='efficient frontier')
        plt.title('Calculated Portfolio Optimization based on Efficient Frontier')
        plt.xlabel('Annualised Volatility')
        plt.ylabel('Annualised Returns')
        plt.legend(labelspacing=0.8)
        plt.grid()
        plt.show()

    return max_sharpe_allocation, min_vol_allocation


def utility_optimal_portfolio(data, risk_aversion_coeff):
    # Importing libraries
    from pypfopt import EfficientFrontier
    from pypfopt import risk_models
    from pypfopt import expected_returns
    from pypfopt import objective_functions

    # Expected Returns
    mu = expected_returns.mean_historical_return(data)
    # Expected Volatility
    Sigma = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, Sigma)  # setup
    ef.add_objective(objective_functions.L2_reg)  # add a secondary objective
    weights = ef.max_quadratic_utility(risk_aversion=risk_aversion_coeff,
                                       market_neutral=False)  # find the portfolio that maximizes utility
    ret, vol, sharpe_r = ef.portfolio_performance(risk_free_rate=0.01125)
    # loop to iterate for values
    res = dict()
    for key in weights:
        # rounding to K using round()
        res[key] = round(weights[key], 2)

    print(f'Risk-Averse Investor (Инвестор, не склонный к риску):'
          f'\nAllocation:\t{res.values()}'
          f'\nAnnualised Reutrn:\t{round(ret, 2)}'
          f'\nAnnualised Volatility:\t{round(vol, 2)}'
          f'\nSharpe Ratio:\t{round(sharpe_r, 2)}')

    return res, round(ret, 2), round(vol, 2), round(sharpe_r, 2)