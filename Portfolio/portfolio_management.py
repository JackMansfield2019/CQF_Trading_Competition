import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from backtest import backtest

def strat_function(preds, prices, optimized_weights):
    print(prices)
    returns = np.log(prices/prices-1)
    print(returns)

    def objective(weights):
        weights = np.array(weights)
        return weights.dot(returns.cov()).dot(weights.T)

    cons = ({"type":"eq","fun":lambda x: np.sum(x)-1},
            {"type":"ineq","fun":lambda x: np.sum(returns.mean()*x)-0.003})

    bounds = tuple((0,1) for x in range(returns.shape[1]))

    guess = [1./returns.shape[1] for x in range(returns.shape[1])]

    optimized_results = minimize(objective, guess, method="SLSQP", bounds=bounds, constraints=cons)
    optimized_weights = optimized_results.x
    return optimized_weights

backtest(strat_function,10000,'../Portfolio/price_data.csv', '../Portfolio/price_data.csv',True,"log.csv")