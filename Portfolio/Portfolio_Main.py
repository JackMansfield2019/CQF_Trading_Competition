!pip install yfinance

git clone https://github.com/Cornell-Quant-Fund/backtesting

import pandas_datareader.data as reader
import pandas as pd
import datetime as dt
import statsmodels.api as sm
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


def optimal_ports(port,start,end,min,plot):
  tick = yf.Tickers(port)
  df= tick.history(interval="1d",start=start, end=end)
  df = df['Close']

  cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()

  corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()

  ind_er = df.resample('1D').last().pct_change().mean()

  ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

  assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
  assets.columns = ['Returns', 'Volatility']

  p_ret = [] # Define an empty array for portfolio returns
  p_vol = [] # Define an empty array for portfolio volatility
  p_weights = [] # Define an empty array for asset weights

  num_assets = len(df.columns)
  num_portfolios = 10000

  for portfolio in range(num_portfolios):
      weights = np.random.random(num_assets)
      weights = weights/np.sum(weights)
      p_weights.append(weights)
      returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                        # weights 
      p_ret.append(returns)
      var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
      sd = np.sqrt(var) # Daily standard deviation
      ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
      p_vol.append(ann_sd)

  data = {'Returns':p_ret, 'Volatility':p_vol}

  for counter, symbol in enumerate(df.columns.tolist()):
      #print(counter, symbol)
      data[symbol+' weight'] = [w[counter] for w in p_weights]

  portfolios  = pd.DataFrame(data)
  portfolios.head() # Dataframe of the 10000 portfolios created
  

  if(min):   
    min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
    if(plot):
      plt.subplots(figsize=[10,10])
      plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
      plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
    # idxmin() gives us the minimum value in the column specified.                               
    return(min_vol_port)
  else:
    # Finding the optimal portfolio
    factors = reader.DataReader('F-F_Research_Data_Factors','famafrench',start,end)[0]
    #extract asset prices
    StockData = df.iloc[0:, 1:]

    #compute asset returns
    arStockPrices = np.asarray(StockData)
    [Rows, Cols]=arStockPrices.shape
    arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

    # Obtain optimal portfolio sets that maximize return and minimize risk

#Dependencies
import numpy as np
import pandas as pd

def mean_variance_optimization(w, V):
    def calculate_portfolio_risk(w, V):
        # function that calculates portfolio risk
        w = np.matrix(w)
        return np.sqrt((w * V * w.T)[0, 0])

    def calculate_portfolio_return(w, r):
        # function that calculates portfolio return
        return np.sum(w*r)

    # optimizer
    def optimize(w, V, target_return=0.1):
        init_guess = np.ones(len(symbols)) * (1.0 / len(symbols))
        weights = minimize(get_portfolio_risk, init_guess,
        args=(normalized_prices,), method='SLSQP',
        options={'disp': False},
        constraints=({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)},
        {'type': 'eq', 'args': (normalized_prices,),
        'fun': lambda inputs, normalized_prices:
        target_return - get_portfolio_return(weights=inputs,
        normalized_prices=normalized_prices)}))
        return weights.x
    optimal_risk_all = np.array([])
    optimal_return_all = np.array([])
    for target_return in np.arange(0.005, .0402, .0005):
        opt_w = optimize(prices=prices, symbols=symbols, target_return=target_return)
        optimal_risk_all = np.append(optimal_risk_all, get_portfolio_risk(opt_w, V))
        optimal_return_all = np.append(optimal_return_all, get_portfolio_return(opt_w, w))
    return optimal_return_all, optimal_risk_all


    ''''''
    rf = factors.tail()['RF'][:1].item()
    optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
    if(plot):
      plt.subplots(figsize=(10, 10))
      plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
      plt.scatter(optimal_risky_port[1], optimal_risky_port[0], color='g', marker='*', s=500)
    return(optimal_risky_port)
    '''

def tocsv(port,start,end):
  result = pd.DataFrame()
  for ticker in port:
    asset = yf.Ticker(ticker)
    hist_asset = asset.history(interval="1d",start=start,end=end)['Close']
    hist_asset= hist_asset[1:]
    result[ticker] = hist_asset
  return result


end = dt.date(2022,3,31)
start = dt.date(end.year-5,end.month,end.day)
port = ['GE','MMM','WM']
p = tocsv(port,start,end)
p.to_csv('price_data.csv')



def strat_function(preds, prices, last_weights): 
  start = dt.datetime(2017,3,31)
  start_stamp = start.timestamp()
  new_end = dt.datetime.fromtimestamp(start_stamp + 86400 * len(prices))
  
  if len(prices) % 180 != 0:
    return last_weights
  else:
    port = ['GE','MMM','WM']
    print(start)
    print(new_end)
    print(port)
    sharpe_port = optimal_ports(port,start,new_end,False,False)
    print(sharpe_port[2:])
  return sharpe_port[2:]

from backtesting.backtest_v2.backtest import backtest
backtest(strat_function, 10000, 'price_data.csv', 'price_data.csv', True, "log.csv")