#import subprocess
#import sys
#subprocess.check_call([sys.executable, "-m", "pip", "install", 'pyportfolioopt'])
import os
os.system('pip install pyportfolioopt')
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf

end = dt.date(2022,3,21)
start = dt.date(end.year - 5, end.month, end.day)
port = ['GE','MMM','WM']
tick = yf.Tickers(port)
df = tick.history(interval="1d",start=start, end=end)
portfolio = df['close']

mu = mean_historical_return(portfolio)
S = CovarianceShrinkage(portfolio).ledoit_wolf()

ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(dict(cleaned_weights)) # portfolio holdings

ef.portfolio_performance(verbose=True) # portfolio performance