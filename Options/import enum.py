import enum
import numpy as np 
import random
import pandas as pd 
import numpy as np 
#import Tkinter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import datetime
from datetime import datetime

import scipy.stats as si
import sympy as sy
from sympy.stats import Normal, cdf
from sympy import init_printing

import statistics
import pandas_datareader.data as web
from scipy.stats import norm
from arch import arch_model
from arch.__future__ import reindexing
"""
## Cornell Trading Competition : Derivatives Case Strategy ##

We have provided you with very basic skeleton code for you to implement your
strategy. The strategy will need to have *at least* two functions. One to read
the options data row-by-row, read_data, and one function that trades,
make_trades.

The Strategy is expected to work by rows: It reads the data for the rows of
market data corresponding for 1 minute, with all the different options being
traded at the time, with read_data() and the executes trades with make_trades().

Please do not modify the existing functions signature and make sure to actually
implement them as we will be using them in the grading of your submission.

This file, along with any other files necessary for the strategy to work, will
need to be submitted so make sure to add comments and make your code easy to
read/understand.

Plagarism is strictly banned and will result in your team being withdrawn from
the case. By writing your names below you agree to follow the code of conduct.

Please provide the name & emails of your team members:
    * Student Name (student@email.com)
    * ...

Best of Luck!
"""

class Strategy:
    start = True
    options_data =0


    hourly_prices = []
    hourly_returns = []
    hourly_predictions = []

    previous_hour =''

    row = 0
    counter = 0
    """
    read_data:
        Function that is responsible for providing the strategy with market data.

    args:
        row_vals - An array of array of strings that represents the different
        values in the rows of the raw csv file of the same format as the raw csv
        file provided to you. The outer array will correspond to an array of
        rows.

    returns:
        Nothing
    """
#"underlying_symbol","quote_datetime","root","expiration","strike","option_type","open","high","low","close","trade_volume","bid_size","bid","ask_size","ask","underlying_bid","underlying_ask","open_interest","level2"
#"underlying_symbol","quote_datetime","expiration","strike","open","high","low","close","trade_volume","bid_size","bid","ask_size","ask","underlying_bid","underlying_ask"
    
    def read_data(self, row_vals):
        #keep this row for use later
        self.row = row_vals
        length = len(self.hourly_prices)
        for i,row in enumerate(row_vals):
            #clean the data
            del row[len(row)-1]
            del row[len(row)-1]
            del row[5]
            del row[2]

            #constrcut pandas dataframe
            '''
            if self.start:
                tmp = {"underlying_symbol":[row[0]],"quote_datetime":[row[1]],"expiration":[row[2]],"strike":[row[3]],"open":[row[4]],"high":[row[5]],"low":[row[6]],"close":[row[7]],"trade_volume":[row[8]],"bid_size":[row[9]],"bid":[row[10]],"ask_size":[row[11]],"ask":[row[12]],"underlying_bid":[row[13]],"underlying_ask":[row[14]]}
                self.options_data = pd.DataFrame(tmp)
                self.start = False
            else:
                self.options_data.loc[len(self.options_data.index)] = row
            '''

            if row[1][11]+row[1][12] != self.previous_hour:
                self.hourly_prices.append((float(row[13])+float(row[14]))/2)
                if len(self.hourly_prices) != 1:
                    self.hourly_returns.append(math.log(self.hourly_prices[(length-1)+i]/self.hourly_prices[(length-2)+i]))
                self.previous_hour = row[1][11]+row[1][12]
        
        #print(self.options_data)
        #print(len(self.options_data))
        print()
        print(row[1])
        print(self.hourly_prices)
        print(self.hourly_returns)
        print()
        
        
        pass


    '''
    def d1(self,S,K,T,r,sigma):
        return(math.log(S/K)+(r+sigma**2/2.)*T)/(sigma*math.sqrt(T))
    def d2(self,S,K,T,r,sigma):
        return self.d1(S,K,T,r,sigma)-sigma*math.sqrt(T)
    def bs_call(self,S,K,T,r,sigma):
        return S*norm.cdf(self.d1(S,K,T,r,sigma))-K*math.exp(-r*T)*norm.cdf(self.d2(S,K,T,r,sigma))
    '''
    
    def bs_call(self,S, K, T, r, sigma):
        
        #S: spot price
        #K: strike price
        #T: time to maturity
        #r: interest rate
        #sigma: volatility of underlying asset
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        
        return call
    

    """
    make_trades:
        Function that tells the exchange whether, and at what quantity, the
        strategy wants to buy or sell, go long or short, the option.

    args:
        None

    returns:
        An array of triples (str(strike_expiry date), sell, buy) where sell, buy
        are float values that give how much quantity of contract options you
        want to sell and buy at bid & ask prices respectively and the strike
        price and expiry date to differentiate the option. Strike price is a
        numeric value and expiry date is a string and of the same format as in
        the raw csv file. Sell & buy may not be higher than ask & bid size at
        the given time. The value should be 0 for buy or sell if you want no
        activity on that side.

        You can buy/sell underlying stock by the same as above but rather than 
        the first element be str(strike)+str(expiry date) we have the word
        'underlying'
    """
    def make_trades(self):
        
        if(len(self.hourly_returns) > 3):
            yhat = math.sqrt(len(self.hourly_returns)) * statistics.pstdev(self.hourly_returns)
            for i, row in enumerate(self.row):
                #2020-11-10 09:31:00
                print(row[1])
                start = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
                print(row[2])
                end =   datetime.strptime(row[2] + " 23:59:00", "%Y-%m-%d %H:%M:%S")
                #r = (web.DataReader("^TNX", 'yahoo', start.replace(hour=start.hour-1),start)['Close'].iloc[-1])/100
                #r = (web.DataReader("^TNX", 'yahoo', start.replace(day=start.day-1), start)['Close'].iloc[-1])/100
                r = 0.0181
                print("r: ",r)
                
                t = ((end-start).total_seconds() / 3600)
                print("t: ",t)
                #t2 = (end-start).days / 365
                #print("t2: ",t2)
                
                if(float(row[7]) != 0.0):
                    print("close:")
                    phat = self.bs_call(float(row[7]),float(row[3]),t,r,yhat)
                    print("close: ",float(row[7]))
                    if (phat>1.25*float(row[7])):
                        self.counter +=1
                        tmp = ("C_"+row[3], 0.0, 1.0)
                else:
                    print(float(row[12]),float(row[3]),t,r,yhat)
                    phat = self.bs_call(float(row[12]),float(row[3]),t,r,yhat)
                    print("ask: ",float(row[12]))
                    if (phat>1.25*float(row[10])):
                        self.counter +=1
                print("phat: ",phat)
                
        
        '''
        if(len(self.hourly_returns) > 1):
            garch_model = arch_model(self.hourly_returns, p = 1, q = 1,mean = 'constant', vol = 'GARCH', dist = 'normal') 
            gm_result = garch_model.fit(disp='off')
            print(gm_result.params)
            
            #for each call in this time AKA each row.
            for i, row in enumerate(self.row):
                #2020-11-10 09:31:00
                date_format_str = '%d-%m-%Y %H:%M:%S'
                start = datetime.strptime(row[1], date_format_str)
                end =   datetime.strptime(row[2], date_format_str)
                steps_til_expiry = (start-end).total_seconds() / 3600
                if len(self.hourly_returns) >= 1.0 * steps_til_expiry:
                    yhat = model_fit.forecast(horizon=steps_til_expiry)
                    r = math.avg(self.hourly_returns)
                    self.bs_call(float(row[12]),float(row[3]),steps_til_expiry,r,yhat)
                    if(math.abs(row[12] - yhat)/row[12] >= 0.2):
                        print("current bid: "+row[12])
                        print("yhat: "+yhat)
                        print("diff: "+math.abs(row[12] - yhat)/row[12])
                '''
        #needs to be of the form: ('C_strike',sell,buy)
        #tmp = ('C_'+str(self.options_data.iat[2,3]) ,0.0,1.0)
        tmp = ('C_352.0', 0.0, 1.0)
        print(tmp)
        print("COUNTER: ",self.counter)
        return  [tmp]

        