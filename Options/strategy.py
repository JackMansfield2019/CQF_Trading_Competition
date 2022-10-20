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
    flag = True
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
            del row[0]
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
            print(row[1])
            #store the hourly price data, and hourly returns data.
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


    
    def d1(self,S,K,T,r,sigma):
        return(math.log(S/K)+(r+sigma**2/2.)*T)/(sigma*math.sqrt(T))
    def d2(self,S,K,T,r,sigma):
        return self.d1(S,K,T,r,sigma)-sigma*math.sqrt(T)

    """
    bs_call:
        Function that is responsible for calulatin the price of the call.

    args:
        #S: spot price
        #K: strike price
        #T: time to maturity
        #r: interest rate
        #sigma: volatility of underlying asset

    returns:
        Phat the estimated price of the option
    """

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
    '''
    '''
    def bs_call(self,S, K, T, r, sigma):
        N = norm.cdf
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * N(d1) - K * np.exp(-r*T)* N(d2)
    '''
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


    
    #start by making everything annualized. 
    def make_trades(self):

        #tuning variables:

        size_of_inital_data_set = 5.0 # the number of time steps we allow our algo to skip over while constrcuting an itial data set to use.
        p = 1.0 # p variable for GARCH model
        q = 1.0 # q variable for GARCH model
        min_traning_size_to_train_GARCH = 1000.0 # the minimum (# of timsteps passed)/size that the traning set needs to be to justify traning the model(
        min_traning_size_to_use_GARCH = 0.75
        #NOTE: these are different because we start traning the model when we have X traning size because there might be an option that we can use the 
        #      model but wether we use the model for a certain option depends on wether or not we have a traning size large enough to forecast out to that 
        #      specfific options expiry
        arbitrage_size_coefficent = 1.25 #price differnce there needs to be in order for there to be arbitrage


        trades = []

        if(len(self.hourly_returns) >= size_of_inital_data_set):
            #calulate volatility
            
            if len(self.hourly_returns) >= min_traning_size_to_train_GARCH:
                garch_model = arch_model(self.hourly_returns, p = 1, q = 1,mean = 'constant', vol = 'GARCH', dist = 'normal') 
                gm_result = garch_model.fit(disp='off')
                print(gm_result.params)
            else:
                #trivial volitlity grab the annualized volatility of the data set we have so far.
                sigma = math.sqrt(252) * statistics.pstdev(self.hourly_returns)
            
            #evaluate every option at this time step
            for i, row in enumerate(self.row):
                print(row[3]+"_"+row[2])
                print(row[1], " : ", row[2])
                print("underlying: ",(float(row[14])+float(row[13]))/2)
                print("option ask: ",float(row[12]))

                #GRAB Inputs to BLACK SCHOLES:

                #FIND S(Spot Price):
                S = (float(row[14])+float(row[13]))/2
                print("S:        ",S)

                #FIND K(strike Price):
                K = float(row[3])
                print("K:        ",K)

                #FIND T(time to expiry):
                #calculate the time until expiry of the option
                start = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")# ex row[1]: 2020-11-10 09:31:00
                end =   datetime.strptime(row[2] + " 23:59:59", "%Y-%m-%d %H:%M:%S") # ex row[2]: 2020-11-19
                steps_til_expiry = int((end-start).total_seconds() / 3600) #number of time steps until expiery
                #T               = ((end-start).total_seconds() / 3600) #Number of hours/ times steps
                T                = ((end-start).total_seconds() / 31536000) # annualized time til expiry
                print("T:        ",T)

                #FIND R(risk free rate of returns)
                R = 0.0181 # annualized return of the 10 year risk free bond
                print("R:        ",R)

                #FIND SIGMA(estimated variance):
                #determine wether or not to use GARCH for this option
                if len(self.hourly_returns) >= min_traning_size_to_use_GARCH * steps_til_expiry :
                    sigma = gm_result.forecast(horizon=steps_til_expiry)
                    print("used Garch, traning set ",len(self.hourly_returns)/steps_til_expiry," time larger than steps_til_expiry")
                print("SIGMA:    ",sigma)

                #use Black-scholes:
                
                phat = self.bs_call(S,K,T,R,sigma)
                
                print("Estimated Price: ", phat)

                #find & print the market price
                if(float(row[7]) != 0.0):
                    market_price = float(row[7])
                    print("Market close:    ",market_price)
                else:
                    market_price = float(row[12])
                    print("Market ask:      ",market_price)
                print()
            
            #determine if there is arbitrage
                if (phat >= arbitrage_size_coefficent*market_price):
                    asset_allocation = float(1000/market_price)
                    trades.append( (row[3]+"_"+row[2] , 0.0 , asset_allocation) )
                    self.counter += 1
                    

            '''
                every minute go thorugh potions 
                if the contract become fair value we sell it
                if contract is +/- 25% sell


                if contract is down 25% cut loss sell
                if contract is above 25% sell contract above it 
            '''
            print("COUNTER: ",self.counter)
            print()
        '''
        if (self.flag):
            self.flag = False
            return  [('underlying', 0.0, 285.9103)]
            
        else:
            return [('underlying', 0.0, 0.0)]
        '''
        return trades

            