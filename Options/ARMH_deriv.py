import enum
import numpy as np 
import random
import pandas as pd 
#import Tkinter
import matplotlib as mpl

import matplotlib.pyplot as plt
import math
import datetime
from datetime import datetime



import py_vollib 
from py_vollib.black_scholes  import black_scholes as bs
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
from py_vollib.black_scholes.greeks.analytical import delta as delta


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

    minute_prices = []
    minute_returns = []

    hourly_prices = []
    hourly_returns = []
    hourly_predictions = []

    previous = ''
    previous_hour =''

    training_counter = 0
    row = 0
    counter = 0
    counter2 = 0
    flag = True
    portfolio = {}
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
        self.row = []

        for i,row in enumerate(row_vals):
            #clean the data

            del row[len(row)-1]
            del row[len(row)-1]
            del row[5]
            del row[2]
            
            if float(row[10]) <= 0.05 or float(row[12]) == 0.01 and float(row[9] == 0.0): continue

            self.row.append(row)
            

            #constrcut pandas dataframe
            if row[1] != self.previous:
                self.minute_prices.append((float(row[13])+float(row[14]))/2)
                if len(self.minute_prices) != 1:
                    self.minute_returns.append(math.log(self.minute_prices[-1]/self.minute_prices[-2]))
                self.previous = row[1]
            #store the hourly price data, and hourly returns data.
            if row[1][11]+row[1][12] != self.previous_hour:
                self.hourly_prices.append((float(row[13])+float(row[14]))/2)
                if len(self.hourly_prices) != 1:
                    self.hourly_returns.append(math.log(self.hourly_prices[-1]/self.hourly_prices[-2]))
                self.previous_hour = row[1][11]+row[1][12]

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
    
    def bs_call(self,S, K, T, r, sigma):

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
        
        return call

    def round_sig(self,x, sig=2):
        print(x)
        tmp = abs(x)
        return round(x, sig-int(math.floor(math.log10(tmp)))-1)


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
        p = 1.0 # p variable for GARCH model
        q = 1.0 # q variable for GARCH model
        size_of_inital_data_set = 1000*(p+q) # the number of time steps we allow our algo to skip over while constrcuting an itial data set to use.
        min_traning_size_to_train_GARCH = 1000*(p+q) # the minimum (# of timsteps passed)/size that the traning set needs to be to justify traning the model(
        min_traning_size_to_use_GARCH = 1
        #NOTE: these are different because we start traning the model when we have X traning size because there might be an option that we can use the 
        #      model but wether we use the model for a certain option depends on wether or not we have a traning size large enough to forecast out to that 
        #      specfific options expiry
        arbitrage_size_coefficent = 0.1 #price differnce there needs to be in order for there to be arbitrage

        volitlity_calulations = {}
        trades = []
        self.counter2 += 1
        if (self.counter2 > 200):
            return [( 'underlying' , 1000000 , 0.0) ]

        if(len(self.minute_returns) >= 0):
            
            if len(self.minute_returns) >= 100000:
                self.training_counter +=1
                garch_model = arch_model(self.minute_returns, p = 1, q = 1,mean = 'constant', vol = 'GARCH', dist = 'normal') 
                gm_result = garch_model.fit(disp='off')
                #print(gm_result.params)
            
            #evaluate every option at this time step
            g = 0
            for i, row in enumerate(self.row):
                #print(row[3]+"_"+row[2])
                #print(row[1], " : ", row[2])
                #print("underlying: ",(float(row[14])+float(row[13]))/2)
                #print("option ask: ",float(row[12]))

                #GRAB Inputs to BLACK SCHOLES:

                #FIND S(Spot Price):
                S = (float(row[14])+float(row[13]))/2
                #print("S:        ",S)

                #FIND K(strike Price):
                K = float(row[3])
                #print("K:        ",K)

                #FIND T(time to expiry):
                #calculate the time until expiry of the option
                start = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")# ex row[1]: 2020-11-10 09:31:00
                end =   datetime.strptime(row[2] + " 23:59:59", "%Y-%m-%d %H:%M:%S") # ex row[2]: 2020-11-19
                steps_til_expiry = int((end-start).total_seconds() / 60) #number of time steps until expiery
                #steps_til_expiry = int((end-start).total_seconds() / 3600) #number of time steps until expiery
                #T               = ((end-start).total_seconds() / 3600) #Number of hours/ times steps
                T                = ((end-start).total_seconds() / 31536000) # annualized time til expiry
                #print("T:        ",T)

                #FIND R(risk free rate of returns)
                R = 0.0181 # annualized return of the 10 year risk free bond
                #print("R:        ",R)

                #FIND SIGMA(estimated variance):
                #determine wether or not to use GARCH for this option
                
                if len(self.minute_returns) >= 1000000 :
                    if(steps_til_expiry in volitlity_calulations):
                        sigma = volitlity_calulations[steps_til_expiry]
                    else:
                        sigma = gm_result.forecast(horizon=steps_til_expiry)
                        sigma = math.sqrt(252) * sigma.variance.values[-1,:][0]
                        volitlity_calulations[steps_til_expiry] = sigma
                    print("used Garch, traning set ",len(self.minute_returns)/steps_til_expiry," time larger than steps_til_expiry")
                try:
                    sigma = iv(float(row[12]),S,K,T,R,'c')

                except Exception:
                    continue
                #print("SIGMA:    ",sigma)

                #use Black-scholes:
                
                phat = self.bs_call(S,K,T,R,sigma)
                phat = round(phat,6)
                
                #print("Estimated Price: ", phat)

                #find & print the market price
                if(float(row[7]) != 0.0):
                    market_price = float(row[7])
                    #print("Market close:    ",market_price)
                else:
                    market_price = float(row[12])
                #print("Market ask:      ",market_price)
                #print()
                g = i
            
                #determine if there is arbitrage
                
                delta_calc = delta('c', S, K, T, R, sigma)
                if  delta_calc < 0.4 and steps_til_expiry < 900 - self.counter2 :
                    position = row[3]+"_"+row[2]
                    market_price = float(row[12])
                    asset_allocation = float(10/market_price)
                    trades.append( ( position , asset_allocation , 0.0) )
                    self.counter+=1
              

                
        print("COUNTER: ",self.counter)
        print("Training_counter: ",self.training_counter)
        #print("num options in a minute: ",g)
        print(len(self.minute_returns))
        print(self.counter2)
        print()
        return trades

            