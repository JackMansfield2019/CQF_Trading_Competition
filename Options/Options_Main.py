import random
import pandas as pd 
import numpy as np 
#import Tkinter
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import datetime

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

    
    def read_data(self, row_vals):
        pass
        
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
        pass