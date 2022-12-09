# CQF_Trading_Competition

# Options Case:

### Strategy 
1. implment py_vollib's implied volatility calulation 
2. implement Black scholes
![image](https://user-images.githubusercontent.com/25088039/206798680-d01ccc61-fb5c-4f16-b15d-acf612b4c592.png)
3. if Black scholes value < market_price && delta < 0.4
    short black scholes
4. after 2 days of trading:
      buy 1000 units of underlying every minute 

## results:

Trades: 1106

INFO - ####2020-11-06 13:29:00####

INFO - #####

INFO - Liquid Cash: 1143031.8356323359

INFO - Current Value: 301740.30960289645

INFO - #####

## usage
  
     python3 backtesting_engine.py 
