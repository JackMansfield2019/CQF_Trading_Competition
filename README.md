# CQF_Trading_Competition

# Options Case:

## Strategy 1 
1. implment py_vollib's implied volatility calulation

![image](https://user-images.githubusercontent.com/25088039/206798680-d01ccc61-fb5c-4f16-b15d-acf612b4c592.png)

2. grab Parameters for Black scholes
3. use Black shcoles

![Capture1](https://user-images.githubusercontent.com/25088039/206802968-bdbf5156-4b7e-4103-8cc3-a42ce92b64dd.JPG)

4. if Black scholes esimated price < market price
    short that call
### Results

![Capture](https://user-images.githubusercontent.com/25088039/206802618-c00ec7d9-2dd6-44e6-acec-d38b2755e92a.JPG)

## Strategy 2
1. implment py_vollib's implied volatility calulation 
2. implement Black scholes
3. if Delta < 0.4
    short that call
4. after 2 days of trading:

      buy 1000 units of underlying every minute
### Results 

Total Trades: Trades: 1106

![Capture2](https://user-images.githubusercontent.com/25088039/206803845-29edb68a-b437-489d-9780-d683aa91303b.JPG)

## usage
  
     python3 backtesting_engine.py 
