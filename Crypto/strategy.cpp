#include <cmath>
#include <vector>

using namespace std;

/*
bollinger band's upper band
formula: MA(TP,n) + stddev * stddev(TP,n)
    - TP = (high + low + close) / 3
    - n = number of hours in smoothing period
    - m = number of standard deviations
    - stddev(TP,n) = stddev over last n periods of TP
double BOLU(double tema, int period, int stddev, double stddev_sum, double close){
    //double TP = (high + low + close) / 3;
    double close = prices[period-1];
    double upper_band = TEMA(close,period) + stddev * stddev_calculation(prices,period);
    return tema + stddev * stddev_sum;
    return upper_band;
}
*/

// bollinger band's upper band
double BOLU(double tema, double stddev_sum, int stddev){
    return tema + stddev * stddev_sum;
}

/* bollinger band's lower band
// formula: MA(TP,n) - stddev * stddev(TP,n)
double BOLD(double prices[],int period, int stddev){
    //double TP = (high + low + close) / 3;
    double close = prices[period-1];
    double lower_band = TEMA(close,period) - stddev * stddev_calculation(prices,period);
    return lower_band;
}
*/

// bollinger band's lower band
double BOLD(double tema, double stddev_sum, int stddev){
    return tema - stddev * stddev_sum;
}

// calculates the standard deviation of x periods
double stddev_calculation(double prices[], int period){
    double sum = 0;

    for(int i=0; i < period; i++){
        sum = sum + prices[i];
    }
    double x_bar = sum / 20;

    sum = 0;
    for(int i=0; i < period; i++){
        double x = prices[i] - x_bar;
        sum = sum + pow(x,2);
    }

    double left = 1 / (period - 1);
    double variance = left * sum;
    double stddev = sqrt(variance);
    return stddev;
}

//SMA
//Initial SMA Formula: 10-period sum / 10
double SMA(int period){
  double sum;
 
  // function to add new data in the
  // list and update the sum so that
  // we get the new mean
  void addData(double num)
  {
    sum += num;
    Dataset.push(num);
 
    // Updating size so that length
    // of data set should be equal
    // to period as a normal mean has
    if (Dataset.size() > period) {
      sum -= Dataset.front();
      Dataset.pop();
    }
  }
 
  // function to calculate mean
  double getMean() { return sum / period; }
};

//EMA
//Formula: {Close-EMA(previous day)} x multiplier + EMA(previous day)
double EMA(double close, double previous_EMA, double multiplier){

    double EMA = ((close - previous_EMA)*multiplier + previous_EMA);
    return EMA;
}

//TEMA: triple exponential moving average
double TEMA(double close, int period){
    double EMA1 = EMA();
    double EMA2 = EMA(EMA1);
    double EMA3 = EMA(EMA2);
    return (3 * EMA1) - (3 * EMA2) + EMA3;
}