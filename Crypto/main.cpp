#include "strategy.cpp"
#include <vector>

using namespace std;

int main(int argc, char** argv){
    int period = 20;
    int stddev = 2;
    double previous_EMA = 0; // need to calculate this at start
    double prices[period];
    double close = prices[period - 1];
    double previous_close = prices[period - 2];

    double stddev_sum = stddev_calculation(prices, period);
    double tema = TEMA(close, period);
    double upper_band = BOLU(tema, stddev_sum, stddev);
    double lower_band = BOLD(tema, stddev_sum, stddev);

    if(previous_close <= lower_band && close >= upper_band) {
        //buy trade
    }
    if(previous_close <= upper_band && close >= upper_band) {
        //sell trade
    }

    return 0;
}