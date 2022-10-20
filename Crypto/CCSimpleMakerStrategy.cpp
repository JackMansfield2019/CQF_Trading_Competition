#include "CCSimpleMakerStrategy.h"
#include "components/VariableManager.h"
#include "components/SymbolManager.h"
#include "components/PricingModelManager.h"
#include <limits>
#include <cmath>

using namespace cts;
using namespace std;

/////////////////////// CCSimpleMakerStrategy ///////////////////////////
void CCSimpleMakerStrategy::init() {
	CCModelStrategy::init();
	SymbolManager *symMgr = SymbolManager::instance();
	PricingModelManager *pmMgr = PricingModelManager::instance();
	_notMgr = NotificationManager::instance();

	_quote_spread = getJsonValue(_cfg, "quote_spread_bps", 5.0) * 0.0001;
	_max_quote_error = getJsonValue(_cfg, "max_quote_error_bps", 1.0) * 0.0001;
	_max_quote_signal_diff = getJsonValue(_cfg, "max_quote_signal_diff_bps", 3.0) * 0.0001;
	
	string make_mid_pm = _cfg.at("make_mid_pm");
	_makeMidPm = pmMgr->getPricingModel(make_mid_pm);
	addNotifiableChild(_makeMidPm);
	
	period = 20;
	stddev = 2;
	previous_EMA = SMA(period); // figure out how to get price during initialization
	
}

/// @brief Simple Moving Average Calculation
/// @param num : Number of periods analyzed
/// @param period: Period length 
/// @return Simple Moving Average
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
}  

/// @brief Exponential Moving Average Calculation
/// @param close : close price, dType = double
/// @param previous_EMA : previous period EMA, dType = double
/// @param multiplier multiplier, dType = double
/// @return Exponential Moving Average
double EMA(double close, double previous_EMA, double multiplier){
    double EMA = ((close - previous_EMA)*multiplier + previous_EMA);
    return EMA;
}
/// @brief  Triple Exponential Moving Average (EMA) Calculation
/// @param close : close price, dType = double
/// @param period: Period length, dType = int
/// @param EMA1 : intial EMA calculated using the SMA, dType = double
/// @param EMA2 : second period EMA calculation, dType = double
/// @param EMA3 third period EMA calculation, dType = double
/// @return Triple Exponential Moving Average (TEMA)
double TEMA(double close, int period){
    double EMA1 = EMA();
    double EMA2 = EMA(EMA1);
    double EMA3 = EMA(EMA2);
    return (3 * EMA1) - (3 * EMA2) + EMA3;
}

/// @brief Bollinger Upper Band Calculation
/// @param tema : Triple Exponential Moving Average, dType = double
/// @param stddev_sum : Standard Deviation Sum, dType = double
/// @param stddev : Standard Deviation, dType = double
/// @return Bollinger Upper Band
double CCSimpleMakerStrategy::BOLU(double tema, double stddev_sum, int stddev){
	return tema + stddev * stddev_sum;
}

/// @brief Bollinger Lower Band Calculation
/// @param tema : Triple Exponential Moving Average, dType = double
/// @param stddev_sum : Standard DEviation Sum, dType = double
/// @param stddev Standard Deviation, dType = double
/// @return Bollinger Lower Band
double CCSimpleMakerStrategy::BOLD(double tema, double stddev_sum, int stddev){
	return tema - stddev * stddev_sum;
}

/// @brief Standard Deviation Summation over an x-period
/// @param prices : array of x prices dType = double
/// @param period : x period, dType = int
/// @return Standard deviation sum
double CCSimpleMakerStrategy::stddev_calculation(double prices[], int period){
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

void CCSimpleMakerStrategy::tradeOnSignal(double signal){
	if(!_si->is_ready()) {
		cancelAllOrders(CR_SIGNAL_NOT_READY);
		return;
	}

	double midPx = _si->mid_px();
	double pos = _account->position(_si->cid);	

	double thresh = _use_bps_thold ? _make_thold * 0.0001 : _make_thold * _si->tick_size / midPx;
	double low_thresh = _use_bps_thold ? _low_make_thold * 0.0001 : _low_make_thold * _si->tick_size / midPx;
	
	bool buyExpPos = willExpandPosition(BUY);
	bool sellExpPos = willExpandPosition(SELL);

	double refPx = _depPm ? _depPm->price() : midPx;
	double fv = refPx * (1.0 + signal);

	double bidTgtPx = fv * (1.0 - (buyExpPos ? thresh : low_thresh));
	double askTgtPx = fv * (1.0 + (sellExpPos ? thresh : low_thresh));

	maintainOrders(BUY, bidTgtPx, signal > 0);
	maintainOrders(SELL, askTgtPx, signal < 0);
}

double CCSimpleMakerStrategy::getMakeMidpx() {
	return _makeMidPm->price();
}

double CCSimpleMakerStrategy::calcQuotePrice(TradeDirs side, double spds) {
	double notional = _account->notional(_si->cid);
	double riskAdj = spds * notional / _max_notional;
	double sideRiskAdj = (side==BUY) ? riskAdj : -riskAdj;
	double adjSpds = fmax(0.0, spds + sideRiskAdj);
	double sign = (side==BUY) ? -1.0 : 1.0;
	return roundPrice(getMakeMidpx() * (1.0 + sign * adjSpds * _quote_spread), side);
}

void CCSimpleMakerStrategy::maintainOrders(TradeDirs side, double tgt_px, bool allowNewOrder) {
	tgt_px = roundPrice(tgt_px, side);
	double qpx = calcQuotePrice(side);
	if(hasOrders(side)) {
		double opx = getMyTopOrderPrice(side);
		if(priceBetterThan(opx, qpx, side)) {
			cancelOrders(side, CR_SIGNAL_LOW, 0, false);
		} else if(fabs(tgt_px / opx - 1.0) > _max_quote_error) {
			if(priceBetterThan(opx, tgt_px, side)) {
				cancelOrders(side, CR_SIGNAL_LOW, 0, false);
			} else {
				if(!priceEquals(opx, qpx)) {
					if(fabs(tgt_px / opx - 1.0) > _max_quote_signal_diff) {
						cancelOrders(side, CR_SIGNAL_HIGHER, 0, false);
					}
				}
			}
		}
	} else {
		if(allowNewOrder) {
			double deepest_qpx = calcQuotePrice(side, 10);
			if(priceBetterThan(tgt_px, deepest_qpx, side)) {
				double px = pickWorsePrice(tgt_px, qpx, side); 
				placeNewOrder(side, px);
			}
		}
	}
}

double CCSimpleMakerStrategy::calc_desired_order_qty(double px, TradeDirs side) {
	double pos = _account->position(_si->cid);
	double notional = _account->notional(_si->cid);
	bool expPos = willExpandPosition(side);
	double order_size = _order_notional / _si->contract_multiplier / px;
	double qty = expPos ? order_size : min(order_size, fabs(pos));
	return qty;
}

bool CCSimpleMakerStrategy::placeNewOrder(TradeDirs side, double tgt_px) {
	const nanoseconds& eventTime = _timerMgr->currentTime();
	double px = tgt_px;
	double qty = calc_desired_order_qty(px, side);

	if(placeOnLevel(side, px, qty, getMakeMidpx())) {
		if(_isLive)
			cout << toStr(eventTime)
				<< " TM_QUOTE_BBO_ONLY cid:" << _si->cid
				<< " symbol:" << _si->ticker << "." << _si->exchange
				<< " tif:day side:" << side
				<< " px:" << px << " qty:" << qty
				<< " bid:" << _si->bid_px << " ask:" << _si->ask_px
				<< " tgt_px:" << tgt_px << endl;
		return true;
	}
	return false;
}
