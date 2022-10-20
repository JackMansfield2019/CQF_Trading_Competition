#include "CCSimTrader.h"
#include "base/XlibComponentFactory.h"
#include "CCSimpleMakerStrategy.h"
#include "components/StrategyManager.h"
#include "components/PricingModelManager.h"
#include "components/SamplerManager.h"
#include "components/VariableManager.h"

using namespace cts;
using namespace std;

//// CONFIGURE BELOW

class MyComponentFactory : public XlibComponentFactory
{
public:
	Strategy *getStrategy(const string &type, const string &name, const json &attr)
	{
		if (type == "CCSimpleMakerStrategy") 
			return new CCSimpleMakerStrategy(name, attr);            
		return nullptr;
	}

	Sampler *getSampler(const string &type, const string &name, const json &attr)
	{
		Sampler *sampler = nullptr;

	//if (type == "TimeSampler")
		//    return new TimeSampler(name, attr);

		return sampler;
	}

	PricingModel *getPricingModel(const string &type, const string &name, const json &attr)
	{
		PricingModel *pm = nullptr;
		
	//if (type == "MidPx")
		//    return new MidPx(name, attr);

		return pm;
	}

	Variable *getVariable(const string &type, const string &name, const json &attr)
	{
		Variable *var = nullptr;
		
	//if (type == "Sum")
		//    return new Sum(name, attr);

		return var;
	}
};




//// CONFIGURE ABOVE

int main(int argc, char **argv)
{
	if (argc <= 2)
	{
		std::cout << "Usage: strat cfg_path date" << std::endl;
		return 0;
	}

	//load configuration
	std::time_t t = std::time(0);
	std::tm *now = std::localtime(&t);
	//int date = (now->tm_year + 1900) * 10000 + (now->tm_mon + 1) * 100 + now->tm_mday;
	int date = atoi(argv[2]);
	json cfg = loadConfig(argv[1],date);
	json &instCfg = cfg["instance"];
	instCfg["tradeDate"] = date;
	instCfg["isLive"] = true;

	//setup logger
	string instName = getJsonValue(instCfg, "name", string("instname"));
	string logPath = getJsonValue(instCfg, "log_path", string("."));
	logPath = replace_home_path(logPath);
	date = instCfg["tradeDate"];
	logPath += "/" + to_string(date);
	mkdirs(logPath);
	logPath += "/inst_" + instName + ".log";
	int logLevel = getJsonValue(instCfg, "log_level", qts::log4z::LOG_LEVEL_WARN);
	QTS_LOG_START(logLevel, logPath);

	//setup your custom component factory to load components
	XlibComponentFactoryPtr xmgr(new MyComponentFactory);
	StrategyManager::instance()->addXlibComponentFactory(xmgr);
	SamplerManager::instance()->addXlibComponentFactory(xmgr);
	VariableManager::instance()->addXlibComponentFactory(xmgr);
	PricingModelManager::instance()->addXlibComponentFactory(xmgr);
	
	//start cctrader
	startSimTrader(cfg);
}
