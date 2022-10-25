import csv
import matplotlib.pyplot as plt
import logging
import numpy as np
import strategy


class BackTestEngine:
    liquid_cash = 0
    log_path, data_path = '', ''
    # portfolio['strike_expiry-date']=(net position) - Is negative for short
    portfolio, historic_value = {}, []
    strategy = None

    def __init__(self, data_path, init_capital=100000, log_path=''):
        logging.basicConfig(filename='strat_logs.log', filemode='w', format='%(levelname)s - %(message)s',
                            level=logging.INFO)
        self.log_path, self.data_path = log_path, data_path
        self.liquid_cash = init_capital
        self.portfolio = {}
        self.historic_value.append(init_capital)
        self.strategy = strategy.Strategy()
        self.trade_arr = [0]
        self.trades = 0

    def feed_minute_data(self, raw_data):
        self.strategy.read_data(raw_data)

    def execute_options(self, raw_date, underlying):
        date, minute = raw_date.split(' ')
        # Is end of day?
        if minute == '16:15:00':
            # Iterate over all open options and close them
            for option in self.portfolio.copy():
                if option != "underlying" and option.split('_')[1] == date:
                    long_alpha = max(0, underlying[1] - float(option.split('_')[0]))
                    # Assumes you have to buy the underlying at execution for a short
                    short_alpha = max(0, underlying[3] - float(option.split('_')[0]))
                    # Executing positions
                    pnl = long_alpha * self.portfolio[option] if self.portfolio[option] > 0 else \
                        short_alpha * self.portfolio[option]
                    self.liquid_cash += pnl
                    logging.info("Exercising " + option + ": Settled position of " + str(self.portfolio[option]) +
                                 " at a pnl of " + str(pnl))
                    self.trades += 1
                    self._produce_stats(underlying)
                    del self.portfolio[option]

    def _is_valid_trade(self, trade, offered_options):
        return trade[0] in offered_options and float(trade[1]) <= offered_options[trade[0]][0] * 100 and \
               float(trade[2]) <= offered_options[trade[0]][2] * 100 and \
               offered_options[trade[0]][3] * trade[2] - offered_options[trade[0]][1] * trade[1] <= self.liquid_cash

    def execute_strategy(self, offered_options):
        trades = self.strategy.make_trades()
        for trade in trades:
            if self._is_valid_trade(trade, offered_options):
                # Converting 1 option to 100 shares
                sell_share_volume = trade[1] * 100
                long_share_volume = trade[2] * 100
                # Informing of and executing trade
                logging.info(trade[0] + ": Shorting " + str(sell_share_volume) + " shares at " + \
                             str(offered_options[trade[0]][1]) + " | Going long " + str(long_share_volume) + \
                             " shares at " + str(offered_options[trade[0]][3]))
                logging.info("Transaction Net Value: " + str(offered_options[trade[0]][1] * sell_share_volume -
                                                             offered_options[trade[0]][3] * long_share_volume))
                # Registers the trade and its affect on liquidity
                self.liquid_cash += \
                    (offered_options[trade[0]][1] * sell_share_volume - offered_options[trade[0]][3] * long_share_volume)
                self.portfolio[trade[0]] = self.portfolio[trade[0]] + long_share_volume - sell_share_volume if \
                    trade[0] in self.portfolio else long_share_volume - sell_share_volume
                self.trades += (sell_share_volume + long_share_volume)
            else:
                logging.warning("The trade order: " + str(trade) + " is not valid!")
                if self.liquid_cash == 0:
                    logging.warning("No liquid cash")

    def _position_liquid_value(self, position, offered_options):
        return offered_options[position][1] * self.portfolio[position] if self.portfolio[position] > 0 else \
            offered_options[position][3] * self.portfolio[position]

    def run_strategy_for_minute(self, raw_data, offered_options):
        self.feed_minute_data(raw_data)
        self.execute_strategy(offered_options)
        self._produce_stats(offered_options)

    def _liquidate_positions(self, offered_options):
        for position in self.portfolio:
            self.liquid_cash += self._position_liquid_value(position, offered_options)
        self.portfolio = {}

    def _call_strategy(self, current_date, raw_minute_data, offered_options):
        # Do we need to call the strategy? We won't call it on empty data
        if len(raw_minute_data) != 0:
            self.run_strategy_for_minute(raw_minute_data, offered_options)
            self.execute_options(current_date, offered_options['underlying'])

    def run(self):
        with open(self.data_path) as data_file:
            data = csv.reader(data_file)
            offered_options, raw_minute_data = {}, []
            current_date = ''
            for i, row in enumerate(data):
                # Ignores the first row as its all titles
                if i == 0:
                    continue
                # If the date has changed
                if current_date != row[1]:
                    # Call the strategy
                    self._call_strategy(current_date, raw_minute_data, offered_options)
                    # Register new data
                    current_date = row[1]
                    logging.info("####" + current_date + "####")
                    offered_options['underlying'] = (1e10, float(row[15]), 1e10, float(row[16]))
                    raw_minute_data = []
                # Dictionary of the volume and price of options (Bid Volume, Bid Price, Ask Volume, Ask Price)
                offered_options[row[4] + "_" + row[3]] = (float(row[11]), float(row[12]), float(row[13]), float(row[14]))
                # Append to raw_minute_data
                raw_minute_data.append(row)
            # Submit any remaining data for the last minute of the file to the strategy
            self._call_strategy(current_date, raw_minute_data, offered_options)
            # Liquidate any remaining positions we might have
            self._liquidate_positions(offered_options)
            self._produce_final_chart()

    def _produce_stats(self, offered_options):
        current_value = self.liquid_cash
        # This will throw if we don't have a current value for the option
        for position in self.portfolio:
            # Assumes that the short has best ask price while long has best bid
            if position in offered_options:
                current_value += self._position_liquid_value(position, offered_options)
        self.historic_value.append(current_value)
        self.trade_arr.append(self.trades)
        logging.info("#####")
        logging.info("Liquid Cash: " + str(self.liquid_cash))
        logging.info("Current Value: " + str(current_value))
        logging.info("#####")

    # Plots a basic final PnL chart
    def _produce_final_chart(self):
        x = np.arange(len(self.historic_value))
        y = np.array(self.historic_value)
        trade_ct = np.array(self.trade_arr)
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        fig.suptitle('Visual Representation of Strategy Performance', fontsize=16, color='#d24b65')

        peak = np.argmax(np.maximum.accumulate(y) - y)  # end of the period
        trough = np.argmax(y[:peak])

        axs[0].plot(x, y, color='#d24b65')
        axs[0].set_title(f"Returns: {100 * (y[-1] - y[0]) / y[0]:.2f}%, max drawdown: {100 * (1 - (y[trough] / y[peak])):.2f}%")
        axs[0].scatter([peak, trough], [y[peak], y[trough]], color='blue')
        axs[0].set_xlabel('Time (mins)')
        axs[0].set_ylabel('Profit / Loss')
        axs[1].scatter(x, trade_ct, color='#d24b65')
        axs[1].set_title(f"{self.trades} total trades made, mean ${y[-1] - y[0]:.2f} profit per trade")
        axs[1].set_xlabel('Time (mins)')
        axs[1].set_ylabel('Trade count')
        axs[1].legend()
        plt.show()


bt = BackTestEngine("data/eighth_data.csv")
bt.run()