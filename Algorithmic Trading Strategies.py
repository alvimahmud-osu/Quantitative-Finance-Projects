from __future__ import print_function
import fix_yahoo_finance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy import stats
import time
start_time = time.time()
start= '2007-01-01'
end= '2019-01-01'
tickers=['TSLA']
data=yf.download(tickers, start, end)
TSLA=pd.concat([data['Adj Close'], data.Volume], axis=1)
TSLA['diff']= data['Open']-data['Close']
TSLA_dret=TSLA['Adj Close'].pct_change().fillna(0)
TSLA_monret=TSLA_dret.resample('M').mean()
TSLA_cum_ret=(1+TSLA['Adj Close'].pct_change()).cumprod()

TSLA['Return']= TSLA_dret
# Short moving window rolling mean
TSLA['12'] = TSLA['Adj Close'].rolling(window=12).mean()

# Long moving window rolling mean
TSLA['26'] = TSLA['Adj Close'].rolling(window=26).mean()

TSLA[['Adj Close', '12', '26']].plot()
plt.show()



#Create Buy-Sell Signals using Moving Average
short_window = 12
long_window = 26
signals = pd.DataFrame(index=TSLA.index)
signals['signal'] = 0.0
signals['short_mavg'] =TSLA['Adj Close'].rolling(window=short_window, min_periods=1, center=False).mean()
signals['long_mavg'] = TSLA['Adj Close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)   

# Generate trading orders
signals['positions'] = signals['signal'].diff()

print(signals.head)

fig = plt.figure()
ax1 = fig.add_subplot(111,  ylabel='Price in $')
TSLA['Adj Close'].plot(ax=ax1, color='r', lw=2.)
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
ax1.plot(signals.loc[signals.positions == 1.0].index, 
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')
         
ax1.plot(signals.loc[signals.positions == -1.0].index, 
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')
plt.show()
#BackTest
initial_capital= float(100000.0)
positions = pd.DataFrame(index=signals.index).fillna(0.0)

positions['TSLA'] = 100*signals['signal']     
portfolio = positions.multiply(TSLA['Adj Close'], axis=0)
pos_diff = positions.diff()
portfolio['holdings'] = (positions.multiply(TSLA['Adj Close'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(TSLA['Adj Close'], axis=0)).sum(axis=1).cumsum()   
portfolio['total'] = portfolio['cash'] + portfolio['holdings']
portfolio['returns'] = portfolio['total'].pct_change()
endtime=time.time()
print("--- %s seconds ---" % (endtime - start_time))
fig2 = plt.figure()

ax3 = fig2.add_subplot(111, ylabel='Portfolio value in $')
portfolio['total'].plot(ax=ax3, lw=2.)
ax3.plot(portfolio.loc[signals.positions == 1.0].index, 
         portfolio.total[signals.positions == 1.0],
         '^', markersize=10, color='m')
ax3.plot(portfolio.loc[signals.positions == -1.0].index, 
         portfolio.total[signals.positions == -1.0],
         'v', markersize=10, color='k')
plt.show()

returns = portfolio['returns']
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
MDD=(returns.max()-returns.min())/returns.max()
monthly_returns=returns.resample('M').last()
Buy= signals.loc[signals.signal == 1.0].count()
Sell= signals.loc[signals.signal == 0.0].count()
portfolio['returns'].cumsum().plot()
monthly_returns.plot()
one_sample = stats.ttest_1samp(returns.dropna(), 0)
print(one_sample)

#Bollinger Bands
start_time = time.time()
window = 21
no_of_std = 2
df2= TSLA
#Calculate rolling mean and standard deviation using number of days set above
rolling_mean = df2['Adj Close'].rolling(window).mean()
rolling_std = df2['Adj Close'].rolling(window).std()

#create two new DataFrame columns to hold values of upper and lower Bollinger bands
df2['Rolling Mean'] = rolling_mean
df2['Bollinger High'] = rolling_mean + (rolling_std * no_of_std)
df2['Bollinger Low'] = rolling_mean - (rolling_std * no_of_std)

#Define function to calculate Bollinger Bands

def bollinger_strat(df,window,std):
    rolling_mean = df2['Adj Close'].rolling(window).mean()
    rolling_std = df2['Adj Close'].rolling(window).std()
    
    df2['Bollinger High'] = rolling_mean + (rolling_std * no_of_std)
    df2['Bollinger Low'] = rolling_mean - (rolling_std * no_of_std)
    
    df2['Short'] = None
    df2['Long'] = None
    df2['Position'] = None
    
    for row in range(len(df2)):
    
        if (df2['Adj Close'].iloc[row] > df2['Bollinger High'].iloc[row]) and (df2['Adj Close'].iloc[row-1] < df2['Bollinger High'].iloc[row-1]):
            df['Position'].iloc[row] = -1
        
        if (df2['Adj Close'].iloc[row] < df2['Bollinger Low'].iloc[row]) and (df2['Adj Close'].iloc[row-1] > df2['Bollinger Low'].iloc[row-1]):
            df2['Position'].iloc[row] = 1
            
    df2['Position'].fillna(method='ffill',inplace=True)
    
    df2['Market Return'] = np.log(df2['Adj Close'] / df2['Adj Close'].shift(1))
    df2['Strategy Return'] = df2['Market Return'] * df2['Position']
    
    df2['Strategy Return'].cumsum().plot()
    df2[['Adj Close','Bollinger High','Bollinger Low','Position']].plot()
bollinger_strat(df2,10,2)

fig3 = plt.figure()

ax2 = fig3.add_subplot(111, ylabel='Price')
df2[['Adj Close','Bollinger High','Bollinger Low']].plot(ax=ax2, lw=2.)
ax1.plot(df2.loc[df2.Position == 1.0].index, 
         df2.Short[df2.Position == 1.0],
         '^', markersize=10, color='m')
         
ax1.plot(df2.loc[df2.Position == -1.0].index, 
         df2.Long[df2.Position == -1.0],
         'v', markersize=10, color='k')
plt.show()
#Backtest
initial_capital= float(100000.0)
positions = pd.DataFrame(index=df2.index).fillna(0.0)

positions['TSLA'] = 100*df2['Position']     
portfolio2 = positions.multiply(TSLA['Adj Close'], axis=0)
pos_diff = positions.diff()
portfolio2['holdings'] = (positions.multiply(TSLA['Adj Close'], axis=0)).sum(axis=1)
portfolio2['cash'] = initial_capital - (pos_diff.multiply(TSLA['Adj Close'], axis=0)).sum(axis=1).cumsum()   
portfolio2['total'] = portfolio2['cash'] + portfolio2['holdings']
portfolio2['returns'] = portfolio2['total'].pct_change()
endtime=time.time()
print("--- %s seconds ---" % (endtime - start_time))
fig2 = plt.figure()

ax3 = fig2.add_subplot(111, ylabel='Portfolio value in $')
portfolio2['total'].plot(ax=ax3, lw=2.)
ax3.plot(portfolio2.loc[df2.Position == 1.0].index, 
         portfolio2.total[df2.Position == 1.0],
         '^', markersize=10, color='m')
ax3.plot(portfolio2.loc[df2.Position == -1.0].index, 
         portfolio2.total[df2.Position == -1.0],
         'v', markersize=10, color='k')
plt.show()

returns2 = portfolio2['returns']
sharpe_ratio2 = np.sqrt(252) * (returns2.mean() / returns2.std())
MDD2=(returns2.max()-returns2.min())/returns2.max()
monthly_returns2=returns2.resample('M').last()
Buy2= df2.Position.loc[df2.Position == -1.0].count()
Sell2= df2.Position.loc[df2.Position == 1.0].count()
monthly_returns2.plot()
returns2.cumsum().plot()
one_sample2 = stats.ttest_1samp(returns2.dropna(), 0)
print(one_sample2)
#Momentum

from pyalgotrade import strategy
from pyalgotrade import plotter
from pyalgotrade.tools import quandl
from pyalgotrade.technical import vwap
from pyalgotrade.stratanalyzer import sharpe
from pyalgotrade.stratanalyzer import returns
from pyalgotrade.stratanalyzer import drawdown
from pyalgotrade.stratanalyzer import trades
start_time = time.time()

class VWAPMomentum(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, vwapWindowSize, threshold):
        super(VWAPMomentum, self).__init__(feed)
        self.__instrument = instrument
        self.__vwap = vwap.VWAP(feed[instrument], vwapWindowSize)
        self.__threshold = threshold

    def getVWAP(self):
        return self.__vwap

    def onBars(self, bars):
        vwap = self.__vwap[-1]
        if vwap is None:
            return

        shares = self.getBroker().getShares(self.__instrument)
        price = bars[self.__instrument].getClose()
        notional = shares * price

        if price > vwap * (1 + self.__threshold) and notional < 1000000:
            self.marketOrder(self.__instrument, 100)
        elif price < vwap * (1 - self.__threshold) and notional > 0:
            self.marketOrder(self.__instrument, -100)


def main(plot):
    instrument = "TSLA"
    vwapWindowSize = 60
    threshold = 0.00000001

    # Download the bars.
    feed = quandl.build_feed("WIKI", [instrument], 2007, 2018, ".")

    strat = VWAPMomentum(feed, instrument, vwapWindowSize, threshold)
    sharpeRatioAnalyzer = sharpe.SharpeRatio()
    strat.attachAnalyzer(sharpeRatioAnalyzer)
    retAnalyzer = returns.Returns()
    strat.attachAnalyzer(retAnalyzer)
    drawDownAnalyzer = drawdown.DrawDown()
    strat.attachAnalyzer(drawDownAnalyzer)
    tradesAnalyzer = trades.Trades()
    strat.attachAnalyzer(tradesAnalyzer)
    returns3= tradesAnalyzer.getAllReturns()
    if plot:
        plt = plotter.StrategyPlotter(strat, True, False, True)
        plt.getInstrumentSubplot(instrument).addDataSeries("vwap", strat.getVWAP())

    strat.run()
    print("Final portfolio value: $%.2f" % strat.getResult())
    print("Cumulative returns: %.2f %%" % (retAnalyzer.getCumulativeReturns()[-1] * 100))
    print("Sharpe ratio: %.2f" % (sharpeRatioAnalyzer.getSharpeRatio(0.05)))
    print("Max. drawdown: %.2f %%" % (drawDownAnalyzer.getMaxDrawDown() * 100))
    print("Longest drawdown duration: %s" % (drawDownAnalyzer.getLongestDrawDownDuration()))
    print(retAnalyzer.getReturns())
    print("")
    print("Total trades: %d" % (tradesAnalyzer.getCount()))
    if tradesAnalyzer.getCount() > 0:
       profits = tradesAnalyzer.getAll()
       print("Avg. profit: $%2.f" % (profits.mean()))
       print("Profits std. dev.: $%2.f" % (profits.std()))
       print("Max. profit: $%2.f" % (profits.max()))
       print("Min. profit: $%2.f" % (profits.min()))
       print(stats.ttest_1samp(tradesAnalyzer.getAllReturns(), 0))
    if plot:
        plt.plot()


if __name__ == "__main__":
    main(True)
 

endtime=time.time()
print("--- %s seconds ---" % (endtime - start_time))

