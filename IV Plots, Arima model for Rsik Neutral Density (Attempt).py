import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as si

options= pd.read_csv(r'C:\Users\Alvi Mahmud\Desktop\QFA\Option.csv')
opt=pd.DataFrame({'date':pd.to_datetime(options.date, dayfirst=True), 'expiry_date': pd.to_datetime(options.exdate, dayfirst=True),'option_type': options.cp_flag,
                  'strike':options.strike_price/1000, 'bid':options.best_bid,'ask':options.best_offer})
index= pd.read_csv(r'C:\Users\Alvi Mahmud\Desktop\QFA\Index.csv')
ind=pd.DataFrame({'date':pd.to_datetime(index.date, dayfirst=True),'index':index.spindx,'rfr':index.r})

df= pd.merge(opt, ind, how="left", left_on="date", right_on="date")
df['days_to_maturity']=df.expiry_date-df.date
print(df.head(30))
# Find Implied volatility
def euro_vanilla_call(S, K, T, r, sigma):
    #S: index price
    #K: strike price
    #T: time to maturity
    #C: Call value
    #r: interest rate
    #sigma: volatility of underlying asset
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call
call=df['option_type']=='C'
df_call= df[call]

Sc= df_call['index']
Kc=df_call.strike
T=30/365
rc=df_call.rfr
sigma_call=0.5
C= euro_vanilla_call(Sc, Kc, T, rc, sigma_call)

def newton_vol_call(S, K, T, C, r, sigma):
    
    #S: index price
    #K: strike price
    #T: time to maturity
    #C: Call value
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    fx = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0) - C
    
    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
    
    tolerance = 0.000001
    x0 = sigma
    xnew  = x0
    xold = x0 - 1
        
    while abs(xnew - xold) > tolerance:
    
        xold = xnew
        xnew = (xnew - fx - C) / vega
        
        return abs(xnew)

df_call['IV']= newton_vol_call(Sc, Kc, T, C, rc, sigma_call)

plt.plot(df_call.strike, df_call.IV)
plt.title('Call Option implied volatility density Plot')
plt.xlim(1000,3600)
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.show()

plt.plot(df_call['index'], df_call.IV)
plt.title('Empirical density of prices over call option implied volatility')
plt.xlabel ('Index Level')
plt.ylabel('Implied Volatility')
plt.show()

def euro_vanilla_put(S, K, T, r, sigma):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    
    return put

put=df['option_type']=='P'
df_put= df[put]

Sp= df_put['index']
Kp=df_put.strike
T=30/365
rp=df_put.rfr
sigma_put=0.5
P= euro_vanilla_put(Sp, Kp, T, rp, sigma_put)

def newton_vol_put(S, K, T, P, r, sigma):
    
    d1 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    fx = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0) - P
    
    vega = (1 / np.sqrt(2 * np.pi)) * S * np.sqrt(T) * np.exp(-(si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)
    
    tolerance = 0.000001
    x0 = sigma
    xnew  = x0
    xold = x0 - 1
        
    while abs(xnew - xold) > tolerance:
    
        xold = xnew
        xnew = (xnew - fx - P) / vega
        
        return abs(xnew)

df_put['IV']= newton_vol_put(Sp, Kp, T, P, rp, sigma_put)

plt.plot(df_put.strike, df_put.IV)
plt.title('Put Option implied volatility density Plot')
plt.xlabel ('Strike Price')
plt.ylabel('Implied Volatility')
plt.show()

#arima fitting
plt.plot(df_put['index'], df_put.IV)
plt.title('Empirical density of prices over put option implied volatility')
plt.xlabel ('Index Level')
plt.ylabel('Implied Volatility')
plt.show()

from statsmodels.tsa.arima_model import ARIMA
srs= pd.concat([df_call.date,df_call.IV],axis=1)
series=srs.set_index('date')
# fit model
model = ARIMA(series, order=(1,0,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())