import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
from scipy.stats import norm

##Read in and formats options data
url="https://finance.yahoo.com/quote/AAPL/options?p=AAPL&date=1563494400"
df=pd.read_html(url) #Reads in the data tables.
calls=df[0] #Selects the table of call options
#Converts IV object to a IV numeric in decimal form.
calls['IV']=calls['Implied Volatility'].str.replace('%', '').astype(float)/100

plt.plot(calls['Strike'],calls['IV'],color='green',marker='o')

plt.title("Implied Volatility curve")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.show()

#Option priced as European
def euro_vanilla_call(S, K, T, r, sigma):
    #S: Stock price
    #K: Option strike price
    #T: time to maturity
    #C: Call value
    #r: 3 month T-bill rate
    #sigma: annual historical volatility of underlying asset
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call
call=pd.DataFrame(calls)
call=call[call.Ask!=0]
S0= 189.95
Kc=call.Strike
T=111/365
r0= 0.02398
sigma=0.1926*np.sqrt(12)
C= euro_vanilla_call(S0, Kc, T, r0, sigma)

def euro_vanilla_put(S, K, T, r, sigma):
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    
    return put

put=pd.DataFrame(df[1])
put=put[put.Ask!=0]

Kp=put.Strike
P= euro_vanilla_put(S0, Kp, T, r0, sigma)
    
# Implementation of Cox-Ross-Rubenstein option's pricing model

def CRRTree(K,T,S,sig,r,N,Type):
    
    dt=T/N
    dxu=np.exp(sig*np.sqrt(dt))
    dxd=np.exp(-sig*np.sqrt(dt))
    pu=((np.exp(r*dt))-dxd)/(dxu-dxd)
    pd=1-pu;
    disc=np.exp(-r*dt)

    St = np.zeros([(N+1),1])
    C = np.zeros([(N+1),1])
    
    St[0]=S*dxd**N
    
    for j in range(1, N+1): 
        St[j] = St[j-1] * dxu/dxd
    
    for j in range(1, N+1):
        if Type == 'p':
            C[j] = max(K-St[j],0)
        elif Type == 'c':
            C[j] = max(St[j]-K,0)
    
    for i in range(N, 0, -1):
        for j in range(0, i):
            C[j] = disc*(pu*C[j+1]+pd*C[j])
            
    return C[0]


N=250 #No. of Iteration
#All other variables as above
call_price=np.zeros([len(call.Strike),1])
put_price=np.zeros([len(put.Strike),1])
for k in range(0, len(call.Strike)):
    call_price[k] = CRRTree(call.Strike[k],T,S0,sigma,r0,N,'c')
for l in range(0, len(put.Strike)):
    put_price[l] = CRRTree(put.Strike[l],T,S0,sigma,r0,N,'p')

#Calculating RMSE Values
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
call_price_actual=np.asarray((call.Bid+call.Ask)/2)
put_price_actual=np.asarray((put.Bid+put.Ask)/2)

call_rmse=np.zeros([len(call.Strike),1])
put_rmse=np.zeros([len(put.Strike),1])
for k in range(0, len(call_price)):
    call_rmse[k]= rmse(call_price[k], call_price_actual[k])
for l in range(0, len(put_price)):
    put_rmse[l]= rmse(put_price[l], put_price_actual[l])
  
print(call_rmse.mean())
print(put_rmse.mean())


call_rmse_eur=np.zeros([len(call.Strike),1])
put_rmse_eur=np.zeros([len(put.Strike),1])
for k in range(0, len(C)):
    call_rmse_eur[k]= rmse(C[k], call_price_actual[k])
for l in range(0, len(P)):
    put_rmse_eur[l]= rmse(P[l], put_price_actual[l])

print(call_rmse_eur.mean())
print(put_rmse_eur.mean())


# Monte Carlo Simulation
np.random.seed(50)

# Number of time steps for simulation
n_steps = int(T * 252)
# Time interval
dt = T / n_steps
# Number of simulations
N = 1000
# Zero array to store values (often faster than appending)
S = np.zeros((n_steps, N))
S[0] = S0

for t in range(1, n_steps):
    # Draw random values to simulate Brownian motion
    Z = np.random.standard_normal(N)
    S[t] = S[t - 1] * np.exp((r0 - 0.5 * sigma ** 2) * \
                             dt + (sigma * np.sqrt(dt) * Z))

# Sum and discount values
call_mc=np.zeros([len(call.Strike),1])
for k in range(0, len(call.Strike)):
    call_mc[k] = np.exp(-r0 * T) * 1 / N * np.sum(np.maximum(S[-1] - call.Strike[k], 0))
put_mc=np.zeros([len(put.Strike),1])
for k in range(0, len(put.Strike)):
    put_mc[k] = np.exp(-r0 * T) * 1 / N * np.sum(np.maximum(put.Strike[k]-S[-1], 0))
    
call_rmse_mc=np.zeros([len(call.Strike),1])
put_rmse_mc=np.zeros([len(put.Strike),1])
for k in range(0, len(call_mc)):
    call_rmse_mc[k]= rmse(call_mc[k], call_price_actual[k])
for l in range(0, len(put_mc)):
    put_rmse_mc[l]= rmse(put_mc[l], put_price_actual[l])

print(call_rmse_mc.mean())
print(put_rmse_mc.mean())
    