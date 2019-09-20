import scipy.stats as ss
import math as m
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_excel(r"C:\Users\Alvi Mahmud\Desktop\QFA\cme.xlsx")

def call_bsm(S0,K,r,T,Otype,sig):
    d1 = (np.log(S0/K)) + (r+ (sig*sig)/2)*T/(sig*(np.sqrt(T)))
    d2 = d1 - sig*(m.sqrt(T))
    if (Otype == "Call"):
        price = S0*(ss.norm.cdf(np.float(d1)))- K*(np.exp(-r*T))*(ss.norm.cdf(np.float(d2)))
        return (price)
    elif (Otype == "Put"):
       price  = -S0*(ss.norm.cdf(np.float(-d1)))+ K*(m.exp(-r*T))*(ss.norm.cdf(np.float(-d2)))
       return (price)

def vega(S0,K,r,T,sig):
    d1 = (np.log(S0/K)) + (r+ (sig*sig)/2)*T/(sig*(np.sqrt(T)))
    vega = S0*(ss.norm.pdf(d1))*(np.sqrt(T))
    return(vega)
    
    
def imp_vol(S0, K, T, r, market,flag):
    e = 10e-10; x0 = 1;  
    def newtons_method(S0, K, T, r, market,flag,x0, e):
        delta = call_bsm(S0,K,r,T,flag,x0) - market
        while delta > e:
            x0 = (x0 - (call_bsm (S0,K,r,T,flag,x0) - market)/vega (S0,K,r,T,x0))
            delta = abs(call_bsm (S0,K,r,T,flag,x0) - market)
        return((x0))
    sig =  newtons_method(S0, K, T, r, market,flag,x0 , e)   
    return(sig*100)

T = 22/360   
S0 = 64.37
flag = 'Call'
r = 0.0225
iv=np.zeros((len(df),1))

for i in range (0, len(iv)):
    K= np.float(df['Strike Price'][i]/100)
    market= np.float(df['Call_Price'][i])
    iv[i] = imp_vol(S0, K, T, r, market,flag)*np.sqrt(T/3)
    i=i+1

iv_put=np.zeros((len(df),1))

for i in range (0, len(iv_put)):
    K= np.float(df['Strike Price'][i]/100)
    market= np.float(df['Put_Price'][i])
    iv_put[i] = imp_vol(S0, K, T, r, market,'Put')*np.sqrt(T/3)
    i=i+1
df['Strike']=df['Strike Price']/100
df['IV_Call']=iv
plt.plot(df['Strike'],df['IV_Call'],color='blue')
plt.title("Implied volatility(call options)")
plt.xlabel("Strike Price")
plt.ylabel("Implied volatility")
plt.show()
df['IV_put']=iv_put 
plt.plot(df['Strike'],df['IV_put'],color='green')
plt.title("Implied volatility(put options)")
plt.xlabel("Strike Price")
plt.ylabel("Implied volatility")
plt.show()
#3rd degree polynomial spline with 6 knots: 0.70,0.80,0.90,0.95,1.05,1.10    
df['Moneyness']=df['Strike Price']/S0
yy=np.matrix(df['IV_Call']).T

xx=np.concatenate((np.matrix(np.ones(len(df))),np.matrix(df.Moneyness),
    np.matrix(df.Moneyness**2),np.matrix(df.Moneyness**3),
    np.matrix((((df.Moneyness-0.70)+0+abs(0-(df.Moneyness-0.70)))/2)**3),
    np.matrix(((((df.Moneyness-0.80)+0+abs(0-(df.Moneyness-0.80)))/2)**3)),
    np.matrix(((((df.Moneyness-0.90)+0+abs(0-(df.Moneyness-0.90)))/2)**3)),
    np.matrix(((((df.Moneyness-0.95)+0+abs(0-(df.Moneyness-0.95)))/2)**3)),
    np.matrix(((((df.Moneyness-1.05)+0+abs(0-(df.Moneyness-1.05)))/2)**3)),
    np.matrix(((((df.Moneyness-1.10)+0+abs(0-(df.Moneyness-1.10)))/2)**3))),axis=0).T
beta=(xx.T*xx)**(-1)*xx.T*yy #OLS estimates of the coefficients
df['IVfit']=xx*beta #Model-fit IV.

#Plots the IV model fit
plt.plot(df.Moneyness,iv,color='black',marker='o',fillstyle='none',linestyle='none')
plt.plot(df.Moneyness,df.IVfit,color='black')
plt.title("Implied volatility fit (call options)")
plt.xlabel("Option moneyness")
plt.ylabel("Implied volatility")
plt.show()
#Grid
##MAKING GRID OF IV
k= df['Strike Price']
for i in range (0, len(df['Strike Price'])):
 Mongrid=np.matrix(np.arange(0.9*df['Moneyness'][i],1.1*df['Moneyness'][i],0.005)).T
 Ivgrid=np.matrix(np.zeros(len(Mongrid))).T
 i=i+1
#This loop fills in a grid of IVs implied by the spline model and moneyness grid
i=0
while i<len(Mongrid):
    Ivgrid[i,0]=beta[0,0]+Mongrid[i,0]*beta[1,0]+Mongrid[i,0]**2*beta[2,0]+Mongrid[i,0]**3*beta[3,0]+(((Mongrid[i,0]-0.70)+0+abs(0-(Mongrid[i,0]-0.70)))/2)**3*beta[4,0]+(((Mongrid[i,0]-0.80)+0+abs(0-(Mongrid[i,0]-0.80)))/2)**3*beta[5,0]+(((Mongrid[i,0]-0.90)+0+abs(0-(Mongrid[i,0]-0.90)))/2)**3*beta[6,0]+(((Mongrid[i,0]-0.95)+0+abs(0-(Mongrid[i,0]-0.95)))/2)**3*beta[7,0]+(((Mongrid[i,0]-1.05)+0+abs(0-(Mongrid[i,0]-1.05)))/2)**3*beta[8,0]+(((Mongrid[i,0]-1.10)+0+abs(0-(Mongrid[i,0]-1.10)))/2)**3*beta[9,0]
    i=i+1
bsd1=np.matrix(np.zeros(len(Mongrid))).T
bsd2=np.matrix(np.zeros(len(Mongrid))).T
#This loop calculates d1 for each moneyness
i=0
while i<len(Mongrid):
    bsd1[i,0]=(np.log(S0/Mongrid[i,0]*S0) + 0.5*(r+ (Ivgrid[i,0]**2)/6)*T)/(Ivgrid[i,0]*(np.sqrt(T/3)))
    i=i+1
bsd2=bsd1-Ivgrid*np.sqrt(T/3) #d2
#Black-Scholes option prices 
bsc= np.matrix(np.zeros(len(Mongrid))).T
for i in range(0,len(Mongrid)):
    bsc[i,0]= (np.exp(-r*T))*(S0*np.exp(0.5*(r- (Ivgrid[i]**2)/6)*T)*ss.norm.cdf(bsd1[i],0.0,1.0)+(ss.norm.cdf(bsd2[i])*(Mongrid[i]*S0)/100))
    i=i+1
#Bootstrap
from sklearn.utils import resample
# data
data = pd.read_csv(r"C:\Users\Alvi Mahmud\Desktop\QFA\spot.csv")
# prepare bootstrap sample

for i in range(0,1000):
        boot = resample(data['Spot Price'], replace=True, n_samples=len(Mongrid), random_state=1)
        boot=np.matrix(boot).T
        i=i+1

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
boot_rmse= np.zeros([len(boot),1])
for k in range(0, len(bsc)):
    boot_rmse[k]= rmse(boot[k], bsc[k])
print(boot_rmse.mean())
#Binomial Tree

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
T = 22/360   
S0 = 64.37
K=df['Strike Price']
r0 = 0.0225
sigma= df['IVfit']
call_price=np.zeros([len(df['Strike Price']),1])
for k in range(0, len(df['Strike Price'])):
    call_price[k] = CRRTree(K[k],T,S0,sigma[k],r0,N,'c')

call_rmse=np.zeros([len(K),1])
for k in range(0, len(call_price)):
    call_rmse[k]= rmse(call_price[k], df['Call_Price'][k]) 
print(call_rmse.mean())
