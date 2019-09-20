import fix_yahoo_finance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
start= '2006-01-01'
end= '2018-01-01'
tickers=['GS','JPM','C','BAC','MS']
data=yf.download(tickers, start, end)
mo=data['Adj Close'].resample('M').last() #monthly obs
#read in the market excess return from the FAMA French Liquidity Factor in WRDS
rf= pd.read_csv(r'C:\Users\Alvi Mahmud\Desktop\QFA\ff.csv', parse_dates=['dateff'], index_col=['dateff']).fillna(value = 0)

mat=np.matrix(pd.concat([mo.GS, mo.JPM, mo.C, mo.BAC, mo.MS], axis=1))

rfr= np.matrix([rf.rf]).T
mktrf= np.matrix([rf.mktrf]).T

M=mat.shape[0] #Number of rows in rmat
N=mat.shape[1] #number of columns (assets) in rmat

import statsmodels.api as sm
#OLS
GS= mo['GS'].pct_change(1).fillna(0)
JPM=mo['JPM'].pct_change(1).fillna(0)
C= mo['C'].pct_change(1).fillna(0)
BAC=mo['BAC'].pct_change(1).fillna(0)
MS=mo['MS'].pct_change(1).fillna(0)
mkr=np.matrix(rf.mktrf.fillna(0)).T
Ygs=np.matrix(GS).T
Yjp=np.matrix(JPM).T
Yc=np.matrix(C).T
Ybac=np.matrix(BAC).T
Yms=np.matrix(MS).T
X = (sm.add_constant(mkr))

[a,b]= (X.T*X)**(-1)*X.T*Ygs
[c,d]= (X.T*X)**(-1)*X.T*Yjp
[e,f]= (X.T*X)**(-1)*X.T*Yc
[g,h]= (X.T*X)**(-1)*X.T*Ybac
[i,j]= (X.T*X)**(-1)*X.T*Yms

# LAD
dt_gs= pd.concat([rf.mktrf, GS], axis=1).fillna(method='ffill')
dt_jp= pd.concat([rf.mktrf, JPM], axis=1).fillna(method='ffill')
dt_c= pd.concat([rf.mktrf, C], axis=1).fillna(method='ffill')
dt_bac= pd.concat([rf.mktrf, BAC], axis=1).fillna(method='ffill')
dt_ms= pd.concat([rf.mktrf, MS], axis=1).fillna(method='ffill')

from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.formula.api as smf

mod_gs = smf.quantreg('mktrf~ GS', dt_gs)
res_gs = mod_gs.fit(q=.5)
print(res_gs.summary())

mod_jp = smf.quantreg('mktrf~ JPM', dt_jp)
res_jp = mod_jp.fit(q=.5)
print(res_jp.summary())

mod_c = smf.quantreg('mktrf~ C', dt_c)
res_c = mod_c.fit(q=.5)
print(res_c.summary())

mod_bac = smf.quantreg('mktrf~ BAC', dt_bac)
res_bac = mod_bac.fit(q=.5)
print(res_bac.summary())

mod_ms = smf.quantreg('mktrf~ MS', dt_ms)
res_ms = mod_ms.fit(q=.5)
print(res_ms.summary())
#Shrinkage
from sklearn.linear_model import Lasso

lassoReg = Lasso(alpha=-0.002,fit_intercept= True, normalize=True)
lassoReg.fit(mkr,Ygs)
print(lassoReg.coef_)
print(lassoReg.intercept_)

lassoReg = Lasso(alpha=0.0031176,fit_intercept= True, normalize=True)
lassoReg.fit(mkr,Yjp)
print(lassoReg.coef_)
print(lassoReg.intercept_)

lassoReg = Lasso(alpha=-0.018568,fit_intercept= True, normalize=True)
lassoReg.fit(mkr,Yc)
print(lassoReg.coef_)
print(lassoReg.intercept_)

lassoReg = Lasso(alpha=0.02,fit_intercept= True, normalize=True)
lassoReg.fit(mkr,Ybac)
print(lassoReg.coef_)
print(lassoReg.intercept_)

lassoReg = Lasso(alpha=-0.00476,fit_intercept= True, normalize=True)
lassoReg.fit(mkr,Yms)
print(lassoReg.coef_)
print(lassoReg.intercept_)


#Bayesian Estimator
import pymc3 as pm

# Context for the model
with pm.Model() as normal_model:
    
    # The prior for the data likelihood is assumed a Normal Distribution
    prior = pm.glm.families.Normal()
    
    # Creating the model requires a formula and data (and optionally a family)
    pm.GLM.from_formula('mktrf~GS', data = dt_gs, family = prior)
    
    # Perform Markov Chain Monte Carlo sampling letting PyMC3 choose the algorithm
    normal_trace = pm.sample(draws=2000, chains = 2, tune = 750, njobs=-1)
    print(pm.summary(normal_trace))
    pm.plot_posterior(normal_trace)
 
with pm.Model() as normal_model:
    # The prior for the data likelihood is assumed a Normal Distribution
    prior = pm.glm.families.Normal()    
    pm.GLM.from_formula('mktrf~JPM', data = dt_jp, family = prior)
    normal_trace = pm.sample(draws=2000, chains = 2, tune = 750, njobs=-1)
    pm.plot_posterior(normal_trace)
    
with pm.Model() as normal_model:

    prior = pm.glm.families.Normal()    
    pm.GLM.from_formula('mktrf~C', data = dt_c, family = prior)
    normal_trace = pm.sample(draws=2000, chains = 2, tune = 750, njobs=-1)
    pm.plot_posterior(normal_trace)

with pm.Model() as normal_model:

    prior = pm.glm.families.Normal()    
    pm.GLM.from_formula('mktrf~BAC', data = dt_bac, family = prior)
    normal_trace = pm.sample(draws=2000, chains = 2, tune = 750, njobs=-1)
    pm.plot_posterior(normal_trace)
    
with pm.Model() as normal_model:

    prior = pm.glm.families.Normal()    
    pm.GLM.from_formula('mktrf~BAC', data = dt_bac, family = prior)
    normal_trace = pm.sample(draws=2000, chains = 2, tune = 750, njobs=-1)
    pm.plot_posterior(normal_trace)
#t-test
model= sm.OLS(Ygs,X)
print(model.fit().summary())

model= sm.OLS(Yjp,X)
print(model.fit().summary())

model= sm.OLS(Yc,X)
print(model.fit().summary())

model= sm.OLS(Ybac,X)
print(model.fit().summary())

model= sm.OLS(Yms,X)
print(model.fit().summary())

#Rolling Estimation 60 month
from pyfinance import ols
rolling = ols.PandasRollingOLS(y= GS, x=rf.mktrf, window=60)
roll= pd.concat([rolling.alpha, rolling.beta, rf.mktrf], axis=1)
pred=roll.intercept + roll.feature1*roll.mktrf #feature1 is the Beta
roll['pred']=pred.shift()
roll['GS']=GS
roll_60=roll.reset_index()
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_gs= rmse(roll.pred,roll.GS)
print(rmse_gs*100)
x=roll.intercept
plt.plot(roll.index,x, color='blue', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('alpha')
plt.show()
y=roll.feature1
plt.plot(roll.index,y, color='green', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('beta')
plt.show()

rollingJPM = ols.PandasRollingOLS(y= JPM, x=rf.mktrf, window=60)
roll_JPM= pd.concat([rollingJPM.alpha, rollingJPM.beta, rf.mktrf], axis=1)
pred_JPM=roll_JPM.intercept + roll_JPM.feature1*roll_JPM.mktrf
roll_JPM['pred']=pred_JPM.shift()
roll_JPM['JPM']=JPM
roll_60=roll_JPM.reset_index()
rmse_JPM= rmse(roll_JPM.pred,roll_JPM.JPM)
print(rmse_JPM*100)
x=roll.intercept
plt.plot(roll.index,x, color='blue', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('alpha')
plt.show()
y=roll.feature1
plt.plot(roll.index,y, color='green', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('beta')
plt.show()

rollingC = ols.PandasRollingOLS(y= C, x=rf.mktrf, window=60)
roll_C= pd.concat([rollingC.alpha, rollingC.beta, rf.mktrf], axis=1)
pred_C=roll_C.intercept + roll_C.feature1*roll_C.mktrf
roll_C['pred']=pred_C.shift()
roll_C['C']=C
roll_60=roll_C.reset_index()
rmse_C= rmse(roll_C.pred,roll_C.C)
print(rmse_C*100)
x=roll.intercept
plt.plot(roll.index,x, color='blue', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('alpha')
plt.show()
y=roll.feature1
plt.plot(roll.index,y, color='green', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('beta')
plt.show()


rollingBAC = ols.PandasRollingOLS(y= BAC, x=rf.mktrf, window=60)
roll_BAC= pd.concat([rollingBAC.alpha, rollingBAC.beta, rf.mktrf], axis=1)
pred_BAC=roll_BAC.intercept + roll_BAC.feature1*roll_BAC.mktrf
roll_BAC['pred']=pred_BAC.shift()
roll_BAC['BAC']=BAC
roll_60=roll_BAC.reset_index()
rmse_BAC= rmse(roll_BAC.pred,roll_BAC.BAC)
print(rmse_BAC*100)
xx=roll.intercept
plt.plot(roll.index,xx, color='blue', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('alpha')
plt.show()
yy=roll.feature1
plt.plot(roll.index,yy, color='green', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('beta')
plt.show()

rollingMS = ols.PandasRollingOLS(y= MS, x=rf.mktrf, window=60)
roll_MS= pd.concat([rollingMS.alpha, rollingMS.beta, rf.mktrf], axis=1)
pred_MS=roll_MS.intercept + roll_MS.feature1*roll_MS.mktrf
roll_MS['pred']=pred_MS.shift()
roll_MS['MS']=MS
roll_60=roll_MS.reset_index()
rmse_MS= rmse(roll_MS.pred,roll_MS.MS)
print(rmse_MS*100)
xms=roll.intercept
plt.plot(roll.index,xms, color='blue', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('alpha')
plt.show()
yms=roll.feature1
plt.plot(roll.index,yms, color='green', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('beta')
plt.show()

#Rolling estimation cumulative
rolling = ols.PandasRollingOLS(y= GS, x=rf.mktrf, window=2)
roll= pd.concat([rolling.alpha, rolling.beta, rf.mktrf], axis=1)
pred=roll.intercept + roll.feature1*roll.mktrf #feature1 is the Beta
roll['pred']=pred.shift()
roll['GS']=GS
roll_60=roll.reset_index()
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_gs= rmse(roll.pred,roll.GS)
print(rmse_gs*100)
xms=roll.intercept
plt.plot(roll.index,xms, color='blue', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('alpha')
plt.show()
yms=roll.feature1
plt.plot(roll.index,yms, color='green', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('beta')
plt.show()

rollingJPM = ols.PandasRollingOLS(y= JPM, x=rf.mktrf, window=2)
roll_JPM= pd.concat([rollingJPM.alpha, rollingJPM.beta, rf.mktrf], axis=1)
pred_JPM=roll_JPM.intercept + roll_JPM.feature1*roll_JPM.mktrf
roll_JPM['pred']=pred_JPM.shift()
roll_JPM['JPM']=JPM
roll_60=roll_JPM.reset_index()
rmse_JPM= rmse(roll_JPM.pred,roll_JPM.JPM)
print(rmse_JPM*100)
xms=roll.intercept
plt.plot(roll.index,xms, color='blue', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('alpha')
plt.show()
yms=roll.feature1
plt.plot(roll.index,yms, color='green', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('beta')
plt.show()

rollingC = ols.PandasRollingOLS(y= C, x=rf.mktrf, window=2)
roll_C= pd.concat([rollingC.alpha, rollingC.beta, rf.mktrf], axis=1)
pred_C=roll_C.intercept + roll_C.feature1*roll_C.mktrf
roll_C['pred']=pred_C.shift()
roll_C['C']=C
roll_60=roll_C.reset_index()
rmse_C= rmse(roll_C.pred,roll_C.C)
print(rmse_C*100)
xms=roll.intercept
plt.plot(roll.index,xms, color='blue', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('alpha')
plt.show()
yms=roll.feature1
plt.plot(roll.index,yms, color='green', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('beta')
plt.show()

rollingBAC = ols.PandasRollingOLS(y= BAC, x=rf.mktrf, window=2)
roll_BAC= pd.concat([rollingBAC.alpha, rollingBAC.beta, rf.mktrf], axis=1)
pred_BAC=roll_BAC.intercept + roll_BAC.feature1*roll_BAC.mktrf
roll_BAC['pred']=pred_BAC.shift()
roll_BAC['BAC']=BAC
roll_60=roll_BAC.reset_index()
rmse_BAC= rmse(roll_BAC.pred,roll_BAC.BAC)
print(rmse_BAC*100)
xms=roll.intercept
plt.plot(roll.index,xms, color='blue', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('alpha')
plt.show()
yms=roll.feature1
plt.plot(roll.index,yms, color='green', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('beta')
plt.show()

rollingMS = ols.PandasRollingOLS(y= MS, x=rf.mktrf, window=60)
roll_MS= pd.concat([rollingMS.alpha, rollingMS.beta, rf.mktrf], axis=1)
pred_MS=roll_MS.intercept + roll_MS.feature1*roll_MS.mktrf
roll_MS['pred']=pred_MS.shift()
roll_MS['MS']=MS
roll_60=roll_MS.reset_index()
rmse_MS= rmse(roll_MS.pred,roll_MS.MS)
print(rmse_MS*100)
xms=roll.intercept
plt.plot(roll.index,xms, color='blue', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('alpha')
plt.show()
yms=roll.feature1
plt.plot(roll.index,yms, color='green', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('beta')
plt.show()


rollingMS = ols.PandasRollingOLS(y= MS, x=rf.mktrf, window=2)
roll_MS= pd.concat([rollingMS.alpha, rollingMS.beta, rf.mktrf], axis=1)
pred_MS=roll_MS.intercept + roll_MS.feature1*roll_MS.mktrf
roll_MS['pred']=pred_MS.shift()
roll_MS['MS']=MS
roll_60=roll_MS.reset_index()
rmse_MS= rmse(roll_MS.pred,roll_MS.MS)
print(rmse_MS*100)
xms=roll.intercept
plt.plot(roll.index,xms, color='blue', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('alpha')
plt.show()
yms=roll.feature1
plt.plot(roll.index,yms, color='green', marker='o',linewidth=2, markersize=4)
plt.xlabel('Date')
plt.ylabel('beta')
plt.show()





