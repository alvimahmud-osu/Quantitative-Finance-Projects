#-----------------------------------------------------------------------------#
#Louis R. Piccotti
#Assistant Professor of Finance
#Director, M.S. Quantitative Financial Economics
#Spears School of Business
#Oklahoma State University
#This version: 03.04.2019
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
#RND for the April 18, 2018 options maturity date.
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
#Notes:1.Obtain the zero rate and the zero coupon bond price from running the
#        zero curve bootstrappig code.
#-----------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import dateutil
import bs4 as bs
import requests
from dateutil.parser import parse
from scipy.stats import norm

#----------------------------------------------------------------------------#
#Manual inputs (These need to be automated).
#----------------------------------------------------------------------------#

indexlevel=pd.to_numeric(input("Current index level="))#Manually insert S&P 500 index level
ttm=pd.to_numeric(input("Time to maturity (years)="))#Manually insert time-to-maturity
r=pd.to_numeric(input("Zero rate=")) #Manually insert the zero rate with the ttm
#Manually insert the zero coupon bond price with the ttm
zb=pd.to_numeric(input("Zero-coupon bond price="))
#Manually insert the S&P 500 per annum dividend yield
divyld=pd.to_numeric(input("S&P 500 dividend yield="))

#----------------------------------------------------------------------------#
#Reads in the options data table and formats it.
#----------------------------------------------------------------------------#

today=datetime.datetime.today().strftime('%Y-%m-%d')
#URL for the April 18 S&P 500 options. I am not sure if this link updates to 
#the one-month forward contract or if it will die after April 18. We will
#see.
url="https://finance.yahoo.com/quote/%5EGSPC/options?p=%5EGSPC&date=1555545600"
data=pd.read_html(url) #Reads in the data tables.
calls=data[0] #Selects the table of call options
calls['Moneyness']=calls.Strike/indexlevel #Formats strikes in terms of moneyness
2#Converts IV object to a IV numeric in decimal form.
calls['IV']=calls['Implied Volatility'].str.replace('%', '').astype(float)/100
#Finds the first IV that is non-missing (>0.0%)
i=0
while i<=len(calls):
    if calls.IV[i]>0:
        st=i
        i=len(calls)
    i=i+1
calls=calls[st:len(calls)]
calls.index=np.arange(1,len(calls)+1)
calls['Mid']=0.5*(calls.Bid+calls.Ask)

#----------------------------------------------------------------------------#
#Plots call option implied volatilities
#----------------------------------------------------------------------------#

plt.plot(calls.Moneyness[1:len(calls)],calls.IV[1:len(calls)],color='black',marker='o',fillstyle='none',linestyle='none')
plt.title("Implied volatility (call options)"+" ("+today+")")
plt.xlabel("Option moneyness")
plt.ylabel("Implied volatility")
plt.show()

#----------------------------------------------------------------------------#
#Fits spline regression to IV
#----------------------------------------------------------------------------#

yy=np.matrix(calls.IV).T
#3rd degree polynomial spline with 6 knots: 0.70,0.80,0.90,0.95,1.05,1.10
xx=np.concatenate((np.matrix(np.ones(len(calls))),np.matrix(calls.Moneyness),
    np.matrix(calls.Moneyness**2),np.matrix(calls.Moneyness**3),
    np.matrix((((calls.Moneyness-0.70)+0+abs(0-(calls.Moneyness-0.70)))/2)**3),
    np.matrix(((((calls.Moneyness-0.80)+0+abs(0-(calls.Moneyness-0.80)))/2)**3)),
    np.matrix(((((calls.Moneyness-0.90)+0+abs(0-(calls.Moneyness-0.90)))/2)**3)),
    np.matrix(((((calls.Moneyness-0.95)+0+abs(0-(calls.Moneyness-0.95)))/2)**3)),
    np.matrix(((((calls.Moneyness-1.05)+0+abs(0-(calls.Moneyness-1.05)))/2)**3)),
    np.matrix(((((calls.Moneyness-1.10)+0+abs(0-(calls.Moneyness-1.10)))/2)**3))),
    axis=0).T
beta=(xx.T*xx)**(-1)*xx.T*yy #OLS estimates of the coefficients
calls['IVfit']=xx*beta #Model-fit IV.

#Plots the IV model fit
plt.plot(calls.Moneyness,calls.IV,color='black',marker='o',fillstyle='none',linestyle='none')
plt.plot(calls.Moneyness,calls.IVfit,color='black')
plt.title("Implied volatility fit (call options)"+" ("+today+")")
plt.xlabel("Option moneyness")
plt.ylabel("Implied volatility")
plt.show()

#----------------------------------------------------------------------------#
#Estimates the RND with the fitted IV spline model from above.
#----------------------------------------------------------------------------#

Mongrid=np.matrix(np.arange(np.min(calls.Moneyness),np.max(calls.Moneyness),0.005)).T
Ivgrid=np.matrix(np.zeros(len(Mongrid))).T
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
    bsd1[i,0]=(1/(Ivgrid[i,0]*np.sqrt(ttm)))*(np.log(indexlevel/(Mongrid[i,0]*indexlevel))+(r-divyld+Ivgrid[i,0]**2/2)*ttm)
    i=i+1
bsd2=bsd1-Ivgrid*np.sqrt(ttm) #d2
fwd=indexlevel*np.exp((r-divyld)*ttm) #Forward index level
#Black-Scholes option prices with continuously paid dividend yield
bsc=zb*(fwd*norm.cdf(bsd1)-np.multiply(norm.cdf(bsd2),Mongrid*indexlevel))

fdif=np.matrix(np.zeros(len(Mongrid))).T
sdif=np.matrix(np.zeros(len(Mongrid))).T
#This loop computes the 1st and 2nd derivatives via finite differences
i=1
while i<len(Mongrid):
    h=indexlevel*(Mongrid[i,0]-Mongrid[i-1,0])
    fdif[i,0]=(bsc[i,0]-bsc[i-1,0])/h
    if i>=2:
        sdif[i,0]=(fdif[i,0]-fdif[i-1,0])/h
    i=i+1
#Estimates the RND and divides the by sum of all estimated probabilities so that
#the density sums to 1.
rnd=np.exp(r*ttm)*sdif

#----------------------------------------------------------------------------#
#Numeric integration of the RND.

#Note:1.This should be approximately 1.  It will vary a bit due to numerical
#       errors and grid discreteness.
#     2.Alternatively, a monte-carlo integration could be used.
#----------------------------------------------------------------------------#

#Numeric integral (composite rule)
rndint=np.sum(rnd)*(np.max(Mongrid)-np.min(Mongrid))/(len(Mongrid)-1)*indexlevel
#Cumulative RND ( int_{min(Strike)}^{x} RND(u)du )
rncd=np.cumsum(rnd).T*(np.max(Mongrid)-np.min(Mongrid))/(len(Mongrid)-1)*indexlevel 

#----------------------------------------------------------------------------#
#Plots call option implied probability and cumulative density functions
#----------------------------------------------------------------------------#

plt.plot(Mongrid,rnd,color='black')
plt.title("Implied Risk-neutral pdf (call options)"+" ("+today+")")
plt.xlabel("Option moneyness")
plt.ylabel("Density")
plt.show()

plt.plot(Mongrid,rncd,color='black')
plt.title("Implied Risk-neutral cdf (call options)"+" ("+today+")")
plt.xlabel("Option moneyness")
plt.ylabel("Density")
plt.show()

k=input("press close to exit") 