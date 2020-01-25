# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:07:17 2019

@author: Jingyu Lu
"""

import numpy as np

rf = 0.03
S0 = 100
sigma = 0.15
mu = 0.10
T = 1 # time to maturaty
simulations=250 #1000, 10000 50,100,250,
K=100
    
n=252
i=0
j= 0
option_data = np.zeros([n+3, simulations])
while j < simulations:
    i = 0
    option_data[i, j] = S0  # ST price
    while i<n-1:
        option_data[i+1,j] = option_data[i,j]*np.exp((1/252)*(mu-0.5*sigma**2)+sigma*np.sqrt(1/252)*np.random.normal(0,1))
        i=i+1
    option_data[n,j] = min(option_data[0:n-1,j]) # min price for the simulation of the whole one year path
    option_data[n+1,j] = max(option_data[n-1,j] - option_data[n,j], 0) # payoff for floating-strike call option: max(ST-Smin,0)
    option_data[n+2,j] = max(K - option_data[n,j], 0) # payoff for fixed-strike put optiion: max(K-Smin,0)
    j=j+1
    
call_option_price_250 = np.exp(-rf*T)*(sum(option_data[n+1,:])/simulations)
put_option_price_250 = np.exp(-rf*T)*(sum(option_data[n+2,:])/simulations)

call_option_price_50 = np.exp(-rf * T) * (sum(option_data[n + 1, :]) / simulations)
put_option_price_50 = np.exp(-rf * T) * (sum(option_data[n + 2, :]) / simulations)

call_option_price_25 = np.exp(-rf * T) * (sum(option_data[n + 1, :]) / simulations)
put_option_price_25 = np.exp(-rf * T) * (sum(option_data[n + 2, :]) / simulations)

call_option_price_10 = np.exp(-rf * T) * (sum(option_data[n + 1, :]) / simulations)
put_option_price_10 = np.exp(-rf * T) * (sum(option_data[n + 2, :]) / simulations)

call_option_price_10000 = np.exp(-rf * T) * (sum(option_data[n + 1, :]) / simulations)
put_option_price_10000 = np.exp(-rf * T) * (sum(option_data[n + 2, :]) / simulations)

call_option_price_1000 = np.exp(-rf * T) * (sum(option_data[n + 1, :]) / simulations)
put_option_price_1000 = np.exp(-rf * T) * (sum(option_data[n + 2, :]) / simulations)

call_option_price_100= np.exp(-rf * T) * (sum(option_data[n + 1, :]) / simulations)
put_option_price_100 = np.exp(-rf * T) * (sum(option_data[n + 2, :]) / simulations)
call = [call_option_price_10, call_option_price_25,call_option_price_50,call_option_price_100,call_option_price_250,call_option_price_1000, call_option_price_10000]
put = [put_option_price_10, put_option_price_25,put_option_price_50,put_option_price_100,put_option_price_250,put_option_price_1000, put_option_price_10000]

#analytic formula
#version 1, mp0 = S0
mp0 = S0
N = norm.cdf
d1 = 1/(np.sqrt(sigma**2 * 1)) * (np.log(S0/mp0) + r*1 + 1/2 * sigma**2 * 1)
d2_Lin = d1 - sigma*np.sqrt(1)
E = S0*N(d1) - np.exp(-r*1)*mp0 * N(d2_Lin)+np.exp(-r)*(sigma**2 /(2*r))*S0*(((S0/mp0)**(-2*r/sigma**2))*N(-d1+(2*r*np.sqrt(1/sigma**2)))-np.exp(r)*N(-d1))


