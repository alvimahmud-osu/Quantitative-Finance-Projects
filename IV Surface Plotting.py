import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
from scipy.stats import norm
from datetime import date

##Read in and formats options data
url1="https://finance.yahoo.com/quote/%5EGSPC/options?p=%5EGSPC&date=1555545600"
df_apr18=pd.read_html(url1) #Reads in the data tables.
d0 = date(2019, 4, 7)
d1 = date(2019, 4, 18)
Maturity = d1 - d0
calls_apr= df_apr18[0]
calls_apr['Maturity'] = pd.Series([Maturity for x in range(len(calls_apr.index))], index=calls_apr.index)

url2="https://finance.yahoo.com/quote/%5EGSPC/options?p=%5EGSPC&date=1558051200"
df_may17=pd.read_html(url2) #Reads in the data tables.
dm = date(2019, 5, 17)
Maturity_m = dm - d0
calls_may= df_may17[0]
calls_may['Maturity'] = pd.Series([Maturity_m for x in range(len(calls_may.index))], index=calls_may.index)

url3="https://finance.yahoo.com/quote/%5EGSPC/options?p=%5EGSPC&date=1561075200"
df_jun21=pd.read_html(url3) #Reads in the data tables.
d2 = date(2019, 6, 21)
Maturity2 = d2 - d0
calls_jun= df_jun21[0]
calls_jun['Maturity'] = pd.Series([Maturity2 for x in range(len(calls_jun.index))], index=calls_jun.index)

url4="https://finance.yahoo.com/quote/%5EGSPC/options?p=%5EGSPC&date=1576800000"
df_dec20=pd.read_html(url4) #Reads in the data tables.
d3 = date(2019, 12, 20)
Maturity3 = d3 - d0
calls_dec= df_dec20[0]
calls_dec['Maturity'] = pd.Series([Maturity3 for x in range(len(calls_dec.index))], index=calls_dec.index)

url5="https://finance.yahoo.com/quote/%5EGSPC/options?p=%5EGSPC&date=1592524800"
df_jun2020=pd.read_html(url5) #Reads in the data tables.
d4 = date(2020, 5, 19)
Maturity4 = d4 - d0
calls_jun20= df_jun2020[0]
calls_jun20['Maturity'] = pd.Series([Maturity4 for x in range(len(calls_jun20.index))], index=calls_jun20.index)

url6="https://finance.yahoo.com/quote/%5EGSPC/options?p=%5EGSPC&date=1608249600"
df_dec2020=pd.read_html(url6) #Reads in the data tables.
d5 = date(2020, 12, 18)
Maturity5 = d5 - d0
calls_dec20= df_dec2020[0]
calls_dec20['Maturity'] = pd.Series([Maturity5 for x in range(len(calls_dec20.index))], index=calls_dec20.index)


url7="https://finance.yahoo.com/quote/%5EGSPC/options?p=%5EGSPC&date=1639699200"
df_dec2021=pd.read_html(url7) #Reads in the data tables.
d6 = date(2021, 12, 17)
Maturity6 = d6 - d0
calls_dec21= df_dec2021[0]
calls_dec21['Maturity'] = pd.Series([Maturity6 for x in range(len(calls_dec21.index))], index=calls_dec21.index)

calls= pd.concat([calls_apr,calls_may,calls_jun,calls_dec,calls_jun20,calls_dec20,calls_dec21])

#df= df_apr18 + df_may17 + df_jun21 + df_dec20 + df_jun2020 + df_dec2020 + df_dec2021

#calls_iv=df[0] #Selects the table of call options
#Converts IV object to a IV numeric in decimal form.
#calls_iv['IV']=calls_iv['Implied Volatility'].str.replace('%', '').astype(float)/100
calls['IV']=calls['Implied Volatility'].str.replace('%', '').astype(float)/100
plt.plot(calls['Strike'],calls['IV'],color='blue',marker='o')

plt.title("Implied Volatility curve")
plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.show()

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D 
from scipy.interpolate import griddata
from matplotlib import cm
 
fig = plt.figure()
ax = Axes3D(fig, azim = -29, elev = 50)
x = np.array(pd.to_timedelta(calls['Maturity']).astype('timedelta64[D]'))
y = np.array(calls['Strike'])
#X, Y = np.meshgrid(x, y)
z= np.array(calls['IV'])
#Z = z.reshape(z.shape[0],-1)

X,Y = np.meshgrid(np.linspace(min(x),max(x),230),np.linspace(min(y),max(y),230))
Z = griddata(np.array([x,y]).T,np.array(z),(X,Y), method='linear')

ax.plot_surface( X, Y, Z, rstride=1, cstride=1, cmap= cm.coolwarm, linewidth=0.1)
ax.contour(X,Y,Z)
ax.set_xlabel('Maturity')
ax.set_ylabel("Strike")
ax.set_zlabel('Implied Volatility')

plt.show()









