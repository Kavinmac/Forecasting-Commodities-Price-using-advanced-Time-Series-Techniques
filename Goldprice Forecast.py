#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Loading Libraries & Data

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


data = pd.read_csv("goldprice.csv")
data.info


# # Data Preparation

# In[15]:


print('Shape of data',data.shape)


#    #Data is from 3 Jan 2000 to 18 November 2022 with 5759 records.

# In[16]:


data.head()


# In[17]:


data.tail()


# In[18]:


data['GOLD.Close'].plot(figsize=(12,4))


# In[ ]:


# Check if a Time Series is Stationary or Not

    # A TS is said to be stationary if its statistical properties such as mean, variance remain constant over time.

    # 2 methods to check stationary time series: Plotting Rolling Stats & Dickey - Fuller Test.

    # Null Hypothesis: TS is non-Stationary 


# In[19]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def ad_test(goldprice):
    
    rolmean = goldprice.rolling(25).mean()
    rolstd = goldprice.rolling(25).std()
    
    plt.figure(figsize = (20,10))
    orig = plt.plot(goldprice, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    print('Results of Dickey-Fuller Test:')
    
    datatest = adfuller(goldprice, autolag = 'AIC')
    dataoutput = pd.Series(datatest[0:4], index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in datatest[4].items():
        dataoutput['Critical Value (%s)'%key] = value
        print(dataoutput)
    


# In[20]:


ad_test(data['GOLD.Close'])


# In[21]:


ts_sqrt = np.sqrt(data['GOLD.Close'])
expwighted_avg = ts_sqrt.ewm(halflife = 25).mean()

ts_sqrt_ewma_diff = ts_sqrt - expwighted_avg
ad_test(ts_sqrt_ewma_diff)


# In[22]:


ts_sqrt_diff = ts_sqrt - ts_sqrt.shift()

plt.figure(figsize = (20,10))
plt.plot(ts_sqrt_diff)
plt.show()


# In[23]:


ts_sqrt = np.sqrt(data['GOLD.Close'])
ts_sqrt_diff = ts_sqrt - ts_sqrt.shift()
ts_sqrt_diff.dropna(inplace = True)
ad_test(ts_sqrt_diff)


# In[ ]:


# Test Statistic < Critical Value and also there is less diversion in mean and std. This is perfect stationary ts.


# # Data Modeling
# 
# # Train Test Split

# In[24]:


print(data.shape)
train = data[:-540]
test = data[-540:]
print("Length of Train data: ",len(train))
print("Length of Test data: ",len(test))


# In[25]:


train.head(2)


# In[26]:


train.tail(2)


# In[27]:


test.head(2)


# In[28]:


test.tail(2)


# In[29]:


ax = train.plot(figsize = (20, 10), color = 'b')
test.plot(ax = ax, color = 'black')
plt.legend(['train set', 'test set'])
plt.show()


# # Arima: Auto Regressive Integrated Moving Average
# 
#     Is actually a class of models that ‘explains’ a given time series based on its own past values, that is, its own lags and the lagged forecast errors, so that equation can be used to forecast future values.
# 
#     Prerequisite: Data must be stationary.
# 
#     An ARIMA model is characterized by 3 terms: p: order of the AR term , q: MA term, d is the number of differencing required to make the time series stationary.

# # Finding d: Number of differences required.

# In[30]:


plt.rcParams.update({'figure.figsize':(15,10), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(4, 2, sharex=True)
axes[0, 0].plot(data['GOLD.Close']); axes[0, 0].set_title('Original Series')
plot_acf(data['GOLD.Close'], ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(data['GOLD.Close'].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(data['GOLD.Close'].diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(data['GOLD.Close'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(data['GOLD.Close'].diff().diff().dropna(), ax=axes[2, 1])

# 3rd Differencing
axes[3, 0].plot(data['GOLD.Close'].diff().diff().diff()); axes[3, 0].set_title('3nd Order Differencing')
plot_acf(data['GOLD.Close'].diff().diff().diff().dropna(), ax=axes[3, 1])

plt.show()


# # Finding order of AR term (p)
#     
#     We will find out the required number of AR terms by inspecting the Partial Autocorrelation (PACF) plot.
# 
#     Partial autocorrelation can be imagined as the correlation between the series and its lag.

# In[31]:


plt.rcParams.update({'figure.figsize':(15,7), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(data['GOLD.Close'].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(data['GOLD.Close'].diff().dropna(), ax=axes[1])

plt.show()


# # Finding order of MA term (q)
# 
#     An MA term is technically, the error of the lagged forecast.

# In[32]:


plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)

axes[0].plot(data['GOLD.Close'].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(data['GOLD.Close'].diff().dropna(), ax=axes[1])

plt.show()


# In[33]:


target= train['GOLD.Close']


# In[34]:


target


# In[35]:


test1=target[-540:]


# In[36]:


from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA


# In[37]:


model = sm.tsa.arima.ARIMA(target, order = (1, 2, 1))
arima_model = model.fit() 
print(arima_model.summary())


# In[38]:


target.shape


# In[39]:


yp_train = arima_model.predict(start = 0, end = (len(train)-1))
yp_test = arima_model.predict(start = 0, end = (len(test)-1)) 

print("Train Data:\nMean Square Error: {}".format(mean_squared_error(target, yp_train)))
print("\nTest Data:\nMean Square Error: {}".format(mean_squared_error(test1, yp_test)))


# In[40]:


print(yp_test)


# In[41]:


print(test1)


# In[42]:


print(yp_train)


# In[44]:


print(target)

