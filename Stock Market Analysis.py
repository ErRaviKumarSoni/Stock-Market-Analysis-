#1.Dataset
#2.Resampling
#3.Moving Windows
#4.Volatility

#StepNo.1
import os
import pandas as pd
amd = pd.read_csv("D:\\INTERVIEW\\2.Project\\Stock Analysis\\AMD.csv")

amd = pd.read_csv('D:\\INTERVIEW\\2.Project\\Stock Analysis\\AMD.csv',header=0, index_col='Date', parse_dates=True)
amd.head()
#!pip install pandas_datareader
import pandas_datareader as pdr
import datetime
nvda = pdr.get_data_yahoo('NVDA', 
			start=datetime.datetime(2004, 1, 1),
			end=datetime.datetime(2019, 9, 15))


qcom= pdr.get_data_yahoo('QCOM', 
			start=datetime.datetime(2004, 1, 1),
			end=datetime.datetime(2019, 9, 15))


intc = pdr.get_data_yahoo('INTC', 
			start=datetime.datetime(2004, 1, 1),
			end=datetime.datetime(2019, 9, 15))


ibm = pdr.get_data_yahoo('IBM', 
			start=datetime.datetime(2004, 1, 1),
			end=datetime.datetime(2019, 9, 15))
type(nvda)
ibm.tail()
ibm.describe()
nvda.columns()
nvda.index, amd.index
nvda.shape

##Time Series Data Analysis
import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.dates as mdates

plt.plot(ibm.index, ibm['Adj Close'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.grid(True)
plt.xticks(rotation=90)
plt.show()

#subplots
f,ax = plt.subplots(2, 2, figsize=(10,10), shrex=True)
f.gca().xaxis.set_major_formatter(mdates.DataFormatter('%Y'))
f.gca().xaxis.set_major_locator(mdates.YearLocator())

ax[0,0].plot((nvda.index, nvda['Adj Close'], color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');

ax[0,1].plot((nvda.index, intc['Adj Close'], color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');

ax[1,0].plot((nvda.index, qcom['Adj Close'], color='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUALCOMM');

ax[1,1].plot((nvda.index, amd['Adj Close'], color='y')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');


#Step No.2
#Resampling
#Zooming-in

ibm_18 = ibm.loc[pd.Timestamp('2018-01-01'):pd.Timestamp('2018-12-31')]
plt.plot(ibm_18.index, ibm_18['Adj Close'])
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DataFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()

#Subplot
f, ax = plt.subplots(2, 2, figsize=(10,10), shrex=True, sharey=True)
f.gca().xaxis.set_major_formatter(mdates.DataFormatter('%Y-%m'))
f.gca().xaxis.set_major_locator(mdates.MonthLocator())

nvda_18 = nvda.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[0,0].plot((nvda_18.index, nvda_18['Adj Close'], '.', color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');

intc_18 = intc.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[0,1].plot((intc_18.index, intc_18['Adj Close'], colur='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');

qcom_18 = qcom.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[1,0].plot((qcom_18.index, qcom_18['Adj Close'], colur='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUALCOMM');

amd_18 = amd.loc[pd.Timestamp('2017-11-01'):pd.Timestamp('2018-12-31')]
ax[1,1].plot((amd_18.index, amd_18['Adj Close'], colur='b')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');

#Step 2 - Resampling-Quaterly 
monthly_nvda_18 = nvda_18.resample('4M').mean()
plt.scatter(monthly_nvda_18.index, monthly_nvda.18['Adj Close']
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DataFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()

#Subplot
f, ax = plt.subplots(2, 2, figsize=(10,10), shrex=True, sharey=True)

monthly_nvda_18 = nvda_18.resample('4M').mean()
ax[0,0].scatter((monthly_nvda_18.index, monthly_nvda_18['Adj Close'], color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');

monthly_intc_18 = intc_18.resample('4M').mean()
ax[0,1].scatter((monthly_intc_18.index, monthly_intc_18['Adj Close'], color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');

monthly_qcom_18 = qcom_18.resample('4M').mean()
ax[1,0].scatter((monthly_qcom_18.index, monthly_qcom_18['Adj Close'], colur='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUALCOMM');

monthly_amd_18 = amd_18.resample('4M').mean()
ax[1,1].scatter((monthly_amd_18.index, monthly_amd_18['Adj Close'], colur='y')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');

# Resampling - Weekly
ibm_19 = ibm.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_ibm_19 = ibm_19.resample('W').mean()
weekly_ibm_19.head()

plt.plot(weekly_ibm_19.index, weekly_ibm_19['Adj Close'], '-o')
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DataFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()


#Subplots
nvda_19 = nvda.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_nvda_19 = weekly_nvda_19.resample('W').mean()

intc_19 = intc.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_intc_19 = weekly_intc_19.resample('W').mean()

qcom_19 = qcom.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_qcom_19 = weekly_qcom_19.resample('W').mean()

amd_19 = amd.loc[pd.Timestamp('2019-01-15'):pd.Timestamp('2019-09-15')]
weekly_amd_19 = weekly_amd_19.resample('W').mean()

f, ax = plt.subplots(2, 2, figsize=(10,10), shrex=True, sharey=True)

#NVIDIA
ax[0,0].plot((weekly_nvda_19, weekly_nvda_19['Adj Close'], '.', color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');

#INTEL
ax[0,1].plot((weekly_intc_19.index, weekly_intc_19['Adj Close'], color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');

#QCOM
ax[1,0].plot((weekly_qcom_19.index, weekly_qcom_19['Adj Close'], color='b')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUALCOMM');

#AMD
ax[1,1].plot((weekly_amd_19.index, weekly_amd_19['Adj Close'], color='b')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');

#Analysing difference between levels (Resampling Weekly)
ibm['diff'] = ibm['open'] - ibm['Close']
ibm_diff = ibm.resample('W').mean()
ibm_diff.tail(10)

plt.scatter(ibm_diff.loc['2019-01-01':'2019-09-15'].index,ibm_diff.loc['2019-01-01':'2019-09-15']
plt.gca().xaxis.set_major_formatter(mdates.DataFormatter('%Y-%m'-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.xticks(rotation=90)
plt.show()       

#Subplots

nvda['diff'] = nvda['Open'] - nvda['Close']
nvda_diff = nvda.resample('W').mean()

intc['diff'] = intc['Open'] - intc['Close']
intc_diff = intc.resample('W').mean()

qcom['diff'] = qcom['Open'] - qcom['Close']
qcom_diff = qcom.resample('W').mean()

amd['diff'] = amd['Open'] - amd['Close']
amd_diff = amd.resample('W').mean()

f, ax = plt.subplots(2, 2, figsize=(10,10), shrex=True, sharey=True)

#NVIDIA
ax[0,0].scatter(nvda_diff.loc['2019-01-01':'2019-09-15'].index, nvda_diff.loc['2019-01-01':'2019-09-15'], color='r')
ax[0,0].grid(True)
ax[0,0].tick_params(labelrotation=90)
ax[0,0].set_title('NVIDIA');

#INTEL
ax[0,1].scatter(intc_diff.loc['2019-01-01':'2019-09-15'].index, intc_diff.loc['2019-01-01':'2019-09-15'], color='g')
ax[0,1].grid(True)
ax[0,1].tick_params(labelrotation=90)
ax[0,1].set_title('INTEL');

#QCOM
ax[1,0].scatter(qcom_diff.loc['2019-01-01':'2019-09-15'].index, qcom_diff.loc['2019-01-01':'2019-09-15'], color='g')
ax[1,0].grid(True)
ax[1,0].tick_params(labelrotation=90)
ax[1,0].set_title('QUALCOMM');

#AMD
ax[1,1].scatter(amd_diff.loc['2019-01-01':'2019-09-15'].index, amd_diff.loc['2019-01-01':'2019-09-15'], color='g')
ax[1,1].grid(True)
ax[1,1].tick_params(labelrotation=90)
ax[1,1].set_title('AMD');


# 3. Moving Windows
# Daily percentages
daily_close_ibm = ibm[['Adj Close']]

#Daily Returns
daily_pct_change_ibm = daily_close_ibm.pct_change()

#Replace NA values with 0
daily_pct_change_ibm.fillna(0, inplace=True)
daily_pct_change_ibm.head()

daily_pct_change_ibm.hist(bins=50)
plt.show()

#Subplot
#NVIDIA
daily_close_nvda = nvda[['Adj Close']]
daily_pct_change_nvda = daily_close_nvda.pct_change()
daily_pct_change_nvda.fillna(0, inplace=True)


#INTEL
daily_close_intc = intc[['Adj Close']]
daily_pct_change_intc = daily_close_intc.pct_change()
daily_pct_change_intc.fillna(0, inplace=True)


#QUALCOMM
daily_close_qcom = qcom[['Adj Close']]
daily_pct_change_qcom = daily_close_qcom.pct_change()
daily_pct_change_qcom.fillna(0, inplace=True)

#AMD
daily_close_amd = amd[['Adj Close']]
daily_pct_change_amd = daily_close_amd.pct_change()
daily_pct_change_amd.fillna(0, inplace=True)

import seaborn as sns
sns.set()
import seaborn as sns
# Setup the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(12, 7))

#Plot a simple histogram with binsize determined automatically
sns.displot(daily_pct_change_nvda['Adj Close'], color="b", ax=axes[0, 0], axlabel='NVIDA')

#Plot a kernel density estimate and rug plot
sns.displot(daily_pct_change_intc['Adj Close'], color="r", ax=axes[0, 1], axlabel='INTEL')


#Plot filled kernel estimate
sns.displot(daily_pct_change_qcom['Adj Close'], color="g", ax=axes[1, 0], axlabel='QUALCOMM')

#Plot a histogram & kernel density estimate
sns.displot(daily_pct_change_amd['Adj Close'], color="m", ax=axes[1, 1], axlabel='AMD')


#4. Volatility

import numpy as np

#calculate the volatility
vol = daily_pct_change_ibm.rolling(min_periods).std() * np.sqrt(min_periods)

vol.fillna(0, inplace=True)

vol.tail()

# Plot volatility
vol.plot(figsize=(10, 8))

plt.show()


#Rollinng Means (Trends and Seasonality)


ibm_adj_close_px = ibmm['Adj Close']

#Short moving windows rolling mean
ibm['42'] = ibm_adj_close_px.rolling(windows=40).mean()

#Long moving windows rolling mean
ibm['252'] = ibm_adj_close_px.rolling(windows=252).mean()

#Plot the adjusted closing price, the short and long windows of rolling means
ibm[['Adj Close', '42', '252']].plot(title='IBM')

#Show plot
plt.show()

#NIVIDIA

nvda_adj_close_px = nvda['Adj Close']
#Short moving windows rolling mean
nvda['42'] = nvda_adj_close_px.rolling(windows=40).mean()

#Long moving windows rolling mean
nvda['252'] = nvda_adj_close_px.rolling(windows=252).mean()

#Plot the adjusted closing price, the short and long windows of rolling means
nvda[['Adj Close', '42', '252']].plot(title='NVIDIA')

#Show plot
plt.show()

#INTEL

intc_adj_close_px = intc['Adj Close']
#Short moving windows rolling mean
intc['42'] = intc_adj_close_px.rolling(windows=40).mean()

#Long moving windows rolling mean
intc['252'] = intc_adj_close_px.rolling(windows=252).mean()

#Plot the adjusted closing price, the short and long windows of rolling means
intc[['Adj Close', '42', '252']].plot(title='INTEL')

#Show plot
plt.show()

#QUALCOMM
qcom_adj_close_px = qcom['Adj Close']

#Short moving windows rolling mean
qcom['42'] = qcom_adj_close_px.rolling(windows=40).mean()

#Long moving windows rolling mean
qcom['252'] = qcom_adj_close_px.rolling(windows=252).mean()

#Plot the adjusted closing price, the short and long windows of rolling means
qcom[['Adj Close', '42', '252']].plot(title='QUALCOMM')

#Show plot
plt.show()

#AMD
amd_adj_close_px = amd['Adj Close']
#Short moving windows rolling mean
amd['42'] = amd_adj_close_px.rolling(windows=40).mean()

#Long moving windows rolling mean
amd['252'] = amd_adj_close_px.rolling(windows=252).mean()

#Plot the adjusted closing price, the short and long windows of rolling means
amd[['Adj Close', '42', '252']].plot(title='AMD')

#Show plot
plt.show()


ibm.loc['2019-01-01':'2019-1-01-15'][['Adj Close','42','242']].plot(title="IBM in 2019");
nvda.loc['2019-01-01':'2019-1-01-15'][['Adj Close','42','242']].plot(title="NVIDIA in 2019");
intc.loc['2019-01-01':'2019-1-01-15'][['Adj Close','42','242']].plot(title="INTEL in 2019");
qcom.loc['2019-01-01':'2019-1-01-15'][['Adj Close','42','242']].plot(title="QUALCOMM in 2019");
amd.loc['2019-01-01':'2019-1-01-15'][['Adj Close','42','242']].plot(title="AMD in 2019");


