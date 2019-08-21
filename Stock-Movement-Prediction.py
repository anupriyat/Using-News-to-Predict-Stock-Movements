#!/usr/bin/env python
# coding: utf-8

# # Stock Movement Prediction

# # Imports

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import *
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_inputMain, news_inputMain) = env.get_training_data()


# In[ ]:


#print(market_inputMain.head())
#print(news_inputMain.head())
market_train_df = market_inputMain.copy(deep=True)


# In[ ]:


print(len(market_train_df["assetCode"].unique()))


# In[ ]:


data = []
for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    price_df = market_train_df.groupby('time')['close'].quantile(i).reset_index()

    data.append(go.Scatter(
        x = price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = price_df['close'].values,
        name = f'{i} quantile'
    ))
layout = go.Layout(dict(title = "Trends of closing prices by quantiles",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"),
    annotations=[
        dict(
            x='2008-09-01 22:00:00+0000',
            y=82,
            xref='x',
            yref='y',
            text='Collapse of Lehman Brothers',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2011-08-01 22:00:00+0000',
            y=85,
            xref='x',
            yref='y',
            text='Black Monday',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2014-10-01 22:00:00+0000',
            y=120,
            xref='x',
            yref='y',
            text='Another crisis',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=-20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2016-01-01 22:00:00+0000',
            y=120,
            xref='x',
            yref='y',
            text='Oil prices crash',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        )
    ])
py.iplot(dict(data=data, layout=layout), filename='basic-line')


# In[ ]:


for i, j in zip([-1, 0, 1], ['negative', 'neutral', 'positive']):
    df_sentiment = news_inputMain.loc[news_inputMain['sentimentClass'] == i, 'assetName']
    print(f'Top mentioned companies for {j} sentiment are:')
    print(df_sentiment.value_counts().head(5))
    print('')


# In[ ]:


df_volumeCount = news_inputMain.loc[news_inputMain['volumeCounts7D'] > 0, 'assetName']
print(f'Top mentioned companies for {j} volumeCounts7D are:')
print(df_volumeCount.value_counts()[1:50])


# # 1. Feature Engineering

# #### **Function to merge Market & News Datasets**

# In[ ]:


#Make a deep copy to keep the main dataset. Environment cannot be restarted at will.
dfm = market_inputMain.copy(deep=True)
dfn = news_inputMain.copy(deep=True)


# In[ ]:


dfm["assetCode"].unique()


# In[ ]:


dfm = dfm[dfm["assetName"].isin([  
'Citigroup Inc'\
,'Apple Inc'\
,'JPMorgan Chase & Co'\
,'Bank of America Corp'\
,'HSBC Holdings PLC'\
,'Goldman Sachs Group Inc'\
,'Deutsche Bank AG'\
,'BHP Billiton PLC'\
,'BP PLC'\
,'Google Inc'\
,'Boeing Co'\
,'Rio Tinto PLC'\
,'Royal Dutch Shell PLC'\
,'Ford Motor Co'\
,'General Electric Co'\
,'Morgan Stanley'\
,'Microsoft Corp'\
,'Exxon Mobil Corp'\
,'UBS AG'\
,'Toyota Motor Corp'\
,'Royal Bank of Scotland Group PLC'\
,'Wal-Mart Stores Inc'\
,'BHP Billiton Ltd'\
,'General Motors Co'\
,'Verizon Communications Inc'\
,'AT&T Inc'\
,'Wells Fargo & Co'\
,'Amazon.com Inc'\
,'Lloyds Banking Group PLC'\
,'Credit Suisse Group AG'\
,'Chevron Corp'\
,'Pfizer Inc'\
,'American International Group Inc'\
,'Vodafone Group PLC'\
,'Federal Home Loan Mortgage Corp'\
,'Sony Corp'\
,'Federal National Mortgage Association'\
,'Total SA'\
,'Motors Liquidation Co'\
,'Nokia Oyj'\
,'Intel Corp'\
,'Twenty-First Century Fox Inc'\
,'Yahoo! Inc'\
,'International Business Machines Corp'\
,'GlaxoSmithKline PLC'\
,'Credit Suisse AG'\
,'Facebook Inc'\
,'HP Inc'\
,'Banco Santander SA'  ])]


# In[ ]:


market_inputMain.shape


# In[ ]:


dfm.shape


# In[ ]:


market_inputMain.head()


# In[ ]:


news_inputMain.shape


# In[ ]:


dfn.shape


# #### Cutdown datasets

# In[ ]:


#utc 
import datetime
import pytz

utc=pytz.UTC

#cut down datasets to return
startdate = pd.to_datetime("2014-01-01").replace(tzinfo=utc)
dfm = dfm[dfm.time > startdate]
dfn = dfn[dfn.time > startdate]


# ### EXPAND NEWS Dataset as each "assetCodes" field is a  list of assetCodes

# In[ ]:


#News dataset shape before expanding
news_df = dfn
news_df.shape


# In[ ]:


dfm.shape


# In[ ]:


#First five asset codes of non-expaned News Dataset
news_df["assetCodes"].head(5)


# In[ ]:


#Expanding assetCodes
from itertools import chain
news_cols = news_df.columns.values
news_df['assetCodes'] = news_df['assetCodes'].str.findall(f"'([\w\./]+)'")  
#print(chain(*news_df['assetCodes']))
assetCodes_expanded = list(chain(*news_df['assetCodes']))


# In[ ]:


assetCodeArray = dfm["assetCode"].unique()


# In[ ]:


#assetCodes_expanded = assetCodes_expanded

assetCodes_index = news_df.index.repeat( news_df['assetCodes'].apply(len) )
assert len(assetCodes_index) == len(assetCodes_expanded)
assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})
assetCodes = assetCodes[assetCodes["assetCode"].isin(assetCodeArray)]
news_df_expanded = pd.merge(assetCodes, news_df[news_cols], left_on='level_0', right_index=True, suffixes=(['','_old']))


# 'AAPL.O', 'BA.N', 'BAC.N', 'BBL.N', 'BCS.N', 'BP.N', 'DB.N', 'F.N',\
#        'GE.N', 'GS.N', 'HBC.N', 'JPM.N', 'MS.N', 'MSFT.O', 'RDSa.N',
#        'RDSb.N', 'RTP.N', 'XOM.N', 'RIO.N', 'C.N', 'HSBC.N'

# In[ ]:


#Shape of news_df after expanding
print(news_df_expanded.shape)


# In[ ]:


news_df_expanded.iloc[:5, :10]


# In[ ]:


#Checking to see if there are missing values in news
news_df_expanded.isna().sum()


# #### We found out that there are no missing values(NAs) in news dataset

# #### Function to Merge Market vs. News datasets

# In[ ]:


#"data_prep" will do Merge and some basic cleaning

def data_prep(market_df,news_df):
    asset_code_dict = {k: v for v, k in enumerate(market_df['assetCode'].unique())}
    columns_tobe_retained = ['time','assetCode', 'assetName' ,'volume', 'open', 'close','returnsClosePrevRaw1',                             'returnsOpenPrevRaw1','returnsClosePrevMktres1','returnsOpenPrevMktres1',                             'returnsClosePrevRaw10','returnsOpenPrevRaw10','returnsClosePrevMktres10',                             'returnsOpenPrevMktres10','returnsOpenNextMktres10',                             'assetCodeT','urgency', 'takeSequence', 'companyCount','marketCommentary','sentenceCount',           'firstMentionSentence','relevance','sentimentClass','sentimentWordCount','noveltyCount24H',           'firstCreated',                         # 'asset_sentiment_count', 'asset_sentence_mean','len_audiences',\
           'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D','volumeCounts24H','volumeCounts3D','volumeCounts5D','volumeCounts7D']
    market_df['date'] = market_df['time'].dt.date
    market_df['close_to_open'] = market_df['close'] / market_df['open']
    market_df['assetCodeT'] = market_df['assetCode'].map(asset_code_dict)
    #News data feature creation
    #news_df['time'] = news_df.time.dt.hour
    #news_df['sourceTimestamp']= news_df.sourceTimestamp.dt.hour
    #news_df['firstCreated'] = news_df['firstCreated'].dt.date 
    #news_df['asset_sentiment_count'] = news_df.groupby(['assetName', 'sentimentClass'])['time'].transform('count')
    #news_df['asset_sentence_mean'] = news_df.groupby(['assetName', 'sentenceCount'])['time'].transform('mean')
    #news_df['len_audiences'] = news_df['audiences'].map(lambda x: len(eval(x)))
    #kcol = ['firstCreated', 'assetCode']
    news_df = news_df.groupby(kcol, as_index=False).mean()

    # Merge news and market data. Only keep numeric columns
    market_df_merge = pd.merge(market_df, news_df, how='left', left_on=['date', 'assetCode'], 
                            right_on=['firstCreated', 'assetCode'])

    #return only data for the numeric columns + key information (assetCode, time)
    return market_df_merge[columns_tobe_retained]


# In[ ]:


#Group News Data by firstCreated & assetCode
kcol = ['firstCreated', 'assetCode']
d = news_df_expanded.sort_values('firstCreated').copy(deep=True)
d['firstCreated'] = d['firstCreated'].dt.date
d = d.groupby(kcol, as_index=False).mean()


# In[ ]:


d.tail(10)


# In[ ]:


dfmI = dfm.copy(deep=True)
dfnI = d.copy(deep=True)


# ## Merge Market & News data

# In[ ]:


len(dfnI)


# In[ ]:


#"data_prep" will do Merge of Market & News Datasets
merged_dataset = data_prep(dfmI,dfnI)


# In[ ]:


#Look at missing value summary
merged_dataset.count()


# In[ ]:


#Checkingfor NAs
merged_dataset.isna().sum()


# In[ ]:


# Function to plot time series data
def plot_vs_time(data_frame, column, calculation='mean', span=10):
    if calculation == 'mean':
        group_temp = data_frame.groupby('firstCreated')[column].mean().reset_index()
    if calculation == 'count':
        group_temp = data_frame.groupby('firstCreated')[column].count().reset_index()
    if calculation == 'nunique':
        group_temp = data_frame.groupby('firstCreated')[column].nunique().reset_index()
    group_temp = group_temp.ewm(span=span).mean()
    fig = plt.figure(figsize=(10,3))
    plt.plot(group_temp['firstCreated'], group_temp[column])
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.title('%s versus time' %column)


# ### Look at NA values of returnsOpenPrevMktres10. We will re-verify this graph after interpolating to miss NA values in returns variables.

# In[ ]:


merged_dataset[merged_dataset['returnsOpenPrevMktres10'].isna()]['assetCode']


# In[ ]:


d = merged_dataset[merged_dataset['assetCode'] == 'SAN.N']
import matplotlib.pyplot as plt
#print(d[d['returnsOpenPrevMktres10'].isna()]['assetCode'])
print(d.head())
plt.plot(d['time'], d['returnsOpenPrevMktres10'])


# ### Impute Market Returns data using  (NOCB) imputation: https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4

# ### Fill nan values in MktRes columns using NOCB interpolation.

# In[ ]:


# Fill nan values in MktRes columns using NOCB interpolation.
market_fill = merged_dataset.copy(deep=True)
column_market = ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
column_raw = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10']
for i in range(len(column_market)):
    market_fill[column_market[i]].interpolate(method='nearest', inplace=True)   


# ### Checking to make sure NAs are filled

# In[ ]:


d = market_fill[market_fill['assetCode'] == 'SAN.N']
import matplotlib.pyplot as plt
#print(d[d['returnsOpenPrevMktres10'].isna()]['assetCode'])
print(d.head())
plt.plot(d['time'], d['returnsOpenPrevMktres10'])


# In[ ]:


#Checkingfor NAs before extracting data only for the 5 companies
market_fill.isna().sum()


# In[ ]:


#fill nulls with NAs
import numpy as np
market_fill = market_fill.fillna(method='bfill')


# In[ ]:


market_fill.isna().sum()


# In[ ]:


market_fill.isnull().sum()


# In[ ]:


#Lets fill NAs with Linear Interpolation
# Fill nan values in News Variables
 
column_market = ["urgency", "takeSequence","companyCount","marketCommentary",
"sentenceCount","firstMentionSentence","relevance",
"sentimentClass","sentimentWordCount","noveltyCount24H",
"firstCreated","noveltyCount3D","noveltyCount5D",
"noveltyCount7D","volumeCounts24H","volumeCounts3D"
,"volumeCounts5D","volumeCounts7D"]
#column_raw = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10']
for i in range(len(column_market)):
    market_fill[column_market[i]].interpolate(method='nearest', inplace=True) 


# # 1. Feature Engineering Contd.

# #### Bin numerical to binary when there is not much data for factors.
# 

# ### Create new Features to account for Time Series auto Correlation between rows.

# In[ ]:


def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi


# In[ ]:


#'AAPL.O',  'CSCO.O', 'IBM.N', 'INTC.O', 'MSFT.O', 'ORCL.O', 'ORCL.N'
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
full_dataset = pd.DataFrame()


for assetCode in assetCodeArray:
   df = pd.DataFrame()
   # Gather asset specific data
   df = market_fill[market_fill["assetCode"] == assetCode]
   df['rsi10D'] = rsiFunc(df['close'].values, 10)
   #Calculating all the trend variables for this assetCode
   df['volume10DMA'] = df["volume"].rolling(window=10,min_periods=1).mean() 
   df['returnsClosePrevMktres1-10DMA'] = df["returnsClosePrevMktres1"].rolling(window=10,min_periods=1).mean()
   df['returnsClosePrevMktres1-yearly'] = df["returnsClosePrevMktres1"].rolling(window=365,min_periods=1).mean()
   df['returnsClosePrevMktres1-quarterly'] = df["returnsClosePrevMktres1"].rolling(window=90,min_periods=1).mean()

   df['returnsOpenPrevMktres1-10DMA'] = df["returnsOpenPrevMktres1"].rolling(window=10,min_periods=1).mean()
   df['returnsOpenPrevMktres1-yearly'] = df["returnsOpenPrevMktres1"].rolling(window=365,min_periods=1).mean()
   df['returnsOpenPrevMktres1-quarterly'] = df["returnsOpenPrevMktres1"].rolling(window=90,min_periods=1).mean()
   
   #Create new feature for close price moving average.
   df['close10DMA'] = df['close'].rolling(window=10,min_periods=1).mean()
   df['sentenceCount7D'] = df['sentenceCount'].rolling(window=7,min_periods=1).sum()
   df['firstMentionSentence7D'] = df['firstMentionSentence'].rolling(window=7,min_periods=1).sum()
   df['relevance7D'] = df['relevance'].rolling(window=7,min_periods=1).sum()
   df['sentimentWordCount7D'] = df['sentimentWordCount'].rolling(window=7,min_periods=1).sum()
   df['sentimentClass7D'] = df['sentimentClass'].rolling(window=7,min_periods=1).sum()
   df['urgency7D'] = df['urgency'].rolling(window=7,min_periods=1).sum()
   df['takeSequence7D'] = df['takeSequence'].rolling(window=7,min_periods=1).sum()
   df['companyCount7D'] = df['companyCount'].rolling(window=7,min_periods=1).sum()
   df['marketCommentary7D'] = df['marketCommentary'].rolling(window=7,min_periods=1).sum()
   #df['bodySize7D'] = df['bodySize'].rolling(window=7).sum()

   #Exponential Moving Average
   ewma = pd.Series.ewm
   df['close_10EMA'] = ewma(df["close"], span=10).mean()
   #Bollinger Bands are a type of statistical chart characterizing the prices and 
   #volatility over time of a financial instrument or commodity, using a formulaic method 
   #propounded by John Bollinger in the 1980s. Financial traders employ these charts as 
   #a methodical tool to inform trading decisions, control automated trading systems, 
   #or as a component of technical analysis. Bollinger Bands display a graphical band 
   #(the envelope maximum and minimum of moving averages, similar to
   #Keltner or Donchian channels) and volatility (expressed by the width of the envelope) 
   #in one two-dimensional chart.

   #ref. https://en.wikipedia.org/wiki/Bollinger_Bands 
   #Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the 
   #relationship between two moving averages of prices.
   #The MACD is calculated by subtracting the 26-day exponential moving average (EMA) from the 12-day EMA
   df['close_26EMA'] = ewma(df["close"], span=26).mean()
   df['close_12EMA'] = ewma(df["close"], span=12).mean()
   df['MACD'] = df['close_12EMA'] - df['close_26EMA']
   no_of_std = 2
   #ref. https://www.investopedia.com/terms/m/macd.asp

   df['MA_10MA'] = df['close'].rolling(window=10,min_periods=1).mean()
   df['MA_10MA_std'] = df['close'].rolling(window=10,min_periods=1).std() 
   df['MA_10MA_BB_high'] = df['MA_10MA'] + no_of_std * df['MA_10MA_std']
   df['MA_10MA_BB_low'] = df['MA_10MA'] - no_of_std * df['MA_10MA_std']
   full_dataset = full_dataset.append(df)



full_dataset["firstCreated"] = full_dataset["time"].dt.date
full_dataset['Year'] = full_dataset.time.dt.year
full_dataset['Month'] = full_dataset.time.dt.month
full_dataset['Day'] = full_dataset.time.dt.day
full_dataset['Week'] = full_dataset.time.dt.week

import datetime
full_dataset['day_of_year']  = full_dataset["time"].dt.dayofyear
full_dataset = full_dataset.sort_values('firstCreated')


# #### Spot outlier Companies for close/open price difference

# In[ ]:


full_dataset["firstCreated"] = full_dataset["time"].dt.date
full_dataset['Year'] = full_dataset.time.dt.year
full_dataset['Month'] = full_dataset.time.dt.month
full_dataset['Day'] = full_dataset.time.dt.day
full_dataset['Week'] = full_dataset.time.dt.week
import datetime
full_dataset['day_of_year']  = full_dataset["time"].dt.dayofyear
full_dataset['quarter']  = full_dataset["time"].dt.quarter
full_dataset = full_dataset.sort_values('firstCreated')

full_dataset["daily_diff"] = full_dataset["close"] - full_dataset["open"]
full_dataset['close_to_open'] =  np.abs(full_dataset['close'] / full_dataset['open'])
    
 # determine whether the day is set on a holiday
from pandas.tseries.holiday import USFederalHolidayCalendar as cal
holidays = cal().holidays(start='2007-01-01', end='2018-09-27').to_pydatetime()
full_dataset['on_holiday'] = full_dataset["firstCreated"].str.slice(0,10).apply(lambda x: 1 if x in holidays else 0)


#  plot_vs_time(full_dataset, 'returnsClosePrevMktres1', calculation='mean', span=30)
#  plot_vs_time(full_dataset, 'returnsClosePrevMktres1', calculation='mean', span=90)

# In[ ]:


full_dataset.tail()


# In[ ]:


pos = len(full_dataset[full_dataset['returnsOpenNextMktres10']>0])
neg = len(full_dataset[full_dataset['returnsOpenNextMktres10']<0])
tot = pos + neg

print ('Positive cases %: ', pos  * 100/tot)
print ('Negative cases %: ', neg  * 100/tot)


# In[ ]:


market_train_df = full_dataset.copy(deep=True)
market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']
grouped = market_train_df.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()
grouped.sort_values(('price_diff', 'std'), ascending=False)[:10].head()


# In[ ]:


g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:10]
g['min_text'] = 'Maximum price drop: ' + (-1 * g['price_diff']['min']).astype(str)
trace = go.Scatter(
    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = g['price_diff']['std'].values,
    mode='markers',
    marker=dict(
        size = g['price_diff']['std'].values,
        color = g['price_diff']['std'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = g['min_text'].values
    #text = f"Maximum price drop: {g['price_diff']['min'].values}"
    #g['time'].dt.strftime(date_format='%Y-%m-%d').values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Top 10 months by standard deviation of price change within a day',
    hovermode= 'closest',
    yaxis=dict(
        title= 'price_diff',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# ### No surprising outliers to remove in terms of prices

# #### Plot Correlations of Market Data

# In[ ]:


import seaborn as sns
market_train_df["target_stockMoveUp"] = market_train_df.returnsOpenNextMktres10 > 0
columns_corr_market =  [ 'volume', 'open', 'close',       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',       'close_10EMA', 'close_26EMA', 'close_12EMA', 'MACD',       'MA_10MA', 'MA_10MA_std', 'MA_10MA_BB_high', 'MA_10MA_BB_low','target_stockMoveUp']
colormap = plt.cm.RdBu
plt.figure(figsize=(18,15))
sns.heatmap(market_train_df[columns_corr_market].astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pair-wise correlation')
plt.rcParams['font.size'] = 10


# **Conclusions:**
# 
# 1. Stock volumes have some positive impact on the Stock movement.
# 2. All of the returns variable have positive correlation with each other.
# 3. Close & Open prices have strong correlation.

# In[ ]:


columns_corr_merge = [    
       'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts24H',
       'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D', 
       'volume10DMA', 'close10DMA', 'sentenceCount7D',
       'firstMentionSentence7D', 'relevance7D', 'sentimentWordCount7D',
       'sentimentClass7D' , 'urgency7D', 'takeSequence7D', 'companyCount7D',
       'marketCommentary7D','target_stockMoveUp']
colormap = plt.cm.RdBu
# Scaling 
df = market_train_df[columns_corr_merge]
mins = np.min(df, axis=0)
maxs = np.max(df, axis=0)
rng = maxs - mins
df = 1 - ((maxs - df) / rng)

plt.figure(figsize=(18,15))
sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pair-wise correlation market and news')
plt.rcParams['font.size'] = 10


# **Conclusions:**
# 
# 1. Stock volumes have positive correlation with Stock Movement variables and the news Novelty/Volume.
# 2. Novelty of the content seems to have correlation with Stock Closing Price.
# 3. Novelty Indicators and Volume counts are postively correlated with each other.
# 

# # Functions

# # 3. Split Train and Test

# In[ ]:


df1 = full_dataset.copy(deep=True)
df1 = df1.dropna()
#create y from stock returns for next 10 days variable.
y = df1.returnsOpenNextMktres10 > 0
# Rest of the dataset is X

#cols = [ 'volume','returnsClosePrevMktres1-20DMA',\
#       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',\
#       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',\
#       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',\
#       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',\
#       'assetCodeT',\
#       'volumeCounts7D', 'rsi20D',\
#       'volume10DMA', 'close10DMA', 'sentenceCount7D',\
#       'firstMentionSentence7D', 'relevance7D', 'sentimentWordCount7D',\
#       'sentimentClass7D', 'urgency7D', 'takeSequence7D', 'companyCount7D',\
#       'close_10EMA', 'close_26EMA', 'close_12EMA',\
#       'MACD', 'MA_10MA', 'MA_10MA_std', 'MA_10MA_BB_high', 'MA_10MA_BB_low',\
#       'Year', 'Month', 'Day', 'Week', 'day_of_year' ]


#df1[cols].head(200)


# In[ ]:


cols = [
       'volume10DMA'
    ,'day_of_year'\
    ,'MA_10MA_BB_high'\
    ,'close_10EMA'\
    ,'returnsClosePrevMktres10'\
    ,'MACD'\
    ,'companyCount7D'\
    ,'sentimentClass7D'\
    ,'rsi10D'\
    #,'assetCodeT'\
    #,'close_to_open'\
    ,'volumeCounts7D'\
    #'returnsClosePrevMktres1-20DMA'\
    ,'returnsClosePrevMktres1-yearly'\
    ,'returnsClosePrevMktres1-quarterly'\
    #'returnsOpenPrevMktres1-20DMA'\
    ,'returnsOpenPrevMktres1-yearly'\
    ,'returnsOpenPrevMktres1-quarterly'\
   
    
]
len(cols)


# In[ ]:




X = df1[cols] 
#train_size = int(len(X) * 0.66)
#X_train, X_test = X[0:train_size], X[train_size:len(X)]
#y_train, y_test = y[0:train_size], y[train_size:len(X)]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=1)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.34, random_state=1)

print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(X_train)))
#print('Validation Observations: %d' % (len(X_val)))
print('Testing Observations: %d' % (len(X_test)))

# The target is binary
import xgboost as xgb


# # Comparing AUC Scores from Classifiers to pick Top 3

# In[ ]:


# Compare Algorithms
import pandas
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('BAGC', BaggingClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('RFC', RandomForestClassifier()))
models.append(('ABC', AdaBoostClassifier()))
models.append(('XGBC', xgb.XGBClassifier()))
models.append(('GBC', GradientBoostingClassifier()))
# evaluate each model in turn
results = []
names = []
scoring = 'roc_auc'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
import matplotlib as mpl
from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 15), dpi=80, facecolor='b', edgecolor='k')
#figure(figsize=(1,1))
fig = plt.figure()
mpl.style.use('bmh')
fig.set_size_inches(15, 15, forward=True)
fig.suptitle('ML Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# # Picked Top 3 Classifiers - XGBoost, Bagging & RF Classifiers

# ## XGBoost - Classifier 1

# In[ ]:


import xgboost as xgb
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
 
xgb_model_train = xgb.XGBClassifier()
 
import matplotlib.pylab as plt
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.metrics  import accuracy_score, roc_auc_score


# ## Random Forest - Classifier 2

# In[ ]:


rf_obj=RandomForestClassifier()
#rf_obj.fit(X_train,y_train)


# ## BaggingClassifier - Classifier 3

# In[ ]:


bgc_obj=BaggingClassifier(base_estimator=None)
#bgc_obj.fit(X_train,y_train)


# # Plotting Underfit vs Overfit. based on Max-Depth

# In[ ]:


max_depths = [  3,  4,  5,  6, 7,  8,  9, 10, 11, 12,13,14,15,16,18]

print (max_depths)
train_results = []
test_results = []
from sklearn.model_selection import cross_val_score
for max_depth in max_depths:
   dt = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=1, learning_rate=0.1, max_delta_step=0,
       max_depth=max_depth, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1 )  
   scores = cross_val_score(dt, X_train, y_train, cv=5,scoring="accuracy")
   dt.fit(X_train,y_train)
   #print("estimated AUC on the XG Boost training data-set: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))   
   # Add auc score to previous train results
   train_results.append(scores.mean())
   #y_pred = dt.predict(X_test)
   #false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)
   #roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
   testAccuracy = round(dt.score(X_test, y_test), 4)
   # Add auc score to previous test results
   test_results.append(testAccuracy)


# In[ ]:



from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train')
line2, = plt.plot(max_depths, test_results, 'r', label='Test')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Scores for XGBoost')
plt.xlabel('Tree depth')
plt.show()
train_results = []
test_results = []
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
for max_depth in max_depths:
   dt = RandomForestClassifier(max_depth=max_depth)
   
   scores = cross_val_score(dt, X_train, y_train, cv=5,scoring="accuracy")
   #print("estimated AUC on the RF training data-set: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
   train_results.append(scores.mean())
   dt.fit(X_train,y_train)
   #y_pred = dt.predict(X_test)
   #false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)
   #roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous test results
   testAccuracy = round(dt.score(X_test, y_test), 4)
   # Add auc score to previous test results
   test_results.append(testAccuracy)


# In[ ]:


from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train')
line2, = plt.plot(max_depths, test_results, 'r', label='Test')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Score for RFClassifier')
plt.xlabel('Tree depth')
plt.show()
train_results = []
test_results = []
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
for max_depth in max_depths:
   dt = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth = max_depth))
   scores = cross_val_score(dt, X_train, y_train, cv=5,scoring="roc_auc")
   #print("estimated AUC on the RF training data-set: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
   train_results.append(scores.mean())
   dt.fit(X_train,y_train)
  #y_pred = dt.predict(X_test)
   #false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_pred)
   #roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
   # Add auc score to previous test results
   testAccuracy = round(dt.score(X_test, y_test), 4)
   # Add auc score to previous test results
   test_results.append(testAccuracy)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train')
line2, = plt.plot(max_depths, test_results, 'r', label='Test')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Score for Bagging Classifier')
plt.xlabel('Tree depth')
plt.show()


# # 5. Using Max-Depth findings to Fit Classifier(s) with GridSearchCV

# In[ ]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import numpy as np 


# ## XGBoost - Classifier 1 - GridSearch

# In[ ]:


#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have 
#much fun of fighting against overfit 
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
#from sklearn.model_selection import TimeSeriesSplit
#tss = TimeSeriesSplit(n_splits=10).split(X_train)

params = {
         
        'gamma': [0.5, 1, 1.5, 2, 5],
        'max_depth': [6,7,8,9,10],
        'n_estimators': [50,100  ]  
        }
 
#gsearch = GridSearchCV(xgb_model_train, params, n_jobs=5,cv=5, scoring='roc_auc',verbose=1, refit=True)
gsearch = RandomizedSearchCV(xgb_model_train, params, cv = 5, scoring = 'roc_auc', verbose=1, 
                              random_state=42, n_jobs = -1)


# ## Random Forest - Classifier 2 - GridSearch

# In[ ]:


params = { 
'max_depth': [14,15,16,17],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [50,100,200,300,400] 
}
#rf_Grid = GridSearchCV(rf_obj, param_grid, cv = 5, scoring = 'roc_auc',refit = True, n_jobs=-1, verbose = 1)
rf_Grid = RandomizedSearchCV(rf_obj, params,cv = 5, scoring = 'roc_auc', verbose=1, random_state=42, n_jobs = -1)


# ## BaggingClassifier - Classifier 3  - GridSearch

# In[148]:


params = {
  'max_features': [5,6, 7,8],
  'max_samples' : [0.05, 0.1, 0.2, 0.5],
  'n_estimators': [50,100, 200, 400] ,
  'bootstrap' :[True,False],
  'bootstrap_features':[True,False]
}
 
bgc_Grid = RandomizedSearchCV(BaggingClassifier(DecisionTreeClassifier(), max_features = 0.5), params , cv = 5, scoring = 'roc_auc', verbose=1, random_state=42, n_jobs = -1)


# # # 7. Use GridSearchCV to tune hyper parameters.**

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


gsearch.fit(X_train, y_train)


# In[ ]:


rf_Grid.fit(X_train, y_train)


# In[ ]:


bgc_Grid.fit(X_train, y_train)


# ## Plotting top features from XGBoost fit

# In[ ]:


plot_importance(gsearch.best_estimator_) 
plt.rcParams['figure.figsize'] = [12,12]
plt.rcParams['font.size'] = 25
plt.show()


# In[ ]:


gsearch.best_estimator_


# In[ ]:


xg_clf = gsearch.best_estimator_


# In[ ]:


rf_clf = rf_Grid.best_estimator_
rf_clf 


# ## Plotting top features from RF fit

# In[ ]:


feature_importances = pd.Series(rf_clf .feature_importances_, index=X_train.columns)
feature_importances.nlargest(10).sort_values(ascending = True).plot(kind='barh')


# In[ ]:


bgc_clf = bgc_Grid.best_estimator_


# # 8. Validation set AUC

# In[ ]:


#from sklearn.metrics import accuracy_score
best_parameters, score, _ = max(gsearch.grid_scores_, key=lambda x: x[1])
print('XGBoost Cross validation AUC score:', np.round(gsearch.best_score_ ,2))
for param_name in sorted(best_parameters.keys()):
   print("%s: %r" % (param_name, best_parameters[param_name]))


# In[ ]:


#from sklearn.metrics import accuracy_score

#best_parameters, score, _ = max(rf_Grid.grid_scores_, key=lambda x: x[1])
print('RandomForest Cross validation AUC score:', np.round(rf_Grid.best_score_ ,2))
#for param_name in sorted(best_parameters.keys()):
#    print("%s: %r" % (param_name, best_parameters[param_name]))


# In[ ]:


#from sklearn.metrics import accuracy_score

#best_parameters, score, _ = max(bgc_Grid.grid_scores_, key=lambda x: x[1])
print('Bagging Classifier Cross validation AUC score:', np.round(bgc_Grid.best_score_ ,2))
#for param_name in sorted(best_parameters.keys()):
#    print("%s: %r" % (param_name, best_parameters[param_name]))


# 

# In[ ]:


print('XgBoost Training AUC score:', np.round(roc_auc_score(clf.predict(X_train),y_train),2))
plot_importance(clf, max_num_features=20) # top 20 most important features
plt.show()

print('RandomForest Training AUC score:', np.round(roc_auc_score(rf_clf.predict(X_train),y_train),2))


# In[ ]:


print('BaggingClassifier Training AUC score:', np.round(roc_auc_score(rf_clf.predict(X_train),y_train),2))


# # Train Predictions

# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_clf, X_train, y_train, cv=5,scoring="roc_auc")
print("estimated AUC on the training data-set: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(xg_clf, X_train, y_train, cv=5,scoring="roc_auc")
print("estimated AUC on the training data-set: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


# # Test set AUC for XGBoost

# In[ ]:


# calculate the fpr and tpr for all thresholds of the classification
probs =  gsearch.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
print ("Test AUC for XGBoost", roc_auc )
# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic XGBoost Test')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


y_scores = gsearch.predict(X_test)
confusion_matrix =  pd.crosstab(index=y_test, columns=y_scores, rownames=['Expected'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True, square=False, fmt='', cbar=False)
plt.title("Classification Matrix", fontsize = 15)
plt.show()
#plot_confusion_matrix_from_data(y_test,  y_scores,columns=["Up","Down" ], annot=True, cmap="Blues",
     # fmt='.5f', fz=20, lw=1, cbar=False, figsize=[12,12], show_null_values=0, pred_val_axis='lin')


# In[ ]:


clfReport = metrics.classification_report(y_test, y_scores, target_names=["Stock-Movement-Up", "Stock-Movement-Down"])
print(clfReport)

