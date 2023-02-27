import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
import datetime
# import pickle
import streamlit as st
import model_building as m
import technical_analysis as t
import correlation_analysis as c
import mpld3
import streamlit.components.v1 as components
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


with st.sidebar:
    st.markdown("# Stock Analysis & Forecasting")
    user_input = st.selectbox(
    'Please select the stock for forecasting and technical analysis ',
    ('ADANIENT.NS','TATASTEEL.NS','PAGEIND.NS','EICHERMOT.NS','INFY.NS'))
    # user_input = st.text_input('Enter Stock Name', "ADANIENT.NS")
    st.markdown("### Choose Date for your anaylsis")
    date_from = st.date_input("From",datetime.date(2020, 1, 1))
    date_to = st.date_input("To",datetime.date(2023, 2, 25))
    options = st.multiselect(
        'Select stocks for diversification analysis',
        ['ADANIENT.NS','TATASTEEL.NS','PAGEIND.NS','EICHERMOT.NS','INFY.NS'],
        ['ADANIENT.NS']
    )
    # st.write('You selected:', options[0])
    btn = st.button('Submit') 

#adding a button
if btn:
    df = yf.download(user_input, start=date_from, end=date_to)
    plotdf, future_predicted_values =m.create_model(df)


    st.markdown("### Original vs predicted close price")
    fig= plt.figure(figsize=(20,10))
    sns.lineplot(data=plotdf)
    st.pyplot(fig)

    st.markdown("### Next 10 days forecast")
    list_of_days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7","Day 8", "Day 9", "Day 10"]

    for i,j in zip(st.tabs(list_of_days),range(10)):
        with i:
            st.write(future_predicted_values.iloc[j:j+1])


    st.markdown("### Adj Close Price")
    fig= plt.figure(figsize=(20,10))
    t.last_2_years_price_plot(df)
    st.pyplot(fig)

    st.markdown("### Daily Percentage Changes")
    fig= plt.figure(figsize=(20,10))
    t.daily_percent_change_plot(df)
    st.pyplot(fig)
    
    st.markdown("### Daily Percentage Changes Histogram")
    fig= plt.figure(figsize=(20,10))
    t.daily_percent_change_histogram(df)
    st.pyplot(fig)

    st.markdown("### Trend Analysis")
    fig= plt.figure(figsize=(20,10))
    t.trend_pie_chart(df)
    st.pyplot(fig)

    st.markdown("### Volume Plot")
    fig= plt.figure(figsize=(20,10))
    t.volume_plot(df)
    st.pyplot(fig)

    st.markdown("### Volatility Plot")
    fig= plt.figure(figsize=(20,10))
    t.volatility_plot(df)
    st.pyplot(fig)


    st.markdown("## Technical Analysis")

    st.markdown("### MACD Indicator")
    
    fig= plt.figure(figsize=(20,10))
    t.plot_price_and_signals(t.get_macd(df),'MACD')
    st.pyplot(fig)

    fig= plt.figure(figsize=(20,10))
    t.plot_macd(df)
    st.pyplot(fig)

    st.write(" ***:blue[Strategy:]:***")
    st.write(":red[Sell  Signal:] The cross over: When the MACD line is below the signal line.")
    st.write(":green[Buy Signal:] The cross over: When the MACD line is above the signal line.")

    st.markdown("### RSI Indicator")

    fig= plt.figure(figsize=(20,10))
    t.plot_price_and_signals(t.get_rsi(df),'RSI')
    st.pyplot(fig)

    fig= plt.figure(figsize=(20,10))
    t.plot_rsi(df)
    st.pyplot(fig)

    st.write(" ***:blue[Strategy:]:***")
    st.write(":red[Sell  Signal:] When RSI increases above 70%")
    st.write(":green[Buy Signal:] When RSI decreases below 30%.")


    st.markdown("### Bollinger Indicator")

    fig= plt.figure(figsize=(20,10))
    t.plot_price_and_signals(t.get_bollinger_bands(df),'Bollinger_Bands')
    st.pyplot(fig)

    fig= plt.figure(figsize=(20,10))
    t.plot_bollinger_bands(df)
    st.pyplot(fig)

    st.write(" ***:blue[Strategy:]:***")
    st.write(":red[Sell  Signal:] As soon as the market price touches the upper Bollinger band")
    st.write(":green[Buy Signal:] As soon as the market price touches the lower Bollinger band")

    st.markdown("### SMA Indicator")
   
    fig= plt.figure(figsize=(20,10))
    t.sma_plot(df)
    st.pyplot(fig)
    st.write(" ***:blue[Strategy:]:***")
    st.write(":red[Sell  Signal:] When the 50-day SMA crosses below the 200-day SMA.")
    st.write(":green[Buy Signal:] When the 50-day SMA crosses above the 200-day SMA.")

    st.markdown("### EMA Indicator")
   
    fig= plt.figure(figsize=(20,10))
    t.ema_plot(df)
    st.pyplot(fig)
    st.write(" ***:blue[Strategy:]:***")
    st.write(":red[Sell  Signal:] When the 50-day EMA crosses below the 200-day EMA.")
    st.write(":green[Buy Signal:] When the 50-day EMA crosses above the 200-day EMA.")
   
    st.markdown("### Diversified Portfolio Analysis")
    combined_df = yf.download(options, start=date_from, end=date_to)['Adj Close']
    combined_df = combined_df.round(2)
    
    fig= plt.figure(figsize=(20,10))
    c.corr_plot(combined_df)
    st.pyplot(fig)

    st.write(" ***:blue[Strategy:]:*** All the stocks which do not show significant correlation can be included in a portfolio.")
    
    
else:
    st.write('Please click on the submit to get the analysis') #displayed when the button is unclicked

    
        
