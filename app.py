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
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components



with st.sidebar:
    st.markdown("# Stock Analysis & Forecasting")
    user_input = st.text_input('Enter Stock Name', "ADANIENT.NS")
    st.markdown("### Choose Date for your anaylsis")
    date_from = st.date_input("From",datetime.date(2020, 1, 1))
    date_to = st.date_input("To",datetime.date(2023, 2, 25))
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

else:
    st.write('Please click on the submit to get the analysis') #displayed when the button is unclicked

    
        
