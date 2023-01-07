# !pip install --upgrade pandas-datareader
# !pip install --upgrade pandas
# !pip install yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
# import pandas_datareader as data
# from keras.model import load_model
from tensorflow.keras.models import load_model 
import pickle
import streamlit as st

st.title("Stock Predictor")
user_input = st.text_input('Enter Stock Name', 'AAPL')
df = yf.download(user_input, start="2010-01-01", end="2022-12-31")

# Describe data
st.subheader('Data from 2010 - 2022')
st.write(df.describe())

# Visualisation

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 days MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df['Close'])
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100 and 200 days MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df['Close'])
st.pyplot(fig)

#Splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#Scaling down the data for LSTM model
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)

# X_train = []
# y_train = []

# for i in range(100, data_training.shape[0]):
#     X_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i,0])

# X_train, y_train = np.array(X_train), np.array(y_train)

#Loaing the model

# model = pickle.load(open('keras_model.pkl', 'rb'))
# cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)

final_df = past_100_days.append(data_testing, ignore_index = True)

# Scaline down the test data
input_data = scaler.transform(final_df)

X_test = []
y_test = [] 

for i in range(100, input_data.shape[0]):
  X_test.append(data_training_array[i-100:i])
  y_test.append(data_training_array[i,0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Making predictions
y_pred = model.predict(X_test)

scale_factor = 1/scaler.scale_
y_pred = y_pred *scale_factor
y_test = y_test *scale_factor

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_pred, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)