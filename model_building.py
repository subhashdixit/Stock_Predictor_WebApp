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
import streamlit as st
# import test as t

# convert an array of values into a dataset matrix
"""
This function creates a sliding window of size time_step over the input dataset and constructs a set of input-output pairs for training a time series forecasting model. 
The input matrix dataX contains n rows, where n is the number of time steps in the input sequence, and time_step columns, representing the past time_step values of the 
input sequence. The output matrix dataY contains n rows and 1 column,representing the next value in the sequence to be predicted.
"""
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def create_model(df):
    pass
    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(df['Close']).reshape(-1,1))

    training_size=int(len(closedf)*0.75)
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 15
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    # LSTM requies 3-dimensional input
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    # Model building
    tf.keras.backend.clear_session()
    model=Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=(time_step,1)))
    model.add(LSTM(32,return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss='mean_squared_error',optimizer=optimizer)
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=5,verbose=1)


    # model.save('lstm_model.h5')
    # model = tf.keras.models.load_model('lstm_model.h5')


    ### Lets Do the prediction and check performance metrics
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

    # shift train predictions for plotting
    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

    plotdf = pd.DataFrame({'Date': df.index,'original_close': df['Close'],'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                        'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

    # st.markdown("### Original vs predicted close price")
    # fig= plt.figure(figsize=(20,8))
    # sns.lineplot(data=plotdf)
    # st.pyplot(fig)

    # Predicton 10 days forecast
    # Creating array of last 15 days
    x_input=test_data[len(test_data)-time_step:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    from numpy import array
    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = 10
    while(i<pred_days):
        if(len(temp_input)>time_step):
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            yhat = model.predict(np.expand_dims(x_input, 2))
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0])
            temp_input=temp_input[1:]
        
            lst_output.extend(yhat.tolist())
            i=i+1
            
        else:
            yhat = model.predict(np.expand_dims(x_input, 2))
            temp_input.extend(yhat[0])
            lst_output.extend(yhat.tolist())
            
            i=i+1
            
    # print("Output of predicted next days: ", len(lst_output))
    # print("Output of predicted next days: ",lst_output)
    next_predicted_days_value = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
    # st.write(next_predicted_days_value)
    return plotdf,pd.DataFrame(next_predicted_days_value)
