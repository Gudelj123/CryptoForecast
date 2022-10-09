# -*- coding: utf-8 -*-
"""
@author: Ivan Gudelj

"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM



#Selecting and loading data
selectedOption = input("Select the number of Cryptocurrency you want to predict:\n\t1. Bitcoin\n\t2. Etherum\n\t3. Doge Coin\n")
if(selectedOption == "1"):
    selectedCrypto = "BTC-USD"
elif(selectedOption == "2"): 
    selectedCrypto = "ETH-USD"
elif(selectedOption == "3"): 
    selectedCrypto = "DOGE-USD"  
#---Debug --- print(selectedCrypto)

#Selecting time range for training data of neural network
startDate = input("Please enter wanted start Date in form (YYYY-MM-DD)\n")
yearStart, monthStart, dayStart = map(int, startDate.split('-'))
startDate = dt.date(yearStart,monthStart,dayStart)
endDate = input("Please enter wanted end Date in form (YYYY-MM-DD)\n")
yearEnd, monthEnd, dayEnd = map(int, endDate.split('-'))
endDate = dt.date(yearEnd,monthEnd,dayEnd)
#Loading the data from Yahoo Finance
dataToRead = web.DataReader(selectedCrypto,'yahoo',startDate,endDate)

#Preparing data for network
scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(dataToRead['Close'].values.reshape(-1,1))

#How many days back you want to consider for prediction
predictionDays = int(input("How many days would you like to consider for prediction:"))

x_train = []
y_train = []

#Preparing data
for x in range (predictionDays, len(scaledData)):
    x_train.append(scaledData[x-predictionDays:x,0])
    y_train.append(scaledData[x,0])
    
x_train,y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True,))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True,))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1)) #Actual prediction of next price

model.compile(optimizer='adam',loss='mean_squared_error', metrics= ['accuracy'])
hisotry = model.fit(x_train, y_train, epochs = 100, batch_size= 32)

#------------TESTING MODEL ACCURACY on Existing data-------------
#Test data from last day of training data until now
testStartDate = endDate
testEndDate = dt.datetime.now()
testData = web.DataReader(selectedCrypto,'yahoo',testStartDate,testEndDate)

actual_prices = testData['Close'].values
totalDataset = pd.concat((dataToRead['Close'],testData['Close']),axis=0)
model_inputs = totalDataset[len(totalDataset)-len(testData)-predictionDays:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#Prepare test data
x_test = []
for x in range (predictionDays,len(model_inputs)): 
    x_test.append(model_inputs[x-predictionDays:x, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape [1],1))


#Make prediction on Test Data
predictedPrices = model.predict(x_test)
predictedPrices = scaler.inverse_transform(predictedPrices)

plt.plot(actual_prices, color="blue",label = f"Actual {selectedCrypto} Price")
plt.plot(predictedPrices, color="green",label = f"Predicted {selectedCrypto} Price")
plt.title(f"{selectedCrypto} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{selectedCrypto} Share Price")
plt.legend()
plt.show()

#Predict for Tomorrow!!!
real_data = [model_inputs[len(model_inputs)+1-predictionDays:len(model_inputs+1),0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

predictionForTomorrow = model.predict(real_data)
predictionForTomorrow = scaler.inverse_transform(predictionForTomorrow)
print(f"Prediction: {predictionForTomorrow}")


print("")
print('-------------------- Model Summary --------------------')
model.summary() # print model summary
print("")
print('-------------------- Weights and Biases --------------------')
print("")
for layer in model.layers:
    print(layer.name)
    for item in layer.get_weights():
        print("  ", item)
print("")

# Print the last value in the evaluation metrics contained within history file
print('-------------------- Evaluation on Training Data --------------------')
for item in hisotry.history:
    print("Final", item, ":", hisotry.history[item][-1])
print("")

# Evaluate the model on the test data using "evaluate"
print('-------------------- Evaluation on Test Data --------------------')
results = model.evaluate(x_test)
print("")

