import numpy as np
import pandas as pd
import yfinance as yahooFinance
import matplotlib.pyplot as plt
from keras.models import load_model
start = "2020-01-01"
end = "2020-12-31"
df = yf.download('AAPL' , start=start, end=end)
#print(data)
plt.figure(figsize=(12, 12))
plt.plot(df['Close'], label=f'{ticker} Stock Price', color='blue')
plt.legend()
plt.grid(True)
plt.show()
df.head()

df = df.reset_index()
df.head()
plt.plot(df.close)
plt.figure(figsize=(12,6))
plt.plot(df.close)
plt.plot(ma100, 'r')
df.shape
'''
#spliting data into training and testing 
data_training = pd.DataFrame(fd['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][:int(int(len(df*0.70) int(len(df))))])
print(data_training.shape)
print(data_testing.shape)
data_source.head()
data_testing.head()
from sklearn.preprocessinng import MinMaxScaler
Scaler = MinMaxScaler(feature_range=(0,1))
data_training_array =Scaler.fit_traNSFORM(data_training)

x_train =[]
y_train = []

for i in range(100,data_training_array.shape[0]):
x_train.append(data_training_array[i-100:i])
y_train.append(data_training_array[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)
form keras.layers import Dense,Dropout,LSTM
form keras.models import Sequential
model = Sequential()
model.add(LSTM(units=50,activation ='relu',return_sequences = True,input_shape = (x_train.shape[1],1)))
modell.add(Dropout(0.2))

model.add(LSTM(units=80,activation ='relu',return_sequences = True,input_shape = (x_train.shape[1],1)))
modell.add(Dropout(0.3))

model.add(LSTM(units=120,activation ='relu',return_sequences = True,input_shape = (x_train.shape[1],1)))
modell.add(Dropout(0.5))

model.add(Dense(units=1))
model.summary()
model.compile(optimizer ='adam',loss = 'mean_squared_error')
model.fix(x_train,y_train,epochs=50)
model.save('keras_model.h5')
data_testing.head()
data_training.tail()
past_100_days = data_training 
final_df = past_100_days.append(data_testing,ignore_index=True)
final_df.head()
input.data=Scaler.fit_transform(final_df)
input_data.shape
x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
    x_testappend(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)
print(x_test.shape)
print(y_test.shape)
#making prediction 
y_predicted = model.predict(x_test)
#y_test
#y_predicted    
scale_factor =1/0.02099517
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor
plt.figure(figsize=(12,6))
plt.plot(y_test,'b',lebel ='Original Price')
plt.plot(y_predicted,'r',lebel='Predicted Price')
plt.xlebel('Time')
plt.ylabel('price')
plt.legend()
plt.show()

'''
