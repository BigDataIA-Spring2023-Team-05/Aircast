import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
import aqi
import pickle
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData, DateTime

## Importing 1 year data
engine = create_engine('mysql+pymysql://admin:123456789@aircast.cjisewdv5jgk.us-east-1.rds.amazonaws.com:3306/aircast')
aqsid = '250250042'


####

#####################   DATA EXTRACTION FROM AWS DataBase #############################

####
# Read data from the SQL table into a dataframe
print('>>>>>>>>>>>>>>>>  data extraction started')
inp_data = pd.read_sql_query('SELECT * from aircast.StationsData WHERE aquid = "%s"' % aqsid, engine)
print('>>>>>>>>>>>>>>>>  data extraction done')

####

##################    FEATURE ENGINEERING  #############################

####
# Renaming columns :
inp_data = inp_data.rename(columns={'collection_timestamp':'datetime','ozone':'OZONE', 'so2':'SO2', 'no2':'NO2','co':'CO','pm2_5':'PM2.5','pm10':'PM10'})

df = inp_data.copy()
# Create a boolean array of the same shape as the dataframe
df = df.replace('NULL',0)
# Dropping a list of columns with all values 'NULL'
df = df.drop(columns=df.columns[df.isnull().all()].tolist())
## Storing aqs_id:
aqsid = df['aquid'].iloc[0]

# Dropping un-required columns
df = df.drop(columns = ['aquid','id'])

## setting 'datetime' column as index
check = df.copy()
check = check.set_index('datetime')

## Sorting values
df_sorted = check.sort_values(by='datetime')

## Converting type of the Columns data of Pollutants:
for i in df_sorted.columns:
	df_sorted[i] = df_sorted[i].astype(float)
print(df_sorted.info(),'df_sorted_info()')

print('>>>>>>>>>>>>>>>>  Feature engineering done')

####

##################    MODELING AND PICKLE GENERATION  #############################

####

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

print('>>>>>>>>>>>>>>>> pickle generation started for :', aqsid)
## Pickle generation:
df = df_sorted.copy()
print(df.info(),'df_info()')
values = df.values
column_names = df.columns.tolist()
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(values)
n_steps = 24
X, y = split_sequences(scaled_data, n_steps)
n_features = X.shape[2]
# Split the data into training and testing sets
train_size = int(len(df_sorted) * 0.2)
X_train, X_test = X[train_size:,] , X[:train_size,] 
# print('X_train' ,X_train.shape)
# print('X_test' ,X_test.shape)
Y_train, Y_test = y[train_size:,] , y[:train_size,]
# print('Y_train' ,Y_train.shape)
# print('Y_test' ,Y_test.shape)
n_features = X.shape[2]
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_train, Y_train, epochs=45, verbose =1 )
# save the model to disk
filename = '%s.pkl' % aqsid
pickle.dump(model, open(filename, 'wb'))


print('>>>>>>>>>>>>>>>> pickle generation ended for :', aqsid)



