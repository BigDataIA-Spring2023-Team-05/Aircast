import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
import aqi
import pickle
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData, DateTime
import boto3
import boto3.s3
import botocore

session = boto3.Session(
    region_name='us-east-1',
    aws_access_key_id='AKIAY5JK22NETTJWNQ46',
    aws_secret_access_key='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'
)

s3 = session.resource('s3')

# %%
src_bucket = s3.Bucket('damg-aircast')

## Importing 1 year data
engine = create_engine('mysql+pymysql://admin:123456789@aircast.cjisewdv5jgk.us-east-1.rds.amazonaws.com:3306/aircast')
my_list = ['90131001', '90159991', '90031003', '250170010', '90030025', '250030008', '250250045', '250251004', '90079007', '250212005', '90050005', '90110124', '90092123', '90099002', '90090027', '90011123', '90013007', '90010010', '90019003', '361030044', '90010017', '340310005', '230230007', '360850055', '340292002', '340210008', '100031012', '100010003', '245105253', '230290033', '240396431', '240476432']
counter = 0
# aqsid = '250250042'
for aqsid in my_list:

	####

	#####################   DATA EXTRACTION FROM AWS DataBase #############################

	####
	# Read data from the SQL table into a dataframe
	print('>>>>>>>>>>>>>>>>  data extraction started')
	inp_data = pd.read_sql_query('SELECT * from aircast.StationsData WHERE aquid = "%s"' % aqsid, engine)
	print('>>>>>>>>>>>>>>>>  data extraction done')

	####

	##################    FEATURE ENGINEERING  #############################
	print('>>>>>>>>>>>>>>>>  Feature engineering started')
	####
	# Renaming columns :
	inp_data = inp_data.rename(columns={'collection_timestamp':'datetime','ozone':'OZONE', 'so2':'SO2', 'no2':'NO2','co':'CO','pm2_5':'PM2.5','pm10':'PM10'})
	print(len(inp_data))
	df = inp_data.copy()
	
	if df.empty: 
		print('DataFrame is empty')
		counter = counter + 1
		print(counter)
	else:
		print('DataFrame is not empty')
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
		
		src_bucket.upload_file(filename, f"models/{filename}")

		print('>>>>>>>>>>>>>>>> pickle generation ended for :', aqsid)



