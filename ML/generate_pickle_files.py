import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
import pickle
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData, DateTime

## Importing 1 year data
engine = create_engine('mysql+pymysql://admin:123456789@aircast.cjisewdv5jgk.us-east-1.rds.amazonaws.com:3306/aircast')
my_list = ['420210011', '420270100', '420279991', '420290100', '420410101', '420430401', '420431100', '420450109', '420479991', '420550001', '420630004', '420690101', '420692006', '420710012', '420730015', '420770004', '420810100', '420850100', '420859991', '420890002', '420910013', '420950025', '421010004', '421010048', '421010055', '421010057', '421010075', '421010076', '421119991', '421174000', '421250005', '421255001', '421255200', '421290008', '421330008', '421330011', '440030002', '440070022', '440071010', '440090007', '450030003', '450190003', '450190046', '450250001', '450310003', '450450015', '450450016', '450510008', '450630010', '450790007', '450790021', '450791001', '450830009', '450830011', '450910008', '460110003', '460290002', '460330132', '460650003', '460710001', '461030020', '470010101', '470090101', '470090102', '470259991', '470370011', '470370026', '470419991', '470450004', '470651011', '470654002', '470654003', '470890002', '470930021', '470931013', '470931020', '471050109', '471071002', '471410005', '471570075', '471571004', '471631007', '471632002', '471632003', '471636002', '471650007', '471870106', '471890103', '480271045', '480271047', '480290052', '480290059', '480371031', '480391016', '480430101', '480551604', '480850005', '481130075', '481211032', '481391044', '481410029', '481410044', '481410057', '481410058', '481830001', '482010024', '482010026', '482010047', '482010051', '482010058', '482010062', '482011017', '482011034', '482011035', '482030002', '482210001', '482311006', '482450009', '482450022', '482510003', '482570005', '483031028', '483091037', '483230004', '483390078', '483491051', '483611001', '483670081', '483739991', '483819991', '483970001', '484392003', '484530020', '484690003', '484790313', '484910690', '490050007', '490071003', '490110004', '490130002', '490137011', '490210005', '490353006', '490353010', '490353013', '490370101', '490450004', '490471004', '490472002', '490472003', '490477022', '490494001', '490495010', '490530007', '490530130', '490571003', '500030004', '500070007', '500210002', '510030001', '510130020', '510330001', '510360002', '510410004', '510590030', '510610002', '510850003', '510870014', '511071005', '511130003', '511479991', '511530009', '511630003', '511650003', '511790001', '511970002', '516500008', '517100024', '530010003', '530030004', '530050002', '530070007', '530070010', '530070011', '530090013', '530110022', '530130002', '530150015', '530251002', '530251003', '530272002', '530330023', '530330031', '530330057', '530330080', '530331011', '530332004', '530350007', '530370002', '530410004', '530450007', '530470010', '530470013', '530530024', '530530029', '530570015', '530610020', '530611007', '530630001', '530630021', '530630047', '530639995', '530650005', '530670013', '530710005', '530710006', '530730019', '530750003', '530750005', '530750006', '530770005', '530770009', '540030003', '540250003', '540511002', '540610003', '540690010', '540939991', '541071002', '550090005', '550090026', '550210015', '550250041', '550290004', '550350014', '550390006', '550410007', '550430009', '550550009', '550590019', '550590025', '550610002', '550630012', '550710007', '550730012', '550790010', '550790085', '550850996', '550870009', '550890008', '550890009', '551050030', '551110007', '551170006', '551170009', '551198001', '551199991', '551250001', '560019991', '560030002', '560050123', '560130099', '560190004', '560210100', '560250100', '560330004', '560350099', '560350100', '560350101', '560350700', '560351002', '560359991', '560370200', '560370300', '560391011', '560450003', '800260006', 'MMMT10001', 'MMMT10006', 'TT9209004', '421010024', '160490002', '180550001', '480610006', '181270026', '220870004', '060072002', '150012020', '320230011', '471570021', '560330002', '420110006', '060070008', '191770006', '340390003', '470370023', '190450019', '420590002', '060793001', '120590004', '171670012', '330150016', '040213014', '191630017', '481411021', '180390008', '471450004', '270370470', '060010007', '040058001', '051130003', '060010012', '181230009', '320030073', '320030075', '320030602', '320031019', '320032003', '320037772', '360130006', '180570007', '320031502', '482451035', '261630033', '380530002', '261390005', '010735003', '482151046', '080450007', '120814012', '530530031', '060631006', '110010050', '484790016', '360470118', '360652001', '360050112', '121030023', '121035003', '180650003', '400670671', '400690324']
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



