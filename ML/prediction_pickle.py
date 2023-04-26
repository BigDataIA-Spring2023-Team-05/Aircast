import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
# import tensorflow as tf
import pickle
# import aqi
import pickle
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, MetaData, DateTime

## Importing 1 year data

engine = create_engine('mysql+pymysql://admin:123456789@aircast.cjisewdv5jgk.us-east-1.rds.amazonaws.com:3306/aircast')
my_list = my_list = ['250154002',
'250270015',
'250130008',
'250130018',
'90131001',
'330115001',
'250270024',
'330050007',
'90159991',
'330111011',
'250170009',
'90031003',
'250170010',
'90030025',
'250036001',
'250030008',
'250250002',
'250250042',
'440070022',
'440071010',
'250250045',
'250250044',
'440030002',
'250251004',
'250095005',
'90079007',
'250092006',
'500030004',
'330131006',
'250212005',
'250230005',
'90050005',
'250051004',
'90110124',
'90092123',
'440090007',
'90099002',
'360010012',
'90090027',
'250051006',
'360910004',
'330090010',
'330150014',
'360270007',
'330150016',
'330012004',
'500210002',
'90011123',
'90013007',
'90010010']

for aqsid in my_list:
    # Read data from the SQL table into a dataframe
    print('>>>>>>>>>>>>>>>>  data extraction started')
    inp_data = pd.read_sql_query('SELECT * from aircast.StationsDataDaily WHERE aquid = "%s"' % aqsid, engine)
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
    ## Pickle generation:
    df = df_sorted.copy()
    values = df.values
    column_names = df.columns.tolist()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(values)
    n_steps = 24
    X, y = split_sequences(scaled_data, n_steps)
    print(X.size)
    n_features = X.shape[2]

    # demonstrate prediction
    x_input = X
    x_input = x_input.reshape((24, n_steps, n_features))




    print('>>>>>>>>>>>>>>>>  Feature engineering done')


    # load the model from disk

    print('>>>>>>>>>>>>>>>>  Prediction started')
    filename = '%s.pkl' % aqsid
    model = pickle.load(open(filename, 'rb'))

    yhat = model.predict(x_input, verbose=0)
    req = scaler.inverse_transform(yhat)
    date_time_index = pd.date_range(start='00:00:00', end='23:00:00', freq='1H')
    final = pd.DataFrame(index=date_time_index, columns=column_names, data = req)
    

    print('>>>>>>>>>>>>>>>> pickle generation ended for :', aqsid)