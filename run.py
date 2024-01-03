import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from utils import *
from model import MODEL
from CONSTANTS import TOP_N
import math

def preprocessing_data(df, seq_len, ppd):
  
  #print length of df
    print('The length of dataframe : ', len(df))
    
    # windowing data
    num_sample = int(len(df)/seq_len)
    time_dataset = []
    for i in range(num_sample):
        t_slice = df.iloc[(i*seq_len):((i+1)*seq_len)]
        time_dataset.append(t_slice)
    # print number of windowing data
    print('The number of windowing data : ', len(time_dataset))
    
    to_drop = ['time', 'confidence', 'handlabeller1', 'handlabeller2', 'handlabeller_final', 'acceleration_1', 'acceleration_2', 'acceleration_4', 'acceleration_8', 'acceleration_16']
    X = []
    for sample in time_dataset:
        features = sample.drop(to_drop, axis=1) # drop necessary features
        for col in features.columns: # normalize features
            features[col] /= ppd
        X.append(features)
    concat_X =pd.concat(X, axis=0)
    np_X = concat_X.values
    X = np_X.reshape(num_sample, seq_len, -1)
    
    #one hot encoding
    df_label = df['handlabeller_final'].values
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(df_label.reshape(-1,1))
    OHE_Y = y_onehot.reshape(num_sample, seq_len, -1)
    
    print('The shape of time series data: ', X.shape)
    print('The shape of time series data label: ', OHE_Y.shape)
    return [X, OHE_Y]

def calculate_ppd(width_px, height_px, width_mm, height_mm, distance):
    theta_w = 2 * math.atan(width_mm / (2 * distance)) * 180. / math.pi
    theta_h = 2 * math.atan(height_mm / (2 * distance)) * 180. / math.pi

    ppdx = width_px / theta_w
    ppdy = height_px / theta_h

    return (ppdx + ppdy) / 2
  
# Hitung PPD
width_px = 1280
height_px = 720
width_mm = 400
height_mm = 225.0
distance = 450.0
ppd = calculate_ppd(width_px, height_px, width_mm, height_mm, distance)
print("ppd value: ", ppd)

seq_len = 65
df = pd.read_csv(r'C:\Users\Asus\Skripsi\MLPNAS_2\cnnblstm\DATA\all_koenigstrasse_csv\merged_data_train_val_3.csv')
# make length of df can be divided 
df = df.iloc[:-61]

dataset = preprocessing_data(df, seq_len,ppd)
x, y = dataset
#print("cek isi x\n", x)
#print("cek isi y\n", y)

nas_object = MODEL(x, y)
data = nas_object.search()

get_top_n_architectures(TOP_N)
get_nas_f1_plot()
get_f1_distribution()