import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from utils import *
from model import MODEL
from CONSTANTS import TOP_N
import math

def load_data(df, seq_len,ppd_value):
  """ Load and preprocess data

  Args:
  - df : dataframe of tubular data (df.iloc[i] == one row)
  - seq_len : length of one sample/sequence

  returns:
  - [X, trainy]
     X      : time series data (num_sample, seq_len, feature) no overlapping,
              numpy array, rescaled, float32
     trainy : labels of each sample in the time series data, numpy array
  -  scaler : object from MinMaxScaler(), this maps the raw data and the [-1,1]
              range
  """
  # Convert tubular data to 3-dimension time series data
  num_sample = int(len(df)/seq_len)
  time_dataset = []
  for i in range(num_sample):
    t_slice = df.iloc[(i*seq_len) : ((i+1)*seq_len)]
    time_dataset.append(t_slice)

  # Get the label of each sample
  trainy= df['handlabeller_final'].values
  encoder = OneHotEncoder(sparse=False)
  OHE_y = encoder.fit_transform(trainy.reshape(-1, 1))
  OHE_Y = OHE_y.reshape(num_sample, seq_len, -1)

  # Get the data without label
  to_drop = ['time', 'confidence', 'handlabeller1', 'handlabeller2', 'handlabeller_final', 'speed_1', 'direction_1', 'acceleration_1', 'speed_2', 'direction_2', 'acceleration_2', 'speed_4', 'direction_4', 'acceleration_4', 'speed_8', 'direction_8', 'acceleration_8', 'acceleration_16']
  X= []
  for sample in time_dataset :
    features = sample.drop(to_drop, axis=1)
    
    #normalize using ppd
    for col in features.columns:
      features[col] /= ppd_value
      
    X.append(features)

  # Convert the list of dataframe to numpy array and normalize/rescale it
  concat_X = pd.concat(X, axis=0)
  numpy_X = concat_X.values
  
  #convert/change the shape of X to (num_sample, seq_len, feature)
  X = numpy_X.reshape(num_sample, seq_len, -1)
  #trainy = np.array(trainy)
  print('The shape of time series data : ', X.shape)
  print('The shape of time series data label : ', OHE_Y.shape)
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
df = pd.read_csv(r'C:\Users\Asus\Skripsi\MLPNAS_2\cnnblstm\DATA\merged_all_bridge2.csv')
# make length of df can be divided 
df = df.iloc[:-8]

dataset = load_data(df, seq_len,ppd)
x, y = dataset
#print("cek isi x\n", x)
#print("cek isi y\n", y)

nas_object = MODEL(x, y)
data = nas_object.search()

get_top_n_architectures(TOP_N)
get_nas_f1_plot()
get_f1_distribution()