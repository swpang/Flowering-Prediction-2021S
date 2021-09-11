import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import multiprocessing
import os
from datetime import datetime
import statistics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer

seed = 20210811
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.set_default_dtype(torch.float64)

print('PyTorch version : {}'.format(torch.__version__))
print("CUDA is available : ", torch.cuda.is_available())
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
print(torch.device)

print("File __name__ is set to: {}" .format(__name__))

# 경로 수정해주어야함
path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\DATA\\'
results_path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\RESULTS\\'
model_path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\MODELS\\'

np_x_1 = pd.read_csv(path + 'x_1wjeju.csv', header=0, index_col=0).to_numpy()
np_x_2 = pd.read_csv(path + 'x_2.csv', header=0, index_col=0).to_numpy()
np_x_4 = pd.read_csv(path + 'x_4.csv', header=0, index_col=0).to_numpy()
np_x_5 = pd.read_csv(path + 'x_5.csv', header=0, index_col=0).to_numpy()
np_y_1 = pd.read_csv(path + 'y_1wjeju.csv', header=0, index_col=0).to_numpy()
np_y_2 = pd.read_csv(path + 'y_2.csv', header=0, index_col=0).to_numpy()
np_y_4 = pd.read_csv(path + 'y_4.csv', header=0, index_col=0).to_numpy()
np_y_5 = pd.read_csv(path + 'y_5.csv', header=0, index_col=0).to_numpy()
np_x_1_seoul = pd.read_csv(path + 'x_1_seoul.csv', header=0, index_col=0).to_numpy()
np_x_2_seoul = pd.read_csv(path + 'x_2_seoul.csv', header=0, index_col=0).to_numpy()
np_x_4_seoul = pd.read_csv(path + 'x_4_seoul.csv', header=0, index_col=0).to_numpy()
np_x_5_seoul = pd.read_csv(path + 'x_5_seoul.csv', header=0, index_col=0).to_numpy()
np_y_1_seoul = pd.read_csv(path + 'y_1_seoul.csv', header=0, index_col=0).to_numpy()
np_y_2_seoul = pd.read_csv(path + 'y_2_seoul.csv', header=0, index_col=0).to_numpy()
np_y_4_seoul = pd.read_csv(path + 'y_4_seoul.csv', header=0, index_col=0).to_numpy()
np_y_5_seoul = pd.read_csv(path + 'y_5_seoul.csv', header=0, index_col=0).to_numpy()

df_1_data = pd.DataFrame(np.hstack((np_x_1, np_y_1)))
df_2_data = pd.DataFrame(np.hstack((np_x_2, np_y_2)))
df_4_data = pd.DataFrame(np.hstack((np_x_4, np_y_4)))
df_5_data = pd.DataFrame(np.hstack((np_x_5, np_y_5)))

df_1_seoul_data = pd.DataFrame(np.hstack((np_x_1_seoul, np_y_1_seoul)))
df_2_seoul_data = pd.DataFrame(np.hstack((np_x_2_seoul, np_y_2_seoul)))
df_4_seoul_data = pd.DataFrame(np.hstack((np_x_4_seoul, np_y_4_seoul)))
df_5_seoul_data = pd.DataFrame(np.hstack((np_x_5_seoul, np_y_5_seoul)))

print(df_2_data.head())
print(df_4_seoul_data.head())

# 각 데이터셋 정리하는 코드 (0, nan 값 없애줌)

df_1_data = df_1_data.replace(0, np.nan)
df_2_data = df_2_data.replace(0, np.nan)
df_4_data = df_4_data.replace(0, np.nan)
df_5_data = df_5_data.replace(0, np.nan)
df_1_data = df_1_data.dropna(axis=0, how='any')
df_2_data = df_2_data.dropna(axis=0, how='any')
df_4_data = df_4_data.dropna(axis=0, how='any')
df_5_data = df_5_data.dropna(axis=0, how='any')

df_1_seoul_data = df_1_seoul_data.replace(0, np.nan)
df_2_seoul_data = df_2_seoul_data.replace(0, np.nan)
df_4_seoul_data = df_4_seoul_data.replace(0, np.nan)
df_5_seoul_data = df_5_seoul_data.replace(0, np.nan)
df_1_seoul_data = df_1_seoul_data.dropna(axis=0, how='any')
df_2_seoul_data = df_2_seoul_data.dropna(axis=0, how='any')
df_4_seoul_data = df_4_seoul_data.dropna(axis=0, how='any')
df_5_seoul_data = df_5_seoul_data.dropna(axis=0, how='any')

df_1_data.to_csv(path + 'x_1.csv', columns=[0,1,2])
df_2_data.to_csv(path + 'x_2.csv', columns=[0,1,2])
df_4_data.to_csv(path + 'x_4.csv', columns=[0,1,2,3])
df_5_data.to_csv(path + 'x_5.csv', columns=[0,1,2,3])
df_1_data.to_csv(path + 'y_1.csv', columns=[3])
df_2_data.to_csv(path + 'y_2.csv', columns=[3])
df_4_data.to_csv(path + 'y_4.csv', columns=[4])
df_5_data.to_csv(path + 'y_5.csv', columns=[4])

df_1_seoul_data.to_csv(path + 'x_1_seoul.csv', columns=[0,1,2])
df_2_seoul_data.to_csv(path + 'x_2_seoul.csv', columns=[0,1,2])
df_4_seoul_data.to_csv(path + 'x_4_seoul.csv', columns=[0,1,2,3])
df_5_seoul_data.to_csv(path + 'x_5_seoul.csv', columns=[0,1,2,3])
df_1_seoul_data.to_csv(path + 'y_1_seoul.csv', columns=[3])
df_2_seoul_data.to_csv(path + 'y_2_seoul.csv', columns=[3])
df_4_seoul_data.to_csv(path + 'y_4_seoul.csv', columns=[4])
df_5_seoul_data.to_csv(path + 'y_5_seoul.csv', columns=[4])
