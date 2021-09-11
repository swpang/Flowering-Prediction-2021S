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

############################################################################################################################################################################

path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\DATA\\'
results_path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\RESULTS\\'
model_path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\MODELS\\'

df_stations = pd.read_csv(path + 'stations_info.csv', header=0, index_col=None)
df_flowering = pd.read_csv(path + 'season_f.csv', header=None, index_col=None, names=['stnId', 'lat', 'lon', 'year', 'date'])
df_budding = pd.read_csv(path + 'season_g.csv', header=None, index_col=None, names=['stnId', 'lat', 'lon', 'year', 'date'])

dict_weather = {}

for stnId in df_stations['stnId']:
    if os.path.exists(path + 'weatherData\\{}_data.csv'.format(stnId)):
        df_temp = pd.read_csv(path + 'weatherData\\{}_data.csv'.format(stnId), header=0, index_col=0)
        dict_weather[str(stnId)] = df_temp

CRITICAL_TEMP = 3
CRITICAL_TEMP_2 = 3

############################################################################################################################################################################

# 각 지점의 데이터에 대해 각 모델 입력 데이터 생성

############################################################################################################################################################################
def preprocess_data_4(stnId):
    try:
        np_x1 = []
        np_x2 = []
        np_x3 = []
        np_ids = []
        years = []
        budding_temp = []

        df_temp = dict_weather[str(stnId)]
        
        for i, rows in df_budding.iterrows():
            if int(rows['stnId']) == int(stnId):
                budding_temp.append(rows['date'])
                years.append(rows['year'])
                np_ids.append(str(rows['year']) + '_' + str(stnId))
        np_y = budding_temp
        
        for i, year in enumerate(years):
            if budding_temp[i] >= 0:
                bdate = budding_temp[i]
                tempSum1, tempSum2, tempSum3 = 0, 0, 0
                for j, rows in df_temp.iterrows():
                    if rows['year'] == year:
                        if int(rows['jday']) <= bdate:
                            if (float(rows['avgTa']) >= 0 or float(rows['avgTa']) < 0) and (float(rows['sumRn']) >= 0 or float(rows['sumRn']) < 0):
                                avgTa = float(rows['avgTa'])
                                jday = float(rows['jday'])
                                tempSum1 += jday * max(avgTa - CRITICAL_TEMP, 0)
                                sumRn = float(rows['sumRn'])
                                tempSum2 += jday * sumRn
                                tempSum3 += float(rows['sumSsHr'])
                np_x1.append(tempSum1)
                np_x2.append(tempSum2)
                np_x3.append(tempSum3)
            del tempSum1, tempSum2, tempSum3, bdate, rows

        np_x1 = np.array(np_x1).reshape(-1, 1)
        np_x2 = np.array(np_x2).reshape(-1, 1)
        np_x3 = np.array(np_x3).reshape(-1, 1)
        np_y = np.array(np_y).reshape(-1, 1)
        np_ids = np.array(np_ids).reshape(-1, 1)

        return np_x1, np_x2, np_x3, np_y, np_ids
    except KeyError:
        print('Station {} Data Doesn\'t Exist'.format(stnId))
############################################################################################################################################################################
def preprocess_data_1(stnId):
    try:
        np_x1 = []
        np_x2 = []
        np_ids = []
        years = []
        budding_temp = []

        df_temp = dict_weather[str(stnId)]
        
        for i, rows in df_budding.iterrows():
            if int(rows['stnId']) == int(stnId):
                budding_temp.append(rows['date'])
                years.append(rows['year'])
                np_ids.append(str(rows['year']) + '_' + str(stnId))
        np_y = budding_temp
        
        for i, year in enumerate(years):
            if budding_temp[i] >= 0:
                bdate = budding_temp[i]
                tempSum1, tempSum2 = 0, 0
                for j, rows in df_temp.iterrows():
                    if rows['year'] == year:
                        if int(rows['jday']) <= bdate:
                            if (float(rows['avgTa']) >= 0 or float(rows['avgTa']) < 0) and (float(rows['sumRn']) >= 0 or float(rows['sumRn']) < 0):
                                avgTa = float(rows['avgTa'])
                                jday = float(rows['jday'])
                                tempSum1 += jday * max(avgTa - CRITICAL_TEMP, 0)
                                sumRn = float(rows['sumRn'])
                                tempSum2 += jday * sumRn
                np_x1.append(tempSum1)
                np_x2.append(tempSum2)
            del tempSum1, tempSum2, bdate, rows

        np_x1 = np.array(np_x1).reshape(-1, 1)
        np_x2 = np.array(np_x2).reshape(-1, 1)
        np_y = np.array(np_y).reshape(-1, 1)
        np_ids = np.array(np_ids).reshape(-1, 1)

        return np_x1, np_x2, np_y, np_ids
    except KeyError:
        print('Station {} Data Doesn\'t Exist'.format(stnId))
############################################################################################################################################################################
def preprocess_data_2(stnId):
    try:
        np_x1 = []
        np_x2 = []
        np_ids = []
        years = []
        budding_temp = []

        df_temp = dict_weather[str(stnId)]
        
        for i, rows in df_budding.iterrows():
            if int(rows['stnId']) == int(stnId):
                budding_temp.append(rows['date'])
                years.append(rows['year'])
                np_ids.append(str(rows['year']) + '_' + str(stnId))
        np_y = budding_temp
        
        for i, year in enumerate(years):
            if budding_temp[i] >= 0:
                bdate = budding_temp[i]
                tempSum1, tempSum2 = 0, 0
                for j, rows in df_temp.iterrows():
                    if rows['year'] == year:
                        if int(rows['jday']) <= bdate:
                            if (float(rows['avgTa']) >= 0 or float(rows['avgTa']) < 0) and (float(rows['sumRn']) >= 0 or float(rows['sumRn']) < 0):
                                tempSum1 += max(float(rows['avgTa']) - CRITICAL_TEMP, 0)
                                tempSum2 += float(rows['sumRn'])
                np_x1.append(tempSum1)
                np_x2.append(tempSum2)
            del tempSum1, tempSum2, bdate, rows

        np_x1 = np.array(np_x1).reshape(-1, 1)
        np_x2 = np.array(np_x2).reshape(-1, 1)
        np_y = np.array(np_y).reshape(-1, 1)
        np_ids = np.array(np_ids).reshape(-1, 1)

        return np_x1, np_x2, np_y, np_ids
    except KeyError:
        print('Station {} Data Doesn\'t Exist'.format(stnId))
############################################################################################################################################################################
def preprocess_data_3(stnId):
    try:
        np_x1 = []
        np_x2 = []
        np_x3 = []
        np_ids = []
        years = []
        budding_temp = []

        df_temp = dict_weather[str(stnId)]
        
        for i, rows in df_budding.iterrows():
            if int(rows['stnId']) == int(stnId):
                budding_temp.append(rows['date'])
                years.append(rows['year'])
                np_ids.append(str(rows['year']) + '_' + str(stnId))
        np_y = budding_temp
        
        for i, year in enumerate(years):
            if budding_temp[i] >= 0:
                bdate = budding_temp[i]
                tempSum1, tempSum2, tempSum3 = 0, 0, 0
                for j, rows in df_temp.iterrows():
                    if rows['year'] == year:
                        if int(rows['jday']) <= bdate:
                            if (float(rows['avgTa']) >= 0 or float(rows['avgTa']) < 0) and (float(rows['sumRn']) >= 0 or float(rows['sumRn']) < 0) and (float(rows['sumSsHr']) >= 0 or float(rows['sumSsHr']) < 0):
                                tempSum1 += max(float(rows['avgTa']) - CRITICAL_TEMP, 0)
                                tempSum2 += float(rows['sumRn'])
                                tempSum3 += float(rows['sumSsHr'])
                np_x1.append(tempSum1)
                np_x2.append(tempSum2)
                np_x3.append(tempSum3)
            del tempSum1, tempSum2, tempSum3, bdate, rows

        np_x1 = np.array(np_x1).reshape(-1, 1)
        np_x2 = np.array(np_x2).reshape(-1, 1)
        np_x3 = np.array(np_x3).reshape(-1, 1)
        np_y = np.array(np_y).reshape(-1, 1)
        np_ids = np.array(np_ids).reshape(-1, 1)

        return np_x1, np_x2, np_x3, np_y, np_ids
    except KeyError:
        print('Station {} Data Doesn\'t Exist'.format(stnId))
############################################################################################################################################################################

np_x1_5 = np.zeros((1,1))
np_x2_5 = np.zeros((1,1))
np_x3_5 = np.zeros((1,1))
np_ids_5 = np.zeros((1,1))
np_y_5 = np.zeros((1,1))

np_x1_5_seoul = np.zeros((1,1))
np_x2_5_seoul = np.zeros((1,1))
np_x3_5_seoul = np.zeros((1,1))
np_ids_5_seoul = np.zeros((1,1))
np_y_5_seoul = np.zeros((1,1))

for stnId in df_stations['stnId']:
    if not stnId in [184, 185, 188, 189]:
        try:
            print('Station : {}'.format(stnId))
            np_x1_temp, np_x2_temp, np_x3_temp, np_y_temp, np_ids_temp = preprocess_data_5(str(stnId))
        except TypeError:
            print('Skipping Station {}'.format(stnId))
        np_x1_5 = np.vstack((np_x1_5, np_x1_temp)).reshape(-1,1)
        np_x2_5 = np.vstack((np_x2_5, np_x2_temp)).reshape(-1,1)
        np_x3_5 = np.vstack((np_x3_5, np_x3_temp)).reshape(-1,1)
        np_ids_5 = np.vstack((np_ids_5, np_ids_temp)).reshape(-1,1)
        np_y_5 = np.vstack((np_y_5, np_y_temp)).reshape(-1,1)

        if stnId == 108:
            np_x1_5_seoul = np.vstack((np_x1_5_seoul, np_x1_temp)).reshape(-1,1)
            np_x2_5_seoul = np.vstack((np_x2_5_seoul, np_x2_temp)).reshape(-1,1)
            np_x3_5_seoul = np.vstack((np_x3_5_seoul, np_x3_temp)).reshape(-1,1)
            np_ids_5_seoul = np.vstack((np_ids_5_seoul, np_ids_temp)).reshape(-1,1)
            np_y_5_seoul = np.vstack((np_y_5_seoul, np_y_temp)).reshape(-1,1)
        
np_x1_5 = np.delete(np_x1_5, 0, 0)
np_x2_5 = np.delete(np_x2_5, 0, 0)
np_x3_5 = np.delete(np_x3_5, 0, 0)
np_y_5 = np.delete(np_y_5, 0, 0)
np_ids_5 = np.delete(np_ids_5, 0, 0)

np_x1_5_seoul = np.delete(np_x1_5_seoul, 0, 0)
np_x2_5_seoul = np.delete(np_x2_5_seoul, 0, 0)
np_x3_5_seoul = np.delete(np_x3_5_seoul, 0, 0)
np_y_5_seoul = np.delete(np_y_5_seoul, 0, 0)
np_ids_5_seoul = np.delete(np_ids_5_seoul, 0, 0)

np_x_5 = np.hstack((np_ids_5, np_x1_5, np_x2_5, np_x3_5)).reshape(-1, 4)
np_x_5_seoul = np.hstack((np_ids_5_seoul, np_x1_5_seoul, np_x2_5_seoul, np_x3_5_seoul)).reshape(-1, 4)

df_x_5 = pd.DataFrame(np_x_5)
df_y_5 = pd.DataFrame(np_y_5)

df_x_5_seoul = pd.DataFrame(np_x_5_seoul)
df_y_5_seoul = pd.DataFrame(np_y_5_seoul)

df_x_5.to_csv(path + 'x_5.csv')
df_y_5.to_csv(path + 'y_5.csv')

df_x_5_seoul.to_csv(path + 'x_5_seoul.csv')
df_y_5_seoul.to_csv(path + 'y_5_seoul.csv')

############################################################################################################################################################################

np_x1_1_seoul = np.zeros((1,1))
np_x2_1_seoul = np.zeros((1,1))
np_ids_1_seoul = np.zeros((1,1))
np_y_1_seoul = np.zeros((1,1))

for stnId in df_stations['stnId']:
    if stnId == 108:
        try:
            print('Station : {}'.format(stnId))
            np_x1_temp, np_x2_temp, np_y_temp, np_ids_temp = preprocess_data_1(str(stnId))
        except TypeError:
            print('Skipping Station {}'.format(stnId))
        np_x1_1_seoul = np.vstack((np_x1_1_seoul, np_x1_temp)).reshape(-1,1)
        np_x2_1_seoul = np.vstack((np_x2_1_seoul, np_x2_temp)).reshape(-1,1)
        np_ids_1_seoul = np.vstack((np_ids_1_seoul, np_ids_temp)).reshape(-1,1)
        np_y_1_seoul = np.vstack((np_y_1_seoul, np_y_temp)).reshape(-1,1)

np_x1_1_seoul = np.delete(np_x1_1_seoul, 0, 0)
np_x2_1_seoul = np.delete(np_x2_1_seoul, 0, 0)
np_y_1_seoul = np.delete(np_y_1_seoul, 0, 0)
np_ids_1_seoul = np.delete(np_ids_1_seoul, 0, 0)

np_x_1_seoul = np.hstack((np_ids_1_seoul, np_x1_1_seoul, np_x2_1_seoul)).reshape(-1, 3)

df_x_1_seoul = pd.DataFrame(np_x_1_seoul)
df_y_1_seoul = pd.DataFrame(np_y_1_seoul)

df_x_1_seoul.to_csv(path + 'x_1_seoul.csv')
df_y_1_seoul.to_csv(path + 'y_1_seoul.csv')

############################################################################################################################################################################

np_x1_2_seoul = np.zeros((1,1))
np_x2_2_seoul = np.zeros((1,1))
np_ids_2_seoul = np.zeros((1,1))
np_y_2_seoul = np.zeros((1,1))

for stnId in df_stations['stnId']:
    if stnId == 108:
        try:
            print('Station : {}'.format(stnId))
            np_x1_temp, np_x2_temp, np_y_temp, np_ids_temp = preprocess_data_2(str(stnId))
        except TypeError:
            print('Skipping Station {}'.format(stnId))
        np_x1_2_seoul = np.vstack((np_x1_2_seoul, np_x1_temp)).reshape(-1,1)
        np_x2_2_seoul = np.vstack((np_x2_2_seoul, np_x2_temp)).reshape(-1,1)
        np_ids_2_seoul = np.vstack((np_ids_2_seoul, np_ids_temp)).reshape(-1,1)
        np_y_2_seoul = np.vstack((np_y_2_seoul, np_y_temp)).reshape(-1,1)

np_x1_2_seoul = np.delete(np_x1_2_seoul, 0, 0)
np_x2_2_seoul = np.delete(np_x2_2_seoul, 0, 0)
np_y_2_seoul = np.delete(np_y_2_seoul, 0, 0)
np_ids_2_seoul = np.delete(np_ids_2_seoul, 0, 0)

np_x_2_seoul = np.hstack((np_ids_2_seoul, np_x1_2_seoul, np_x2_2_seoul)).reshape(-1, 3)

df_x_2_seoul = pd.DataFrame(np_x_2_seoul)
df_y_2_seoul = pd.DataFrame(np_y_2_seoul)

df_x_2_seoul.to_csv(path + 'x_2_seoul.csv')
df_y_2_seoul.to_csv(path + 'y_2_seoul.csv')

############################################################################################################################################################################

np_x1_4_seoul = np.zeros((1,1))
np_x2_4_seoul = np.zeros((1,1))
np_x3_4_seoul = np.zeros((1,1))
np_ids_4_seoul = np.zeros((1,1))
np_y_4_seoul = np.zeros((1,1))

for stnId in df_stations['stnId']:
    if stnId == 108:
        try:
            print('Station : {}'.format(stnId))
            np_x1_temp, np_x2_temp, np_x3_temp, np_y_temp, np_ids_temp = preprocess_data_4(str(stnId))
        except TypeError:
            print('Skipping Station {}'.format(stnId))
        np_x1_4_seoul = np.vstack((np_x1_4_seoul, np_x1_temp)).reshape(-1,1)
        np_x2_4_seoul = np.vstack((np_x2_4_seoul, np_x2_temp)).reshape(-1,1)
        np_x3_4_seoul = np.vstack((np_x3_4_seoul, np_x3_temp)).reshape(-1,1)
        np_ids_4_seoul = np.vstack((np_ids_4_seoul, np_ids_temp)).reshape(-1,1)
        np_y_4_seoul = np.vstack((np_y_4_seoul, np_y_temp)).reshape(-1,1)

np_x1_4_seoul = np.delete(np_x1_4_seoul, 0, 0)
np_x2_4_seoul = np.delete(np_x2_4_seoul, 0, 0)
np_x3_4_seoul = np.delete(np_x3_4_seoul, 0, 0)
np_y_4_seoul = np.delete(np_y_4_seoul, 0, 0)
np_ids_4_seoul = np.delete(np_ids_4_seoul, 0, 0)

np_x_4_seoul = np.hstack((np_ids_4_seoul, np_x1_4_seoul, np_x2_4_seoul, np_x3_4_seoul)).reshape(-1, 4)

df_x_4_seoul = pd.DataFrame(np_x_4_seoul)
df_y_4_seoul = pd.DataFrame(np_y_4_seoul)

df_x_4_seoul.to_csv(path + 'x_4_seoul.csv')
df_y_4_seoul.to_csv(path + 'y_4_seoul.csv')

############################################################################################################################################################################