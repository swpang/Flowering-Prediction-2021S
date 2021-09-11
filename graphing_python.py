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

path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\DATA\\'
results_path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\RESULTS\\'
model_path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\MODELS\\'

np_result_1 = pd.read_csv(results_path + 'model1_results.csv', header=0, index_col=0).to_numpy()
np_result_2 = pd.read_csv(results_path + 'model2_results.csv', header=0, index_col=0).to_numpy()
np_result_4 = pd.read_csv(results_path + 'model4_results.csv', header=0, index_col=0).to_numpy()
np_result_5 = pd.read_csv(results_path + 'model5_results.csv', header=0, index_col=0).to_numpy()

np_budding_1 = pd.read_csv(results_path + '1_seoul_results.csv', header=0, index_col=0).to_numpy()
np_budding_2 = pd.read_csv(results_path + '2_seoul_results.csv', header=0, index_col=0).to_numpy()
np_budding_4 = pd.read_csv(results_path + '4_seoul_results.csv', header=0, index_col=0).to_numpy()
np_budding_5 = pd.read_csv(results_path + '5_seoul_results.csv', header=0, index_col=0).to_numpy()

np_flowering = pd.read_csv(path + 'flowering_data.csv', header=0, index_col=0).to_numpy()
df_budding = pd.read_csv(path + 'seoul_data.csv', header=0, index_col=0)

print(df_budding.head())

x = np.linspace(0, 1000)

#################################################################################################################################################

for i, item in enumerate(np_budding_1[:, 0]):
    np_budding_1[i, 0] = item // 1000
    for j, rows in df_budding.iterrows():
        if int(rows['year']) == int(np_budding_1[i, 0]):
            np_budding_1[i, 0] = rows['budding date']
    if np_budding_1[i, 0] == item // 1000:
        np_budding_1[i, 0] = np.nan
for i, item in enumerate(np_budding_2[:, 0]):
    np_budding_2[i, 0] = item // 1000
    for j, rows in df_budding.iterrows():
        if int(rows['year']) == int(np_budding_2[i, 0]):
            np_budding_2[i, 0] = rows['budding date']
    if np_budding_2[i, 0] == item // 1000:
        np_budding_2[i, 0] = np.nan
for i, item in enumerate(np_budding_4[:, 0]):
    np_budding_4[i, 0] = item // 1000
    for j, rows in df_budding.iterrows():
        if int(rows['year']) == int(np_budding_4[i, 0]):
            np_budding_4[i, 0] = rows['budding date']
    if np_budding_4[i, 0] == item // 1000:
        np_budding_4[i, 0] = np.nan
for i, item in enumerate(np_budding_5[:, 0]):
    np_budding_5[i, 0] = item // 1000
    for j, rows in df_budding.iterrows():
        if int(rows['year']) == int(np_budding_5[i, 0]):
            np_budding_5[i, 0] = rows['budding date']
    if np_budding_5[i, 0] == item // 1000:
        np_budding_5[i, 0] = np.nan

sum, count = 0, 0
for i, item in enumerate(np_budding_1[:, 1]):
    if np_budding_1[i, 1] > 0 and np_budding_1[i, 0] > 0:
        sum += (np_budding_1[i, 1] - np_budding_1[i, 0]) ** 2
        count += 1
print('Model 1 RMSE Score : {}'.format(np.sqrt(sum / count)))

sum, count = 0, 0
for i, item in enumerate(np_budding_2[:, 1]):
    if np_budding_2[i, 1] > 0 and np_budding_2[i, 0] > 0:
        sum += (np_budding_2[i, 1] - np_budding_2[i, 0]) ** 2
        count += 1
print('Model 2 RMSE Score : {}'.format(np.sqrt(sum / count)))

sum, count = 0, 0
for i, item in enumerate(np_budding_4[:, 1]):
    if np_budding_4[i, 1] > 0 and np_budding_4[i, 0] > 0:
        sum += (np_budding_4[i, 1] - np_budding_4[i, 0]) ** 2
        count += 1
print('Model 3 RMSE Score : {}'.format(np.sqrt(sum / count)))

sum, count = 0, 0
for i, item in enumerate(np_budding_5[:, 1]):
    if np_budding_5[i, 1] > 0 and np_budding_5[i, 0] > 0:
        sum += (np_budding_5[i, 1] - np_budding_5[i, 0]) ** 2
        count += 1
print('Model 4 RMSE Score : {}'.format(np.sqrt(sum / count)))

print('________________________________________________')



plt.figure(figsize=(6,6))
plt.plot(x,x, color='black')
plt.scatter(np_budding_1[:, 0], np_budding_1[:, 1], color='black', alpha=0.75, s=15)
plt.title('True - Predicted Budding Date ML Model 1 (Seoul 108)')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.xlim(50, 130)
plt.ylim(50, 130)
plt.show()

plt.figure(figsize=(6,6))
plt.plot(x,x, color='black')
plt.scatter(np_budding_2[:, 0], np_budding_2[:, 1], color='black', alpha=0.75, s=15)
plt.title('True - Predicted Budding Date ML Model 2 (Seoul 108)')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.xlim(50, 130)
plt.ylim(50, 130)
plt.show()

plt.figure(figsize=(6,6))
plt.plot(x,x, color='black')
plt.scatter(np_budding_4[:, 0], np_budding_4[:, 1], color='black', alpha=0.75, s=15)
plt.title('True - Predicted Budding Date ML Model 3 (Seoul 108)')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.xlim(50, 130)
plt.ylim(50, 130)
plt.show()

plt.figure(figsize=(6,6))
plt.plot(x,x, color='black')
plt.scatter(np_budding_5[:, 0], np_budding_5[:, 1], color='black', alpha=0.75, s=15)
plt.title('True - Predicted Budding Date ML Model 4 (Seoul 108)')
plt.xlabel('True')
plt.ylabel('Predicted')
plt.xlim(50, 130)
plt.ylim(50, 130)
plt.show()

#################################################################################################################################################

sum, count = 0, 0
for i, item in enumerate(np_result_1[:, 1]):
    np_result_1[i, 1] = item + 10
    if np_result_1[i, 1] > 0 and np_result_1[i, 0] > 0:
        sum += (np_result_1[i, 1] - np_result_1[i, 0]) ** 2
        count += 1
print('Model 1 RMSE Score : {}'.format(np.sqrt(sum / count)))

plt.figure(figsize=(6,6))
plt.plot(x,x, color='black')
plt.scatter(np_flowering[:,1], np_flowering[:,2], color='black', alpha=0.75, s=15, label='Budding Inconsidered Baseline (GDD)')
plt.scatter(np_result_1[:,0], np_result_1[:,1], color='red', alpha=0.75, s=15, label='Budding Considered Model 1')
plt.xlabel('True')
plt.ylabel('Predict')
plt.title('ML Model Predicted - True FFD (Seoul, 108)')
plt.legend()
plt.xlim(60,150)
plt.ylim(60,150)
plt.show()

sum, count = 0, 0
for i, item in enumerate(np_result_2[:, 1]):
    np_result_2[i, 1] = item + 10
    if np_result_2[i, 1] > 0 and np_result_2[i, 0] > 0:
        sum += (np_result_2[i, 1] - np_result_2[i, 0]) ** 2
        count += 1
print('Model 2 RMSE Score : {}'.format(np.sqrt(sum / count)))

plt.figure(figsize=(6,6))
plt.plot(x,x, color='black')
plt.scatter(np_flowering[:,1], np_flowering[:,2], color='black', alpha=0.75, s=15, label='Budding Inconsidered Baseline (GDD)')
plt.scatter(np_result_2[:,0], np_result_2[:,1], color='red', alpha=0.75, s=15, label='Budding Considered Model 2')
plt.xlabel('True')
plt.ylabel('Predict')
plt.title('ML Model Predicted - True FFD (Seoul, 108)')
plt.legend()
plt.xlim(60,150)
plt.ylim(60,150)
plt.show()

sum, count = 0, 0
for i, item in enumerate(np_result_4[:, 1]):
    np_result_4[i, 1] = item + 10
    if np_result_4[i, 1] > 0 and np_result_4[i, 0] > 0:
        sum += (np_result_4[i, 1] - np_result_4[i, 0]) ** 2
        count += 1
print('Model 4 RMSE Score : {}'.format(np.sqrt(sum / count)))

plt.figure(figsize=(6,6))
plt.plot(x,x, color='black')
plt.scatter(np_flowering[:,1], np_flowering[:,2], color='black', alpha=0.75, s=15, label='Budding Inconsidered Baseline (GDD)')
plt.scatter(np_result_4[:,0], np_result_4[:,1], color='red', alpha=0.75, s=15, label='Budding Considered Model 3')
plt.xlabel('True')
plt.ylabel('Predict')
plt.title('ML Model Predicted - True FFD (Seoul, 108)')
plt.legend()
plt.xlim(60,150)

plt.ylim(60,150)
plt.show()

sum, count = 0, 0
for i, item in enumerate(np_result_5[:, 1]):
    np_result_5[i, 1] = item + 10
    if np_result_5[i, 1] > 0 and np_result_5[i, 0] > 0:
        sum += (np_result_5[i, 1] - np_result_5[i, 0]) ** 2
        count += 1
print('Model 5 RMSE Score : {}'.format(np.sqrt(sum / count)))

plt.figure(figsize=(6,6))
plt.plot(x,x, color='black')
plt.scatter(np_flowering[:,1], np_flowering[:,2], color='black', alpha=0.75, s=15, label='Budding Inconsidered Baseline (GDD)')
plt.scatter(np_result_5[:,0], np_result_5[:,1], color='red', alpha=0.75, s=15, label='Budding Considered Model 4')
plt.xlabel('True')
plt.ylabel('Predict')
plt.title('ML Model Predicted - True FFD (Seoul, 108)')
plt.legend()
plt.xlim(60,150)
plt.ylim(60,150)
plt.show()