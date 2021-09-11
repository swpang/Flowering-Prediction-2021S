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


CRITICAL_TEMP = 3.0
path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\DATA\\'
results_path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\RESULTS\\'
model_path = 'C:\\Users\\DELL\\OneDrive - SNU\\Documents\\2021-S\\Research\\Data\\MODELS\\'

df_stations = pd.read_csv(path + 'stations_info.csv', header=0, index_col=None)
df_flowering = pd.read_csv(path + 'season_f.csv', header=None, index_col=None, names=['stnId', 'lat', 'lon', 'year', 'date'])
df_budding = pd.read_csv(path + 'season_g.csv', header=None, index_col=None, names=['stnId', 'lat', 'lon', 'year', 'date'])
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

dict_weather = {}
for stnId in df_stations['stnId']:
    if os.path.exists(path + 'weatherData\\{}_data.csv'.format(stnId)):
        df_temp = pd.read_csv(path + 'weatherData\\{}_data.csv'.format(stnId), header=0, index_col=0)
        dict_weather[str(stnId)] = df_temp

########### Process Data ############

print('x, y of 5 : {}, {}'.format(np_x_5.shape, np_y_5.shape))

class JobDataset(Dataset):
    def __init__(self, np_x_data, np_y_data):
        self.ids = np_x_data[:,-1]
        self.truevals = np_y_data
        np_temp = np.delete(np_x_data, -1, 1)
        self.data = np_temp

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.data[idx], self.truevals[idx]

    def get_ids(self):
        return self.ids

    def get_inputsize(self):
        return self.data.shape[1]

np_ids_1 = list(np_x_1[:,0])
for i, id in enumerate(np_ids_1):
    ids = id.split('_')
    ids = list(map(float, ids))
    np_x_1[i, 0] = ids[0] * 1000 + ids[1]

np_ids_2 = list(np_x_2[:,0])
for i, id in enumerate(np_ids_2):
    ids = id.split('_')
    ids = list(map(float, ids))
    np_x_2[i, 0] = ids[0] * 1000 + ids[1]

np_ids_4 = list(np_x_4[:,0])
for i, id in enumerate(np_ids_4):
    ids = id.split('_')
    ids = list(map(float, ids))
    np_x_4[i, 0] = ids[0] * 1000 + ids[1]

np_ids_5 = list(np_x_5[:,0])
for i, id in enumerate(np_ids_5):
    ids = id.split('_')
    ids = list(map(float, ids))
    np_x_5[i, 0] = ids[0] * 1000 + ids[1]

np_ids_1_seoul = list(np_x_1_seoul[:,0])
for i, id in enumerate(np_ids_1_seoul):
    ids = id.split('_')
    ids = list(map(float, ids))
    np_x_1_seoul[i, 0] = ids[0] * 1000 + ids[1]

np_ids_2_seoul = list(np_x_2_seoul[:,0])
for i, id in enumerate(np_ids_2_seoul):
    ids = id.split('_')
    ids = list(map(float, ids))
    np_x_2_seoul[i, 0] = ids[0] * 1000 + ids[1]

np_ids_4_seoul = list(np_x_4_seoul[:,0])
for i, id in enumerate(np_ids_4_seoul):
    ids = id.split('_')
    ids = list(map(float, ids))
    np_x_4_seoul[i, 0] = ids[0] * 1000 + ids[1]

np_ids_5_seoul = list(np_x_5_seoul[:,0])
for i, id in enumerate(np_ids_5_seoul):
    ids = id.split('_')
    ids = list(map(float, ids))
    np_x_5_seoul[i, 0] = ids[0] * 1000 + ids[1]

transformer_1 = ColumnTransformer(transformers=[('num', RobustScaler(), [1, 2])], remainder='passthrough')
transformer_2 = ColumnTransformer(transformers=[('num', RobustScaler(), [1, 2, 3])], remainder='passthrough')

fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection = '3d')

x = np_x_5[:,1]
y = np_x_5[:,2]
z = np_x_5[:,3]
c = x + y + z

ax.scatter(x, y, z, s=15, alpha=0.3, c=c)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('Input Features Non-Scaled')
plt.show()

np_x_1 = transformer_1.fit_transform(np_x_1).astype(np.float64)
np_x_2 = transformer_1.fit_transform(np_x_2).astype(np.float64)
np_x_4 = transformer_2.fit_transform(np_x_4).astype(np.float64)
np_x_5 = transformer_2.fit_transform(np_x_5).astype(np.float64)
np_x_1_seoul = transformer_1.fit_transform(np_x_1_seoul).astype(np.float64)
np_x_2_seoul = transformer_1.fit_transform(np_x_2_seoul).astype(np.float64)
np_x_4_seoul = transformer_2.fit_transform(np_x_4_seoul).astype(np.float64)
np_x_5_seoul = transformer_2.fit_transform(np_x_5_seoul).astype(np.float64)

fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection = '3d')

x = np_x_5[:,0]
y = np_x_5[:,1]
z = np_x_5[:,2]
c = x + y + z

ax.scatter(x, y, z, s=15, alpha=0.3, c=c)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('Input Features Scaled')
plt.show()

np_x_train_1, np_x_test_1, np_y_train_1, np_y_test_1 = train_test_split(np_x_1, np_y_1, test_size=0.2, random_state=seed)
np_x_train_2, np_x_test_2, np_y_train_2, np_y_test_2 = train_test_split(np_x_2, np_y_2, test_size=0.2, random_state=seed)
np_x_train_4, np_x_test_4, np_y_train_4, np_y_test_4 = train_test_split(np_x_4, np_y_4, test_size=0.2, random_state=seed)
np_x_train_5, np_x_test_5, np_y_train_5, np_y_test_5 = train_test_split(np_x_5, np_y_5, test_size=0.2, random_state=seed)

########### Create Machine Learning Model ############

################################################################################################

BATCH_SIZE = 512
LEARNING_RATE = 0.0015
SHUFFLE = False
MAX_EPOCHS = 1000

train_params = {'batch_size': BATCH_SIZE, 'shuffle': SHUFFLE}
test_params = {'batch_size': BATCH_SIZE, 'shuffle': SHUFFLE}

################################################################################################

class NNModel(nn.Module):
    def __init__(self, INPUT_SIZE, WEIGHT_COUNTS, OUTPUT_SIZE):
        super(NNModel, self).__init__()
        self.INPUT_SIZE = INPUT_SIZE
        self.WEIGHT_COUNTS = WEIGHT_COUNTS
        self.OUTPUT_SIZE = OUTPUT_SIZE
        self.input = nn.Linear(self.INPUT_SIZE, self.WEIGHT_COUNTS[0], bias=True)
        self.fc1 = nn.Linear(self.WEIGHT_COUNTS[0], self.WEIGHT_COUNTS[1], bias=True)
        self.fc2 = nn.Linear(self.WEIGHT_COUNTS[1], self.WEIGHT_COUNTS[2], bias=True)
        self.fc3 = nn.Linear(self.WEIGHT_COUNTS[2], self.WEIGHT_COUNTS[3], bias=True)
        #self.fc4 = nn.Linear(self.WEIGHT_COUNTS[3], self.WEIGHT_COUNTS[4], bias=True)
        self.output = nn.Linear(WEIGHT_COUNTS[3], self.OUTPUT_SIZE, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        #nn.init.xavier_uniform_(self.fc4.weight)
        #nn.init.zeros_(self.fc4.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.relu(self.fc3(x))
        #x = self.relu(self.fc4(x))
        return self.output(x).view(-1,1)

#########################################################################################################################

if __name__ == '__main__':
    train_dataset_1 = JobDataset(np_x_train_1, np_y_train_1)
    train_dataloader_1 = DataLoader(train_dataset_1, **train_params)
    train_dataset_2 = JobDataset(np_x_train_2, np_y_train_2)
    train_dataloader_2 = DataLoader(train_dataset_2, **train_params)
    train_dataset_4 = JobDataset(np_x_train_4, np_y_train_4)
    train_dataloader_4 = DataLoader(train_dataset_4, **train_params)
    train_dataset_5 = JobDataset(np_x_train_5, np_y_train_5)
    train_dataloader_5 = DataLoader(train_dataset_5, **train_params)

    test_dataset_1 = JobDataset(np_x_test_1, np_y_test_1)
    test_dataloader_1 = DataLoader(test_dataset_1, **test_params)
    test_dataset_2 = JobDataset(np_x_test_2, np_y_test_2)
    test_dataloader_2 = DataLoader(test_dataset_2, **test_params)
    test_dataset_4 = JobDataset(np_x_test_4, np_y_test_4)
    test_dataloader_4 = DataLoader(test_dataset_4, **test_params)
    test_dataset_5 = JobDataset(np_x_test_5, np_y_test_5)
    test_dataloader_5 = DataLoader(test_dataset_5, **test_params)

    test_dataset_1_seoul = JobDataset(np_x_1_seoul, np_y_1_seoul)
    test_dataloader_1_seoul = DataLoader(test_dataset_1_seoul, **test_params)
    test_dataset_2_seoul = JobDataset(np_x_2_seoul, np_y_2_seoul)
    test_dataloader_2_seoul = DataLoader(test_dataset_2_seoul, **test_params)
    test_dataset_4_seoul = JobDataset(np_x_4_seoul, np_y_4_seoul)
    test_dataloader_4_seoul = DataLoader(test_dataset_4_seoul, **test_params)
    test_dataset_5_seoul = JobDataset(np_x_5_seoul, np_y_5_seoul)
    test_dataloader_5_seoul = DataLoader(test_dataset_5_seoul, **test_params)

    criterion_1 = nn.MSELoss().to(device)
    criterion_2 = nn.MSELoss().to(device)
    criterion_4 = nn.MSELoss().to(device)
    criterion_5 = nn.MSELoss().to(device)

##########################################################################################################

    WEIGHT_COUNTS = [32, 64, 64, 32]
    OUTPUT_SIZE = 1

##########################################################################################################

    model_1 = NNModel(train_dataset_1.get_inputsize(), WEIGHT_COUNTS, OUTPUT_SIZE).to(device)
    model_2 = NNModel(train_dataset_2.get_inputsize(), WEIGHT_COUNTS, OUTPUT_SIZE).to(device)
    model_4 = NNModel(train_dataset_4.get_inputsize(), WEIGHT_COUNTS, OUTPUT_SIZE).to(device)
    model_5 = NNModel(train_dataset_5.get_inputsize(), WEIGHT_COUNTS, OUTPUT_SIZE).to(device)

    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=LEARNING_RATE)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=LEARNING_RATE)
    optimizer_4 = torch.optim.Adam(model_4.parameters(), lr=LEARNING_RATE)
    optimizer_5 = torch.optim.Adam(model_5.parameters(), lr=LEARNING_RATE)

    filename_1 = '_type1'
    for i in WEIGHT_COUNTS:
        filename_1 += '_{}'.format(int(i))
    filename_1 += '_lr{}'.format(LEARNING_RATE)
    filename_1 += '_ep{}'.format(MAX_EPOCHS)
    filename_1 += '_bs{}'.format(BATCH_SIZE)

    idx_1 = 0
    while os.path.exists(results_path + 'test' + filename_1 + '_ver{}.csv'.format(idx_1)):
        idx_1 += 1

    filename_2 = '_type2'
    for i in WEIGHT_COUNTS:
        filename_1 += '_{}'.format(int(i))
    filename_2 += '_lr{}'.format(LEARNING_RATE)
    filename_2 += '_ep{}'.format(MAX_EPOCHS)
    filename_2 += '_bs{}'.format(BATCH_SIZE)

    idx_2 = 0
    while os.path.exists(results_path + 'test' + filename_2 + '_ver{}.csv'.format(idx_2)):
        idx_2 += 1

    filename_5 = '_type5'
    for i in WEIGHT_COUNTS:
        filename_5 += '_{}'.format(int(i))
    filename_5 += '_lr{}'.format(LEARNING_RATE)
    filename_5 += '_ep{}'.format(MAX_EPOCHS)
    filename_5 += '_bs{}'.format(BATCH_SIZE)

    idx_5 = 0
    while os.path.exists(results_path + 'test' + filename_5 + '_ver{}.csv'.format(idx_5)):
        idx_5 += 1

    filename_4 = '_type4'
    for i in WEIGHT_COUNTS:
        filename_4 += '_{}'.format(int(i))
    filename_4 += '_lr{}'.format(LEARNING_RATE)
    filename_4 += '_ep{}'.format(MAX_EPOCHS)
    filename_4 += '_bs{}'.format(BATCH_SIZE)

    idx_4 = 0
    while os.path.exists(results_path + 'test' + filename_4 + '_ver{}.csv'.format(idx_4)):
        idx_4 += 1

    def train(max_epochs, model, optimizer, criterion, train_dataloader, val_dataloader):
        best_loss = None
        counter = 0
        for epoch in range(max_epochs):
            model.train()
            tr_loss, val_loss = 0, 0
            for (x, y) in train_dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y).to(device)
                loss.backward()
                optimizer.step()
                tr_loss += loss.cpu().item()

                v_loss = 0
                with torch.no_grad():
                    for (x_v, y_v) in val_dataloader:
                        x_v, y_v = x_v.to(device), y_v.to(device)
                        val_pred = model(x_v)
                        v_loss += criterion(val_pred, y_v)
                    val_loss = v_loss

            if (epoch + 1) % 10 == 0:
                print('==== Epoch : {:03d}\t Training Loss : {:.4f}\t Validation Loss : {:.4f} ===='.format(
                    epoch + 1, tr_loss / len(train_dataloader), val_loss / len(val_dataloader)))
                
            if best_loss == None:
                best_loss = val_loss
            elif best_loss - val_loss > 0:
                best_loss = val_loss
            elif best_loss - val_loss < 0:
                counter += 1
                if counter > 50:
                    print('Early Stopping Training')
                    break

    def test(model, test_dataloader):
        model.eval()
        with torch.no_grad():
            preds = []
            for (x, y) in test_dataloader:
                x, y = x.to(device), y.to(device)
                pred = model(x).cpu().numpy().reshape(-1,1)
                for i in list(pred):
                    preds.append(i)
        return preds

###################################################################################################################################
################# Train ####################


    train(MAX_EPOCHS, model_1, optimizer_1, criterion_1, train_dataloader_1, test_dataloader_1)
    print('Training Ended')
    torch.save(model_1.state_dict(), model_path + filename_1 + '_ver{}.pth'.format(idx_1))
    train(MAX_EPOCHS, model_2, optimizer_2, criterion_2, train_dataloader_2, test_dataloader_2)
    print('Training Ended')
    torch.save(model_2.state_dict(), model_path + filename_2 + '_ver{}.pth'.format(idx_2))
    train(MAX_EPOCHS, model_5, optimizer_5, criterion_5, train_dataloader_5, test_dataloader_5)
    print('Training Ended')
    torch.save(model_5.state_dict(), model_path + filename_5 + '_ver{}.pth'.format(idx_5))
    train(MAX_EPOCHS, model_4, optimizer_4, criterion_4, train_dataloader_4, test_dataloader_4)
    print('Training Ended')
    torch.save(model_4.state_dict(), model_path + filename_4 + '_ver{}.pth'.format(idx_4))

##################################################################################################################################

    np_x_train_1_temp = np.hstack((np.array(train_dataset_1.get_ids()).reshape(-1,1), np_x_train_1))
    df_x_train_dataset_1 = pd.DataFrame(np_x_train_1_temp)
    np_x_train_2_temp = np.hstack((np.array(train_dataset_2.get_ids()).reshape(-1,1), np_x_train_2))
    df_x_train_dataset_2 = pd.DataFrame(np_x_train_2_temp)
    np_x_train_5_temp = np.hstack((np.array(train_dataset_5.get_ids()).reshape(-1,1), np_x_train_5))
    df_x_train_dataset_5 = pd.DataFrame(np_x_train_5_temp)
    np_x_train_4_temp = np.hstack((np.array(train_dataset_4.get_ids()).reshape(-1,1), np_x_train_4))
    df_x_train_dataset_4 = pd.DataFrame(np_x_train_4_temp)
    np_y_train_1_temp = np.hstack((np.array(train_dataset_1.get_ids()).reshape(-1,1), np_y_train_1))
    df_y_train_dataset_1 = pd.DataFrame(np_y_train_1_temp)
    np_y_train_2_temp = np.hstack((np.array(train_dataset_2.get_ids()).reshape(-1,1), np_y_train_2))
    df_y_train_dataset_2 = pd.DataFrame(np_y_train_2_temp)
    np_y_train_5_temp = np.hstack((np.array(train_dataset_5.get_ids()).reshape(-1,1), np_y_train_5))
    df_y_train_dataset_5 = pd.DataFrame(np_y_train_5_temp)
    np_y_train_4_temp = np.hstack((np.array(train_dataset_4.get_ids()).reshape(-1,1), np_y_train_4))
    df_y_train_dataset_4 = pd.DataFrame(np_y_train_4_temp)
    np_x_test_1_temp = np.hstack((np.array(test_dataset_1.get_ids()).reshape(-1,1), np_x_test_1))
    df_x_test_dataset_1 = pd.DataFrame(np_x_test_1_temp)
    np_x_test_2_temp = np.hstack((np.array(test_dataset_2.get_ids()).reshape(-1,1), np_x_test_2))
    df_x_test_dataset_2 = pd.DataFrame(np_x_test_2_temp)
    np_x_test_5_temp = np.hstack((np.array(test_dataset_5.get_ids()).reshape(-1,1), np_x_test_5))
    df_x_test_dataset_5 = pd.DataFrame(np_x_test_5_temp)
    np_x_test_4_temp = np.hstack((np.array(test_dataset_4.get_ids()).reshape(-1,1), np_x_test_4))
    df_x_test_dataset_4 = pd.DataFrame(np_x_test_4_temp)
    np_y_test_1_temp = np.hstack((np.array(test_dataset_1.get_ids()).reshape(-1,1), np_y_test_1))
    df_y_test_dataset_1 = pd.DataFrame(np_y_test_1_temp)
    np_y_test_2_temp = np.hstack((np.array(test_dataset_2.get_ids()).reshape(-1,1), np_y_test_2))
    df_y_test_dataset_2 = pd.DataFrame(np_y_test_2_temp)
    np_y_test_5_temp = np.hstack((np.array(test_dataset_5.get_ids()).reshape(-1,1), np_y_test_5))
    df_y_test_dataset_5 = pd.DataFrame(np_y_test_5_temp)
    np_y_test_4_temp = np.hstack((np.array(test_dataset_4.get_ids()).reshape(-1,1), np_y_test_4))
    df_y_test_dataset_4 = pd.DataFrame(np_y_test_4_temp)

    df_x_train_dataset_1.to_csv(path + 'train_x' + filename_1 + '_ver{}.csv'.format(idx_1))
    df_x_train_dataset_2.to_csv(path + 'train_x' + filename_2 + '_ver{}.csv'.format(idx_2))
    df_x_train_dataset_5.to_csv(path + 'train_x' + filename_5 + '_ver{}.csv'.format(idx_5))
    df_x_train_dataset_4.to_csv(path + 'train_x' + filename_4 + '_ver{}.csv'.format(idx_4))
    df_y_train_dataset_1.to_csv(path + 'train_y' + filename_1 + '_ver{}.csv'.format(idx_1))
    df_y_train_dataset_2.to_csv(path + 'train_y' + filename_2 + '_ver{}.csv'.format(idx_2))
    df_y_train_dataset_5.to_csv(path + 'train_y' + filename_5 + '_ver{}.csv'.format(idx_5))
    df_y_train_dataset_4.to_csv(path + 'train_y' + filename_4 + '_ver{}.csv'.format(idx_4))
    df_x_test_dataset_1.to_csv(path + 'test_x' + filename_1 + '_ver{}.csv'.format(idx_1))
    df_x_test_dataset_2.to_csv(path + 'test_x' + filename_2 + '_ver{}.csv'.format(idx_2))
    df_x_test_dataset_5.to_csv(path + 'test_x' + filename_5 + '_ver{}.csv'.format(idx_5))
    df_x_test_dataset_4.to_csv(path + 'test_x' + filename_4 + '_ver{}.csv'.format(idx_4))
    df_y_test_dataset_1.to_csv(path + 'test_y' + filename_1 + '_ver{}.csv'.format(idx_1))
    df_y_test_dataset_2.to_csv(path + 'test_y' + filename_2 + '_ver{}.csv'.format(idx_2))
    df_y_test_dataset_5.to_csv(path + 'test_y' + filename_5 + '_ver{}.csv'.format(idx_5))
    df_y_test_dataset_4.to_csv(path + 'test_y' + filename_4 + '_ver{}.csv'.format(idx_4))

    ###

    test_result_1 = test(model_1, test_dataloader_1)
    train_result_1 = test(model_1, train_dataloader_1)

    train_rmse_1 = mean_squared_error(np_y_train_1, train_result_1)
    print('Model 1 Train RMSE Loss : {}'.format(train_rmse_1))
    test_rmse_1 = mean_squared_error(np_y_test_1, test_result_1)
    print('Model 1 Test RMSE Loss : {}'.format(test_rmse_1))

    df_test_result = pd.DataFrame(np.array(test_result_1).reshape(-1,1), dtype=np.float64)
    df_train_result = pd.DataFrame(np.array(train_result_1).reshape(-1,1), dtype=np.float64)

    df_test_result.to_csv(results_path + 'test' + filename_1 + '_ver{}.csv'.format(idx_1))
    df_train_result.to_csv(results_path + 'train' + filename_1 + '_ver{}.csv'.format(idx_1))

    ###

    test_result_2 = test(model_2, test_dataloader_2)
    train_result_2 = test(model_2, train_dataloader_2)

    train_rmse_2 = mean_squared_error(np_y_train_2, train_result_2)
    print('Model 2 Train RMSE Loss : {}'.format(train_rmse_2))
    test_rmse_2 = mean_squared_error(np_y_test_2, test_result_2)
    print('Model 2 Test RMSE Loss : {}'.format(test_rmse_2))

    df_test_result = pd.DataFrame(np.array(test_result_2).reshape(-1,1), dtype=np.float64)
    df_train_result = pd.DataFrame(np.array(train_result_2).reshape(-1,1), dtype=np.float64)

    df_test_result.to_csv(results_path + 'test' + filename_2 + '_ver{}.csv'.format(idx_2))
    df_train_result.to_csv(results_path + 'train' + filename_2 + '_ver{}.csv'.format(idx_2))

    ###

    test_result_5 = test(model_5, test_dataloader_5)
    train_result_5 = test(model_5, train_dataloader_5)

    train_rmse_5 = mean_squared_error(np_y_train_5, train_result_5)
    print('Model 5 Train RMSE Loss : {}'.format(train_rmse_5))
    test_rmse_5 = mean_squared_error(np_y_test_5, test_result_5)
    print('Model 5 Test RMSE Loss : {}'.format(test_rmse_5))

    df_test_result = pd.DataFrame(np.array(test_result_5).reshape(-1,1), dtype=np.float64)
    df_train_result = pd.DataFrame(np.array(train_result_5).reshape(-1,1), dtype=np.float64)

    df_test_result.to_csv(results_path + 'test' + filename_5 + '_ver{}.csv'.format(idx_5))
    df_train_result.to_csv(results_path + 'train' + filename_5 + '_ver{}.csv'.format(idx_5))

    ###

    test_result_4 = test(model_4, test_dataloader_4)
    train_result_4 = test(model_4, train_dataloader_4)

    train_rmse_4 = mean_squared_error(np_y_train_4, train_result_4)
    print('Model 4 Train RMSE Loss : {}'.format(train_rmse_4))
    test_rmse_4 = mean_squared_error(np_y_test_4, test_result_4)
    print('Model 4 Test RMSE Loss : {}'.format(test_rmse_4))

    df_test_result = pd.DataFrame(np.array(test_result_4).reshape(-1,1), dtype=np.float64)
    df_train_result = pd.DataFrame(np.array(train_result_4).reshape(-1,1), dtype=np.float64)

    df_test_result.to_csv(results_path + 'test' + filename_4 + '_ver{}.csv'.format(idx_4))
    df_train_result.to_csv(results_path + 'train' + filename_4 + '_ver{}.csv'.format(idx_4))

    ###

    print('___________________________________________________________________________________________')

    test_result_1_seoul = test(model_1, test_dataloader_1_seoul)
    test_result_2_seoul = test(model_2, test_dataloader_2_seoul)
    test_result_4_seoul = test(model_4, test_dataloader_4_seoul)
    test_result_5_seoul = test(model_5, test_dataloader_5_seoul)

    ids_1_seoul = test_dataset_1_seoul.get_ids()
    ids_2_seoul = test_dataset_2_seoul.get_ids()
    ids_4_seoul = test_dataset_4_seoul.get_ids()
    ids_5_seoul = test_dataset_5_seoul.get_ids()

    df_test_result_1_seoul = pd.DataFrame(np.hstack((np.array(ids_1_seoul).reshape(-1, 1), np.array(test_result_1_seoul).reshape(-1, 1))))
    df_test_result_2_seoul = pd.DataFrame(np.hstack((np.array(ids_2_seoul).reshape(-1, 1), np.array(test_result_2_seoul).reshape(-1, 1))))
    df_test_result_4_seoul = pd.DataFrame(np.hstack((np.array(ids_4_seoul).reshape(-1, 1), np.array(test_result_4_seoul).reshape(-1, 1))))
    df_test_result_5_seoul = pd.DataFrame(np.hstack((np.array(ids_5_seoul).reshape(-1, 1), np.array(test_result_5_seoul).reshape(-1, 1))))

    df_test_result_1_seoul.to_csv(results_path + '1_seoul_results.csv')
    df_test_result_2_seoul.to_csv(results_path + '2_seoul_results.csv')
    df_test_result_4_seoul.to_csv(results_path + '4_seoul_results.csv')
    df_test_result_5_seoul.to_csv(results_path + '5_seoul_results.csv')

#####################################################################################################################################################################

CRITICAL_TEMP_2 = 5

def calc_gdd(stnId, year, start_date):
    df_temp = dict_weather[str(stnId)]

    sumT = 0
    fdate = np.nan

    for i, rows in df_flowering.iterrows():
        if int(rows['stnId']) == stnId:
            if int(rows['year']) == year:
                fdate = int(rows['date'])
    for i, rows in df_temp.iterrows():
        if int(rows['year']) == year:
            if int(rows['jday']) < fdate and int(rows['jday']) >= start_date:
                if float(rows['avgTa']) >= 0 or float(rows['avgTa']) < 0:
                    sumT += max(0, float(rows['avgTa']) - CRITICAL_TEMP_2)
        elif sumT > 0:
            break
    return sumT, fdate

def calc_crit(stnId, year, avgT, start_date):
    
    df_temp = dict_weather[str(stnId)]

    sumT = 0

    for i, rows in df_temp.iterrows():
        if int(rows['year']) == year:
            if int(rows['jday']) >= start_date:
                if sumT <= avgT:
                    jdate = int(rows['jday'])
                    sumT += max(0, float(rows['avgTa']) - CRITICAL_TEMP_2)
                return jdate

#######################################################################################################################

sumT = []
fdates = []

for i, result in enumerate(test_result_1_seoul):
    id = test_dataset_1_seoul.get_ids()[i]
    year = int(id // 1000)
    stnId = int(id % 1000)
    t, date = calc_gdd(stnId, year, result[0])
    sumT.append(t)
    fdates.append(date)
sum = 0.0
count = 0.0

for t in sumT:
    if t > 0:
        sum += t
        count += 1
avgT = sum / count
crit = []

for i, result in enumerate(test_result_1_seoul):
    id = test_dataset_1_seoul.get_ids()[i]
    year = int(id // 1000)
    stnId = str(int(id % 1000))
    crit.append(calc_crit(stnId, year, avgT, result[0]))
for i, value in enumerate(crit):
    if value != None:
        crit[i] = value + 10

df_model1_results = pd.DataFrame(np.hstack((np.array(fdates).reshape(-1, 1), np.array(crit).reshape(-1, 1))), columns=['fdates', 'criticaldates'])
df_model1_results.to_csv(results_path + 'model1_results.csv')

x = np.linspace(0,200)

print('Model 1')

plt.figure(figsize=(10,10))
plt.scatter(fdates, crit, color='black', s=15)
plt.plot(x,x, color='black')
plt.xlabel('True')
plt.ylabel('Predict')
plt.title('ML Model Predicted - True FFD (Seoul, 108)')
plt.xlim(60,150)
plt.ylim(60,150)
plt.show()

#######################################################################################################################

sumT = []
fdates = []

for i, result in enumerate(test_result_2_seoul):
    id = test_dataset_2_seoul.get_ids()[i]
    year = int(id // 1000)
    stnId = int(id % 1000)
    t, date = calc_gdd(stnId, year, result[0])
    sumT.append(t)
    fdates.append(date)
sum = 0.0
count = 0.0

for t in sumT:
    if t > 0:
        sum += t
        count += 1
avgT = sum / count
crit = []

for i, result in enumerate(test_result_2_seoul):
    id = test_dataset_2_seoul.get_ids()[i]
    year = int(id // 1000)
    stnId = str(int(id % 1000))
    crit.append(calc_crit(stnId, year, avgT, result[0]))
for i, value in enumerate(crit):
    if value != None:
        crit[i] = value + 10

df_model2_results = pd.DataFrame(np.hstack((np.array(fdates).reshape(-1, 1), np.array(crit).reshape(-1, 1))), columns=['fdates', 'criticaldates'])
df_model2_results.to_csv(results_path + 'model2_results.csv')

x = np.linspace(0,200)

print('Model 2')

plt.figure(figsize=(10,10))
plt.scatter(fdates, crit, color='black', s=15)
plt.plot(x,x, color='black')
plt.xlabel('True')
plt.ylabel('Predict')
plt.title('ML Model Predicted - True FFD (Seoul, 108)')
plt.xlim(60,150)
plt.ylim(60,150)
plt.show()

#######################################################################################################################

sumT = []
fdates = []

for i, result in enumerate(test_result_4_seoul):
    id = test_dataset_4_seoul.get_ids()[i]
    year = int(id // 1000)
    stnId = int(id % 1000)
    t, date = calc_gdd(stnId, year, result[0])
    sumT.append(t)
    fdates.append(date)
sum = 0.0
count = 0.0

for t in sumT:
    if t > 0:
        sum += t
        count += 1
avgT = sum / count
crit = []

for i, result in enumerate(test_result_4_seoul):
    id = test_dataset_4_seoul.get_ids()[i]
    year = int(id // 1000)
    stnId = str(int(id % 1000))
    crit.append(calc_crit(stnId, year, avgT, result[0]))
for i, value in enumerate(crit):
    if value != None:
        crit[i] = value + 10

df_model4_results = pd.DataFrame(np.hstack((np.array(fdates).reshape(-1, 1), np.array(crit).reshape(-1, 1))), columns=['fdates', 'criticaldates'])
df_model4_results.to_csv(results_path + 'model4_results.csv')

x = np.linspace(0,200)

print('Model 4')

plt.figure(figsize=(10,10))
plt.scatter(fdates, crit, color='black', s=15)
plt.plot(x,x, color='black')
plt.xlabel('True')
plt.ylabel('Predict')
plt.title('ML Model Predicted - True FFD (Seoul, 108)')
plt.xlim(60,150)
plt.ylim(60,150)
plt.show()

#######################################################################################################################

sumT = []
fdates = []

for i, result in enumerate(test_result_5_seoul):
    id = test_dataset_5_seoul.get_ids()[i]
    year = int(id // 1000)
    stnId = int(id % 1000)
    t, date = calc_gdd(stnId, year, result[0])
    sumT.append(t)
    fdates.append(date)
sum = 0.0
count = 0.0

for t in sumT:
    if t > 0:
        sum += t
        count += 1
avgT = sum / count
crit = []

for i, result in enumerate(test_result_5_seoul):
    id = test_dataset_5_seoul.get_ids()[i]
    year = int(id // 1000)
    stnId = str(int(id % 1000))
    crit.append(calc_crit(stnId, year, avgT, result[0]))
for i, value in enumerate(crit):
    if value != None:
        crit[i] = value + 10

df_model5_results = pd.DataFrame(np.hstack((np.array(fdates).reshape(-1, 1), np.array(crit).reshape(-1, 1))), columns=['fdates', 'criticaldates'])
df_model5_results.to_csv(results_path + 'model5_results.csv')

x = np.linspace(0,200)

print('Model 5')

plt.figure(figsize=(10,10))
plt.scatter(fdates, crit, color='black', s=15)
plt.plot(x,x, color='black')
plt.xlabel('True')
plt.ylabel('Predict')
plt.title('ML Model Predicted - True FFD (Seoul, 108)')
plt.xlim(60,150)
plt.ylim(60,150)
plt.show()