import numpy as np
import pandas as pd
from pandas import DataFrame, concat
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score,mean_absolute_error
import torch
import logging
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import psutil
import torch

# Fucntion to return model metrics
def metrics_pytorch(model, y_test=0, y_pred=0):
    metrics = {}
        
    metrics['mse'] = mean_squared_error(y_test, y_pred)
    metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
    metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred)
    metrics['mae'] = round(mean_absolute_error(y_test, y_pred), 2)
    metrics['smape'] = round(symmetric_mean_absolute_percentage_error(y_test, y_pred), 2)
    metrics['r2'] = r2_score(y_test, y_pred)
    metrics['Model Size (MB)'] = print_size_of_model(model,"int8")
    metrics['Memory Usage (MB)'] = get_memory_usage()
    return metrics

def prepare_data(train_df, test_df, look_back=6, batch_size=64):
    # Initialize data_components
    data_components = {}
    logging.info(f"Preparing dataset for PyTorch model")
    
    scaled_data_train, scaled_data_test, scaler_obj = scale_data(train_df, test_df, scaler="MinMax")
    supervised_train_data = ts_supervised_structure(scaled_data_train, n_in=look_back, n_out=1)
    supervised_test_data = ts_supervised_structure(scaled_data_test, n_in=look_back, n_out=1)
    
    print("supervised_train_data: ", supervised_train_data.shape)

    # Extract inputs (lagged features) and outputs (targets)
    X_train_np = supervised_train_data.iloc[:, :-2].values  # First 12 cols
    y_train_np = supervised_train_data.iloc[:, -2:].values  # Last 2 cols
    X_test_np = supervised_test_data.iloc[:, :-2].values
    y_test_np = supervised_test_data.iloc[:, -2:].values

    print("X_train_np: ", X_train_np.shape)

    # Reshape correctly to (batch_size, seq_len=look_back, input_size=2)
    X_train = X_train_np.reshape(X_train_np.shape[0], look_back, 2)
    X_test = X_test_np.reshape(X_test_np.shape[0], look_back, 2)

    print("X_train (reshaped): ", X_train.shape)  # Should be (1585, 6, 2)

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train_np).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test_np).float()

    # Create PyTorch datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # Store components
    data_components['X_train'] = X_train
    data_components['X_test'] = X_test
    data_components['y_train'] = y_train
    data_components['y_test'] = y_test
    data_components['scaler_obj'] = scaler_obj
    data_components['train_dataset'] = train_dataset
    data_components['test_dataset'] = test_dataset
    data_components['batch_size'] = batch_size

    return data_components

# Function to prepare data
def prepare_data_old(train_df, test_df, look_back=6,  batch_size=64):
    # Initialize data_components in general for either model
    data_components = {}
    logging.info(f"Preparing dataset for PyTorch model")
    scaled_data_train, scaled_data_test, scaler_obj = scale_data(train_df, test_df, scaler="MinMax")
    supervised_train_data = ts_supervised_structure(scaled_data_train, n_in=look_back, n_out=1)
    supervised_test_data = ts_supervised_structure(scaled_data_test, n_in=look_back, n_out=1)
    print("supervised_train_data: ", supervised_train_data.shape)
    X_train_np = supervised_train_data.iloc[:, :-2].values  # All features except the last two columns
    y_train_np = supervised_train_data.iloc[:, -2:].values  # Only the last two columns
    X_test_np = supervised_test_data.iloc[:, :-2].values
    y_test_np = supervised_test_data.iloc[:, -2:].values
    print("X_train_np: ", X_train_np.shape)

    # Reshape to 3D tensors (batch_size, seq_len, input_size)
    X_train = np.expand_dims(X_train_np, axis=2)
    y_train = np.expand_dims(y_train_np, axis=2)
    X_test = np.expand_dims(X_test_np, axis=2)
    y_test = np.expand_dims(y_test_np, axis=2)
    print("X_train: ", X_train.shape)

    # Convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Create PyTorch datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    data_components['X_train']=X_train
    data_components['X_test']=X_test
    data_components['y_train']=y_train
    data_components['y_test']=y_test
    data_components['scaler_obj']=scaler_obj
    # data_components['model_parameters'] = {"pytorch_model_parameters": {"hidden_size": 64, "num_epochs": 100}}
    data_components['train_dataset'] = train_dataset
    data_components['test_dataset']  = test_dataset
    # data_components['device']  = device
    data_components['batch_size']  = batch_size

    return data_components

def create_dataloaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Perform data cleaning
def data_clean(df):
    # Keep only the relevant columns in the DataFrame
    df = df[['Date', 'CPU Core 1 Usage (%)', 'CPU Core 2 Usage (%)']]

    # Rename the columns for better readability
    cols = {'Date': 'date', 'CPU Core 1 Usage (%)': 'TARGET', 'CPU Core 2 Usage (%)': 'RAM'}
    df = df.rename(columns=cols, inplace=False)

    return df

def ts_supervised_structure(data, n_in=1, n_out=1, dropnan=True, autoregressive=True):
        no_autoregressive = not(autoregressive)
        if no_autoregressive:
            n_in = n_in - 1

        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            if no_autoregressive:
                cols.append(df.shift(i).iloc[:,:-1])
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars-1)]
            else:
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
# Perform train, test and split for time-series dataset
def data_simple_split(data_df,test_size=0.2):
    size = int(len(data_df) * 0.8)              
    train_df, test_df = data_df[0:size], data_df[size:len(data_df)] 
    train_df = train_df.reset_index(drop=True)  
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df

# Calculate SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

# Performs normalisation on complete dataset 
def scale_data(train_df, test_df, scaler):
    if scaler == "MinMax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        scaler = StandardScaler()
    
    scaled_data_train = scaler.fit_transform(train_df)
    scaled_data_test = scaler.transform(test_df)

    return scaled_data_train, scaled_data_test, scaler


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size/1e6# kb (1e3) to MB (1e6)

def bytes_to_mb(memory_bytes):
    """
    Takes the memory usage in bytes as input and returns the memory usage converted to megabytes (MB).
    """
    return memory_bytes / (1024 * 1024)
    
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return bytes_to_mb(memory_info.rss)  # Returns the memory usage in bytes


# Performs normalisation separately on input and output
def normalize_data(X_train, X_test, y_train, y_test):
    # Convert input data to pandas Series if they are not
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=X_test.columns if isinstance(X_test, pd.DataFrame) else None)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_train.columns if isinstance(X_train, pd.DataFrame) else None)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, name=y_test.name if isinstance(y_test, pd.Series) else None)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test, name=y_train.name if isinstance(y_train, pd.Series) else None)

    # Normalize X data
    scaler_X = StandardScaler()
    scaler_X.fit(X_train)
    X_train_normalized = pd.DataFrame(scaler_X.transform(X_train), columns=X_train.columns)
    X_test_normalized = pd.DataFrame(scaler_X.transform(X_test), columns=X_test.columns)

    # Normalize y data
    scaler_y = StandardScaler()
    scaler_y.fit(y_train.values.reshape(-1, 1))
    y_train_normalized = pd.Series(scaler_y.transform(y_train.values.reshape(-1, 1)).flatten(), name=y_train.name)
    y_test_normalized = pd.Series(scaler_y.transform(y_test.values.reshape(-1, 1)).flatten(), name=y_test.name)

    return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, scaler_y
