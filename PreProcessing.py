import pandas as pd
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from keras.utils import to_categorical

class PreProcessing:
    def __init__(self):
        self.concatenated_df = None
        self.sw_df = None
        
    
    def read_files(self):   
        path = r'Yahoo_S5_dataset/A1Benchmark' #set the path accordingly
        all_files=glob.glob(path+"/*.csv")
        
        # ensure same data order as notebooks
        all_files = ['Yahoo_S5_dataset/A1Benchmark/real_54.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_13.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_27.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_23.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_12.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_46.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_63.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_11.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_43.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_26.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_41.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_42.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_60.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_39.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_18.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_3.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_1.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_47.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_5.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_31.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_32.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_64.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_56.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_14.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_20.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_40.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_9.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_35.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_7.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_51.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_10.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_36.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_52.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_21.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_4.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_30.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_6.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_2.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_44.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_24.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_59.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_45.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_8.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_15.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_55.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_16.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_62.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_53.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_33.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_49.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_22.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_65.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_34.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_17.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_57.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_25.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_28.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_19.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_29.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_48.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_50.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_61.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_67.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_38.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_58.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_66.csv',
        'Yahoo_S5_dataset/A1Benchmark/real_37.csv']
        
        preprocessed_dfs = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            df['value'] = df['value'].replace(0, np.nan)
            df = df.dropna(subset=['value'])

            # Normalize the 'value' column
            normalized_values = preprocessing.normalize([df['value'].to_numpy()])[0]
            df['value'] = normalized_values

            preprocessed_dfs.append(df)
            
        concatenated_df = pd.concat(preprocessed_dfs, axis=0, ignore_index=True)

        def convert_2d(df):
            rows = []

            for i in range(len(df) - 59):
                segment = df.iloc[i:i+60]
                is_anomaly = segment['is_anomaly'].any()
                new_row = segment['value'].tolist() + [int(is_anomaly)]
                rows.append(new_row)
            data_frame = pd.DataFrame(rows)

            return data_frame


        sw_df = convert_2d(concatenated_df)
        
        self.concatenated_df = concatenated_df
        self.sw_df = sw_df
    
    def train_test_split(self, scaleX=False):
        
        # Splitting the dataset into training and testing sets
        y = self.sw_df.iloc[:, 60]
        X = self.sw_df.iloc[:, 0:60]

        # Train-test split
        #no shuffling since we are using sliding window
        train_size = int(0.7 * len(X))
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        if scaleX:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    def pre_process_autoencoder(self):
        self.read_files()
        self.train_test_split(scaleX=True)
        
        # Convert data to PyTorch tensors
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test.values, dtype=torch.float32)
        
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=512, shuffle=True)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
        
    def pre_process_clstm(self, batch_size=512):
        self.read_files()
        self.train_test_split()
        
        self.X_train = self.X_train.to_numpy().reshape(-1, 60, 1)
        self.X_test = self.X_test.to_numpy().reshape(-1, 60, 1)
        self.y_test = self.y_test.to_numpy()
        self.y_train = to_categorical(self.y_train)
        
        
        self.X_train = torch.Tensor(self.X_train).permute(0, 2, 1)
        self.X_test = torch.Tensor(self.X_test).permute(0, 2, 1)
        self.y_train = torch.Tensor(self.y_train)
        self.y_test = torch.LongTensor(self.y_test)
        
        self.train_dataset = TensorDataset(self.X_train, self.y_train)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataset = TensorDataset(self.X_test, self.y_test)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)