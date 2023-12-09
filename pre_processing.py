import pandas as pd
import numpy as np
import glob
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class PreProcessing:
    def __init__(self):
        self.concatenated_df = None
        self.sw_df = None
        
    
    def read_files(self):   
        path = r'Yahoo_S5_dataset/A1Benchmark' #set the path accordingly
        all_files=glob.glob(path+"/*.csv")

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