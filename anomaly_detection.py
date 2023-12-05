#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import glob
from sklearn import preprocessing
import matplotlib.pyplot as plt


#For google colab
from google.colab import drive
drive.mount('/content/drive',force_remount=True)
#Your path
get_ipython().run_line_magic('cd', "'/content/drive/MyDrive/DL/Project/'")


#view of a sample of data
df=pd.read_csv("data/A1Benchmark/real_1.csv")
df


#Convert 0 in value to NaN and drop
df['value'] = df['value'].replace(0, np.nan)
df = df.dropna(subset=['value'])
df.value = preprocessing.normalize([df.value]).T


plt.plot(df.timestamp,df.value)
plt.xlabel("Timestamp")
plt.ylabel("NormalizedValue")
plt.title(" Example plot of web traffic after preprocessing ")
plt.show()


#function to convert dataframe to into 2d array
#creating sliding window of length 60 values in a sequence

def convert_2d(df):
    rows = []

    for i in range(len(df) - 59):
        segment = df.iloc[i:i+60]
        is_anomaly = segment['is_anomaly'].any()
        new_row = segment['value'].tolist() + [int(is_anomaly)]
        rows.append(new_row)
    data_frame = pd.DataFrame(rows)

    return data_frame


#test on sample
df2 = convert_2d(df)
df2


#Import and concat all files
path = r'data/A1Benchmark' #set the path accordingly
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


concatenated_df


frame = convert_2d(concatenated_df)


frame


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CLSTM(nn.Module):
    def __init__(self):
        super(CLSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2)
        self.lstm = nn.LSTM(input_size=960, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1, 960)

        x, _ = self.lstm(x)
        x = self.flatten(x)
        #x = x.contiguous().view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x


y = frame.iloc[:, 60]
X = frame.iloc[:, 0:60]

# Train-test split
#no shuffling since we are using sliding window
train_size = int(0.7 * len(X))
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]


#convert to tensors
X_train = torch.tensor(X_train.values.reshape(train_size, 1, 60)).float()
X_test = torch.tensor(X_test.values.reshape(len(X_test), 1, 60)).float()

y_train = torch.tensor(pd.get_dummies(y_train).values).float()
y_test = torch.tensor(pd.get_dummies(y_test).values).float()


batch_size=512

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


model = CLSTM()


# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())


#Total parameters
sum(p.numel() for p in model.parameters())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Train
epochs = 100

epoch_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    batch_loss = loss.item()
    epoch_losses.append(average_loss)
    print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {batch_loss}')


# Plotting the training loss
plt.figure(figsize=(10, 6))
plt.plot(epoch_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()





# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predicted = torch.argmax(outputs, 1)
        total += targets.size(0)
        correct += (predicted == torch.argmax(targets, 1)).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy * 100}%')

