# %%
import pandas as pd
import numpy as np
import glob
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# %%
#For google colab
from google.colab import drive
drive.mount('/content/drive',force_remount=True)
#Your path
%cd '/content/drive/MyDrive/DL/Project/'


# %%
#view of a sample of data
df=pd.read_csv("data/A1Benchmark/real_1.csv")
df

# %%
#Convert 0 in value to NaN and drop
df['value'] = df['value'].replace(0, np.nan)
df = df.dropna(subset=['value'])
df.value = preprocessing.normalize([df.value]).T

# %%
plt.plot(df.timestamp,df.value)
plt.xlabel("Timestamp")
plt.ylabel("NormalizedValue")
plt.title(" Example plot of web traffic after preprocessing ")
plt.show()

# %%
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

# %%
#test on sample
df2 = convert_2d(df)
df2

# %%
#Import and concat all files
path = r'data/A1Benchmark' #set the path accordingly
all_files=glob.glob(path+"/*.csv")


# %%
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

# %%
concatenated_df

# %%
frame = convert_2d(concatenated_df)

# %%
frame

# %%
frame.to_csv('final_dataset.csv', index=False)

# %%
frame=pd.read_csv("final_dataset.csv")
frame

# %%
from keras.models import Sequential
from keras.layers import Dense,Reshape,Conv2D,Flatten,MaxPooling1D,Conv1D,LSTM
from keras import optimizers
from keras.utils import to_categorical

# %%
#Keras model for comparison
model_k=Sequential()
model_k.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='tanh',input_shape=(60, 1)))
model_k.add(MaxPooling1D(pool_size=2, strides=2))
model_k.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='tanh'))
model_k.add(MaxPooling1D(pool_size=2, strides=2))
model_k.add(Reshape((1,15*64)))
model_k.add(LSTM(64, activation='tanh',return_sequences='False'))
model_k.add(Flatten())
model_k.add(Dense(32, activation='tanh'))
model_k.add(Dense(2, activation='softmax'))
model_k.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model_k.summary()

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# %%
class CLSTM(nn.Module):
    def __init__(self):
        super(CLSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding = 'same')
        self.pool = nn.MaxPool1d(2, stride=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding = 'same')
        self.lstm = nn.LSTM(input_size=960, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.conv1(x)
        x = self.tanh(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.tanh(x)
        x = self.pool(x)

        # Reshape for LSTM layer
        #x = x.permute(0, 2, 1)
        #x = x.reshape(x.size(0), 1, -1)
        x = x.view(x.size(0), 1, -1)
        x, _ = self.lstm(x)

        # Flatten the output for Dense layer
        #x = self.flatten(x)
        x = x.contiguous().view(x.size(0), -1)

        # Dense layers
        x = self.tanh(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# %%
y = frame.iloc[:, 60]
X = frame.iloc[:, 0:60]

# Train-test split
train_size = int(0.7 * len(X))
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# %%
#reshaping the data
X_train = X_train.to_numpy().reshape(-1, 60, 1)
X_test = X_test.to_numpy().reshape(-1, 60, 1)
y_test = y_test.to_numpy()

# %%
#converting y_train to categorical
y_train = to_categorical(y_train)

# %%
#train the Keras model
history=model_k.fit(x=X_train, y=y_train, batch_size=512, epochs=100, verbose=2)

# %%
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %%
#convert to tensors
X_train_tensor = torch.Tensor(X_train).permute(0, 2, 1)
X_test_tensor = torch.Tensor(X_test).permute(0, 2, 1)
y_train_tensor = torch.Tensor(y_train)
y_test_tensor = torch.LongTensor(y_test)


# Create DataLoader
batch_size=512
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
model = CLSTM()

# %%
learning_rate = 0.001

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# %%
def model_summary(model):
    print("Model Summary:")
    total_params = 0

    for name, parameter in model.named_parameters():
        param = parameter.numel()
        total_params += param
        if parameter.requires_grad:
            print(f"Layer: {name} | Size: {param} | Requires Grad: {parameter.requires_grad}")

    print(f"Total Parameters: {total_params}")

model_summary(model)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
#Train Pytorch model
epochs = 100
max_grad_norm = 1.0

train_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) #gradient clipping

        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}')

# %%
# Plotting the training loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# %%
model.eval()

true_labels = []
predicted_labels = []
test_loss = 0.0
total_samples = 0

with torch.no_grad():
    for inputs, targets in test_loader:
      inputs = inputs.to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs, 1)
      true_labels.extend(targets.cpu().numpy())
      predicted_labels.extend(predicted.cpu().numpy())
      test_loss += loss.item() * inputs.size(0)
      total_samples += inputs.size(0)


accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

average_test_loss = test_loss / total_samples

print(f'Test Loss: {average_test_loss:.4f}')


print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# %%
cm = confusion_matrix(true_labels, predicted_labels)

sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()


