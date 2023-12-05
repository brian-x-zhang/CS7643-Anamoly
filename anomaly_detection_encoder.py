#!/usr/bin/env python
# coding: utf-8

# ## **Anomaly Detection with Autoencoders**

import pandas as pd
import numpy as np
import glob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#For google colab
from google.colab import drive
drive.mount('/content/drive',force_remount=True)
#Your path
get_ipython().run_line_magic('cd', "'/content/drive/MyDrive/project/'")


#Import and concat all files
path = r'S5_database/A1Benchmark' #set the path accordingly
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


# I save the result, so I just read it from the result
#concatenated_df=pd.read_csv("final_dataset.csv")


concatenated_df


# Set the style of seaborn
sns.set(style="whitegrid")

# Create a scatter plot with seaborn
plt.figure(figsize=(10, 6))
scatter_plot = sns.scatterplot(x='timestamp', y='value', hue='is_anomaly', data=concatenated_df, palette={0: 'blue', 1: 'red'}, s=100)

# Create custom legend handles
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10)
]

# Add legend with custom handles
plt.legend(handles=legend_handles, title='Is Anomaly', labels=['No', 'Yes'], loc='upper left')


# Set plot title and labels
plt.title('Scatter Plot of Value over Time')
plt.xlabel('Timestamp')
plt.ylabel('Value')

# Show the plot
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


sw_df = convert_2d(concatenated_df)
sw_df


#concatenated_df.to_csv('final_dataset.csv', index=False)


# first try random forest and see how the model performs

# Splitting the dataset into training and testing sets
y = sw_df.iloc[:, 60]
X = sw_df.iloc[:, 0:60]

# Train-test split
#no shuffling since we are using sliding window
train_size = int(0.7 * len(X))
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Creating a Random Forest classifier object
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
# Fitting the Random Forest classifier to the training data
rfc.fit(X_train, y_train)

# Making predictions on the testing data
y_pred = rfc.predict(X_test)

# Printing the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate Precision
precision = precision_score(y_test, y_pred)

# Calculate Recall
recall = recall_score(y_test, y_pred)

# Calculate F1-Score
f1 = f1_score(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')


# **Anomaly Detection with Autoencoders with Pytorch**

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = scaler.fit_transform(X_test)

# Convert data to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Create DataLoader for training
dataset = TensorDataset(X_train, X_train)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)


# Instantiate the autoencoder
input_dim = X_train.shape[1]
autoencoder = Autoencoder(input_dim)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.005)


num_epochs = 100
epoch_losses = []

for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in dataloader:
        inputs, _ = batch
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    epoch_loss = total_loss / len(dataloader)
    epoch_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


# Plotting the training loss
plt.figure(figsize=(10, 6))
plt.plot(epoch_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()


# Extract the encoder for feature representation
encoder = autoencoder.encoder

class Classifier(nn.Module):
    def __init__(self, encoder):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.sigmoid(self.fc(x))
        return x

# Instantiate the classifier
classifier = Classifier(encoder)

# Define the loss function and optimizer for the classifier
criterion_classifier = nn.BCELoss()
optimizer_classifier = optim.Adam(classifier.parameters(), lr=0.001)

# Convert labels to PyTorch tensor
y_train = y_train.view(-1, 1)

# Training loop for the classifier
num_classifier_epochs = 100

for epoch in range(num_classifier_epochs):
    optimizer_classifier.zero_grad()
    outputs = classifier(X_train)
    loss = criterion_classifier(outputs, y_train)
    loss.backward()
    optimizer_classifier.step()

    print(f"Classifier Epoch {epoch+1}/{num_classifier_epochs}, Loss: {loss.item():.4f}")


# Check if GPU is available and use it, otherwise use CPU
device = torch.device("cpu")
autoencoder.to(device)
classifier.to(device)


# Create DataLoader for the test set
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Set the model to evaluation mode
autoencoder.eval()
classifier.eval()

# Lists to store true labels and predicted labels
predicted_labels = []

# Evaluation loop
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Pass inputs through the autoencoder
        encoded_features = autoencoder(inputs)

        # Pass encoded features through the classifier
        outputs = classifier(encoded_features)

        # Assuming binary classification
        predicted = (outputs > 0.5).float()

        # Convert PyTorch tensors to NumPy arrays
        predicted_labels.extend(predicted.cpu().numpy())

# Convert lists to NumPy arrays
predicted_labels = np.array(predicted_labels)

# Convert y_test to NumPy array
y_test_np = y_test.cpu().numpy()

# Calculate Accuracy, Precision, Recall, and F1-Score
accuracy = accuracy_score(y_test_np, predicted_labels)
precision = precision_score(y_test_np, predicted_labels)
recall = recall_score(y_test_np, predicted_labels)
f1 = f1_score(y_test_np, predicted_labels)

# Print the results
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-Score: {f1 * 100:.2f}%')


# Plotting the confusion matrix
cm = confusion_matrix(y_test_np, predicted_labels)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()




