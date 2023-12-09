from pre_processing import PreProcessing
from models import AutoEncoder, CLSTM
from itertools import product
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def tune_autoencoder():
    data = PreProcessing()
    data.pre_process_autoencoder()
    
    X_train = data.X_train
    y_train = data.y_train
    X_test = data.X_test
    y_test = data.y_test   

    input_dim = X_train.shape[1]

    autoencoder_grid = {
        'encoder_layer_sizes': [[64, 32, 16], [128, 64, 32]],
        'decoder_layer_sizes': [[32, 64, input_dim], [64, 128, input_dim]],
        'learning_rate': [0.001, 0.01],
        'optimizer': ['Adam', 'SGD'],
        'criterion': 'MSELoss'
    }

def tune_clstm():
    data = PreProcessing()
    data.pre_process_clstm()
    
    model = CLSTM()
    
    clstm_grid = {
        'conv1_out_channels': [32, 64],
        'kernel_size': [3, 5],
        'lstm_hidden_size': [32, 64],
        'fc1_out_features': [16, 32],
        'learning_rate': [0.001, 0.01],
        'optimizer': ['Adam', 'SGD'],
        'criterion': 'BCELoss'
    }
    
def train_and_validate(data, model, params, epochs=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    true_labels = []
    predicted_labels = []
    
    
    if params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
    else: # Adam
        optimizer = optim.Adam(model.parameters(), lr = params['learning_rate'])
        
    if params['criterion'] == 'BCELoss':
        criterion = 'MSELoss'
    else:
        criterion = 'BCELoss'

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for inputs, targets in data.train_loader:
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
    

def hyperparameter_tuning(hyperparams, data_loader, input_dim):
    param_combinations = [dict(zip(hyperparams, v)) for v in product(*hyperparams.values())]
    best_performance = None
    best_params = None

    for params in param_combinations:
        model = AutoEncoder(input_dim, params['encoder_layer_sizes'], params['decoder_layer_sizes'])
        performance = train_and_validate(model, data_loader, params)

        if best_performance is None or performance < best_performance:  # Assuming lower error is better
            best_performance = performance
            best_params = params

    return best_params, best_performance