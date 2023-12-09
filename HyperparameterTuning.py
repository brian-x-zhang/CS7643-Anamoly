import pandas as pd
import numpy as np

from PreProcessing import PreProcessing
from models.AutoEncoder import Autoencoder
from models.CLSTM import CLSTM
from itertools import product
from pprint import pprint
from copy import deepcopy
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def tune_autoencoder():
    data = PreProcessing()
    data.pre_process_autoencoder()

    input_dim = data.X_train.shape[1]

    autoencoder_grid = {
        'type': ['autoencoder'],
        'encoder_layer_sizes': [[64, 32, 16], [128, 64, 32]],
        'decoder_layer_sizes': [[32, 64, input_dim], [64, 128, input_dim]],
        'learning_rate': [0.001, 0.01],
        'optimizer': ['Adam'],
        'criterion': ['MSELoss']
    }

def tune_clstm():
    data = PreProcessing()
    data.pre_process_clstm()
    
    clstm_grid = {
        'type': ['clstm'],
        'conv1_out_channels': [32, 64],
        'kernel_size': [3, 5],
        'lstm_hidden_size': [32, 64],
        'fc1_out_features': [16, 32],
        'learning_rate': [0.001, 0.01],
        'optimizer': ['Adam'],
        'criterion': ['BCELoss']
    }
    
    results, best_model = hyperparameter_tuning(clstm_grid, data)
    
    return results, best_model
    
    
def train_and_validate(data, model, params, epochs=100):
    pprint(f"Training Model: {pprint(params)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    true_labels = []
    predicted_labels = []
    
    # Optimizer
    if params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
    else: # Adam
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
    # Loss    
    if params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
        
    # Gradient Clipping
    max_grad_norm = 1.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for inputs, targets in data.train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if params['type'] == 'clstm':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) #gradient clipping

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(data.train_loader)
        train_losses.append(avg_train_loss)

        print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}')
    
    model.eval()
    
    true_labels = []
    predicted_labels = []
    test_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data.test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(targets.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            test_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)


    # Performance Metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    
    avg_test_loss = test_loss / total_samples
    
    return f1, accuracy, precision, recall, avg_test_loss, model

def hyperparameter_tuning(hyperparams, data):
    
    param_combinations = [dict(zip(hyperparams, v)) for v in product(*hyperparams.values())]
    best_performance = None
    best_params = None

    results = []

    for params in param_combinations:
        
        if params['type'] == 'autoencoder':
            model = Autoencoder(data.X_train.shape[1], params['encoder_layer_sizes'], params['decoder_layer_sizes'])
        else:
            model = CLSTM(params['conv1_out_channels'], params['kernel_size'], params['lstm_hidden_size'], params['fc1_out_features'])    
        
        f1, accuracy, precision, recall, avg_test_loss, _model = train_and_validate(data, model, params)
        
        params['F1'] = f1
        params['Accuracy'] = accuracy
        params['Recall'] = recall
        params['Average Test Loss']: avg_test_loss
        results.append(params)
        
        print(f"""
              F1-Score: {f1}\n
              Accuracy: {accuracy}\n
              Precision: {precision}\n
              Recall: {recall}\n
              Average Test Loss: {avg_test_loss}      
              """)
        
        performance = deepcopy(f1)
        
        if best_performance is None or performance < best_performance:  # Assuming lower error is better
            best_performance = performance
            best_params = params
            best_model = _model
            
    print(f'Best Performance: {best_performance}\nBest Params: {best_params}')
    
    results = pd.DataFrame(results)
    
    return results, best_model
