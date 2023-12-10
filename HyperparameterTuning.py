import pandas as pd
import numpy as np

from PreProcessing import PreProcessing
from models.AutoEncoder import Autoencoder, Classifier
from models.CLSTM import CLSTM
from itertools import product
from pprint import pprint
from copy import deepcopy
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def tune_autoencoder(autoencoder_grid=None):
    data = PreProcessing()
    data.pre_process_autoencoder()

    input_dim = data.X_train.shape[1]

    if autoencoder_grid is None:
        autoencoder_grid = {
            'type': ['autoencoder'],
            'encoder_layer_sizes': [[32, 16, 8], [64, 32, 16], [128, 64, 32]],
            # 'decoder_layer_sizes': [[32, 64, input_dim], [64, 128, input_dim]],
            'learning_rate': [0.001, 0.01, 0.05],
            'learning_rate_classifier': [0.0005, 0.001, 0.005],
            'activation': [nn.LeakyReLU(), nn.ReLU()],
            'optimizer': ['Adam'],
            'criterion': ['MSELoss'],
            'epochs': [200]
        }
        
    results, best_model = hyperparameter_tuning(autoencoder_grid, data)
    
    return results, best_model

def tune_clstm(clstm_grid=None):
    data = PreProcessing()
    data.pre_process_clstm()
    
    if clstm_grid is None:
        clstm_grid = {
            'type': ['clstm'],
            'conv1_out_channels': [16, 32],
            'kernel_size': [3, 5],
            'lstm_hidden_size': [16, 32],
            'fc1_out_features': [4, 16, 32],
            'learning_rate': [0.001, 0.01],
            'optimizer': ['Adam'], # SGD performed poorly
            'criterion': ['BCELoss'],
            'epochs': [200]
        }
    
    results, best_model = hyperparameter_tuning(clstm_grid, data)
    
    return results, best_model
    
    
def train_and_validate(data, model, params):
    pprint(f"Training Model: {pprint(params)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    true_labels = []
    predicted_labels = []
    
    # Epochs
    epochs = params['epochs']
    
    # Optimizer
    if params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['learning_rate'])
        if params['type'] == 'autoencoder':
            optimizer_classifier = torch.optim.SGD(model.parameters(), lr=params['learning_rate_classifier'])
    else: # Adam
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        if params['type'] == 'autoencoder':
            optimizer_classifier = optim.Adam(model.parameters(), lr=params['learning_rate_classifier'])
        
    # Loss    
    if params['criterion'] == 'MSELoss':
        criterion = nn.MSELoss()
        criterion_classifier = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
        criterion_classifier = nn.BCELoss()
        
    # Gradient Clipping
    max_grad_norm = 1.0

    # Training Loop
    for epoch in range(epochs):
        train_loss = 0
        
        if params['type'] == 'clstm':
            model.train()

            # CLSTM Training Loop
            for inputs, targets in data.train_loader:
                
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) #gradient clipping

                optimizer.step()
                train_loss += loss.item()

            
        else:
            # Autoencoder Training Loop
            for batch in data.train_loader:
                inputs, _ = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
        avg_train_loss = train_loss / len(data.train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}')
        
    autoencoder = model
        
    if params['type'] == 'autoencoder':
    # Autoencoder Classifier Training Loop
        classifier = Classifier(autoencoder.encoder) 

        y_train = data.y_train.view(-1, 1)
        
        for epoch in range(epochs):
            optimizer_classifier.zero_grad()
            outputs = classifier(data.X_train)
            loss = criterion_classifier(outputs, y_train)
            loss.backward()
            optimizer_classifier.step()

            print(f"Classifier Epoch {epoch+1}, Loss: {loss.item():.4f}")

        autoencoder.to(device)
        classifier.to(device)
        
        autoencoder.eval()
        classifier.eval()
        
    model.eval()
    
    true_labels = []
    predicted_labels = []
    test_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        
        if params['type'] == 'clstm':
        
            for inputs, targets in data.test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                true_labels.extend(targets.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
                
                test_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                
        elif params['type'] == 'autoencoder':
            probs = []
            for inputs, targets in data.test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Pass inputs through the autoencoder
                encoded_features = autoencoder(inputs)

                # Pass encoded features through the classifier
                outputs = classifier(encoded_features)
                probs.extend(outputs)
                
                # Assuming binary classification
                predicted = (outputs > 0.5).float()

                # Convert PyTorch tensors to NumPy arrays
                predicted_labels.extend(predicted.cpu().numpy())

                true_labels.extend(targets.cpu().numpy())
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
            model = Autoencoder(data.X_train.shape[1], params['encoder_layer_sizes'], params['activation'])
        else:
            model = CLSTM(params['conv1_out_channels'], params['kernel_size'], params['lstm_hidden_size'], params['fc1_out_features'])    
        
        f1, accuracy, precision, recall, avg_test_loss, _model = train_and_validate(data, model, params)
        
        params['F1'] = f1
        params['Accuracy'] = accuracy
        params['Recall'] = recall
        params['Precision'] = precision
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

def find_best_threshold(y_true, y_probs):
    best_threshold = 0.5
    best_f1 = 0.0

    # Iterate through a range of possible threshold values
    for threshold in np.arange(0.0, 1.01, 0.01):
        # Convert probabilities to binary predictions based on the current threshold
        y_pred = (y_probs >= threshold).astype(int)
        
        # Calculate the F1 score
        f1 = f1_score(y_true, y_pred)
        
        # Update the best threshold if this one is better
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1
