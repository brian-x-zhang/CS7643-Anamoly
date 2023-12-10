import torch
import torch.nn as nn

from PreProcessing import PreProcessing
from HyperparameterTuning import tune_autoencoder, tune_clstm

# Best clstm params
clstm_params = {
    'type': ['clstm'],
    'conv1_out_channels': [16],
    'kernel_size': [5],
    'lstm_hidden_size': [16],
    'fc1_out_features': [32],
    'learning_rate': [0.01],
    'optimizer': ['Adam'], # SGD performed poorly
    'criterion': ['BCELoss'],
    'epochs': [500]
}

results, clstm = tune_clstm(clstm_params)

# Best autoencoder params
autoencoder_grid = {
    'type': ['autoencoder'],
    'encoder_layer_sizes': [[128, 64, 32]],
    # 'decoder_layer_sizes': [[32, 64, input_dim], [64, 128, input_dim]],
    'learning_rate': [0.01],
    'learning_rate_classifier': [0.005],
    'activation': [nn.ReLU()],
    'optimizer': ['Adam'],
    'criterion': ['MSELoss'],
    'epochs': [500]
}

results, autoencoder = tune_autoencoder(autoencoder_grid)

print('done')