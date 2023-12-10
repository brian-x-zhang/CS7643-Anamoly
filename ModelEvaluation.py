from PreProcessing import PreProcessing
from HyperparameterTuning import train_and_validate

clstm_params = {
    'type': ['clstm'],
    'conv1_out_channels': [64],
    'kernel_size': [5],
    'lstm_hidden_size': [64],
    'fc1_out_features': [16],
    'learning_rate': [0.001],
    'optimizer': ['Adam'], # SGD performed poorly
    'criterion': ['BCELoss']
}



autoencoder_params = {
    
}