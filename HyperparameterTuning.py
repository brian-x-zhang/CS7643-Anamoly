input_dim = 1

autoencoder_grid = {
    'encoder_layer_sizes': [[64, 32, 16], [128, 64, 32]],
    'decoder_layer_sizes': [[32, 64, input_dim], [64, 128, input_dim]],
    'learning_rate': [0.001, 0.01],
    'optimizer': ['Adam', 'SGD']
}

clstm_grid = {
    'conv1_out_channels': [32, 64],
    'kernel_size': [3, 5],
    'lstm_hidden_size': [32, 64],
    'fc1_out_features': [16, 32],
    'learning_rate': [0.001, 0.01],
    'optimizer': ['Adam', 'SGD']
}