import pandas as pd

autoenc = pd.read_csv('autoencoder_tuning_2024-12-10.csv')
clstm = pd.read_csv('clstm_tuning_2024-12-10.csv')

print(autoenc.to_latex(caption='Autoencoder Hyperparameter Tuning Results'))

print(clstm.to_latex(caption='C-LSTM Hyperparameter Tuning Results'))