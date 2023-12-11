import pandas as pd

autoenc = pd.read_csv('autoencoder_tuning_2024-12-10.csv')
clstm = pd.read_csv('clstm_tuning_2024-12-10.csv')

a = autoenc.drop(labels='Unnamed: 0', axis=1).to_latex(caption='Autoencoder Hyperparameter Tuning Results', index=False, escape=True)
print(a)
# print(clstm.drop(labels='Unnamed: 0', axis=1).to_latex(caption='C-LSTM Hyperparameter Tuning Results', index=False))