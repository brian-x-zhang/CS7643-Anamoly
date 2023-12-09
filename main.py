import pandas as pd
from HyperparameterTuning import tune_clstm, tune_autoencoder

clstm_tuning_results, clstm_model = tune_clstm()


writer = pd.ExcelWriter('HyperparameterTuning.xlsx')
clstm_tuning_results.to_excel(writer, 'CLSTM')
writer.close()

print('done')