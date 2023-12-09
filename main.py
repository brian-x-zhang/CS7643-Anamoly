import pandas as pd
from HyperparameterTuning import tune_clstm, tune_autoencoder

def main():
    clstm_tuning_results, clstm_model = tune_clstm()

    writer = pd.ExcelWriter('HyperparameterTuning.xlsx')
    clstm_tuning_results.to_excel(writer, 'CLSTM')
    writer.close()

    print('done')

if __name__ == "__main__":
    main()