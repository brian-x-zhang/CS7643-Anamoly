import pandas as pd
from HyperparameterTuning import tune_clstm, tune_autoencoder

# Custom print function to log to both console and file
def custom_print(*args, **kwargs):
    with open('log.txt', 'a') as log_file:
        print(*args, **kwargs, file=log_file)
    original_print(*args, **kwargs)

# Replace the built-in print with custom_print
original_print = print
print = custom_print

def main():
    # clstm_tuning_results, clstm_model = tune_clstm()
    autoencoder_tuning_results, autoencoder_model = tune_autoencoder()

    # clstm_tuning_results.to_csv('clstm_tuning.csv')
    autoencoder_tuning_results.to_csv('autoencoder_tuning.csv')

if __name__ == "__main__":
    main()