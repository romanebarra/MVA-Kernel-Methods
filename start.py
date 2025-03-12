import pandas as pd
from processing import process_dataset

# My best parameters
k=11 # length of each substrings extracted
m=2 # maximum number of mismatch
C=1.0 # SVM regularization parameter

def main():
    predictions = []
    for dataset in [0, 1, 2]:
        prediction_dataset = process_dataset(dataset, k, m, C)
        predictions.append(prediction_dataset)
    Yte = []
    for dataset, prediction_dataset in enumerate(predictions):
        offset = dataset * 1000
        for i, pred in enumerate(prediction_dataset):
            Yte.append((offset + i, pred))
    Yte = pd.DataFrame(Yte, columns=["Id", "Bound"])
    Yte.to_csv("Yte.csv", index=False)

if __name__ == '__main__':
    main()