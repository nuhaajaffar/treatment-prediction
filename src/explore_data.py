import pandas as pd
import numpy as np

TRAIN_PATH = "data/TrainDataset2025.xls"
TEST_PATH = "data/TestDatasetExample.xls"

PCR_TARGET = "pCR (outcome)"
RFS_TARGET = "RelapseFreeSurvival (outcome)"
ID_COL = "ID"

def main():
    train_df = pd.read_excel(TRAIN_PATH)
    test_df = pd.read_excel(TEST_PATH)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nColumns in training but not test:")
    print([col for col in train_df.columns if col not in test_df.columns])

    print("\nPCR target distribution:")
    print(train_df[PCR_TARGET].value_counts(dropna = False))

    print("\nRFS target summary:")
    print(train_df[RFS_TARGET].describe())

    feature_cols = [
        col for col in train_df.columns
        if col not in [ID_COL, PCR_TARGET, RFS_TARGET]
    ]

    train_999_count = (train_df[feature_cols] == 999).sum().sum()
    test_feature_cols = [col for col in test_df.columns if col != ID_COL]
    test_999_count = (test_df[test_feature_cols] == 999).sum().sum()

    print("\nTotal 999 values in train features:", train_999_count)
    print("Total 999 values in test features:", test_999_count)

    print("\nTrain columns containing 999:")
    train_missing_cols = (train_df[feature_cols] == 999).sum()
    print(train_missing_cols[train_missing_cols > 0].sort_values(ascending = False))

    print("\nTest columns containing 999:")
    test_missing_cols = (test_df[test_feature_cols] == 999).sum()
    print(test_missing_cols[test_missing_cols > 0].sort_values(ascending = False))

    print("\nNon-numeric feature columns:")
    print(train_df[feature_cols].select_dtypes(exclude = "number").columns.tolist())

if __name__ == "__main__":
    main()