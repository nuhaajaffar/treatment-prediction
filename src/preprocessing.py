import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ID_COL = "ID"
PCR_TARGET = "pCR (outcome)"
RFS_TARGET = "RelapseFreeSurvival (outcome)"
MISSING_VALUE = 999

IMPORTANT_FEATURES = ["ER", "HER2", "Gene"]

def load_data(train_path, test_path = None):
    train_df = pd.read_excel(train_path)

    if test_path is not None:
        test_df = pd.read_excel(test_path)
        return train_df, test_df

    return train_df

def replace_missing_values(df):
    return df.replace(MISSING_VALUE, np.nan)

def get_feature_columns(df):
    excluded_cols = [ID_COL, PCR_TARGET, RFS_TARGET]
    return [col for col in df.columns if col not in excluded_cols]

def prepare_pcr_data(df):
    df = replace_missing_values(df)
    df = df[df[PCR_TARGET].notna()].copy()

    X = df[get_feature_columns(df)]
    y = df[PCR_TARGET].astype(int)

    return X, y

def prepare_rfs_data(df):
    df = replace_missing_values(df)

    X = df[get_feature_columns(df)]
    y = df[RFS_TARGET]

    return X, y

def prepare_test_data(df):
    df = replace_missing_values(df)

    patient_ids = df[ID_COL]
    X = df.drop(columns = [ID_COL])

    return patient_ids, X

def build_preprocessing_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy = "median")),
        ("scaler", StandardScaler())
    ])