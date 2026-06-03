import joblib
import pandas as pd
from pathlib import Path
from preprocessing import prepare_test_data

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = BASE_DIR / "data" / "FinalTestDataset2025.xls"
MODEL_PATH = BASE_DIR / "models" / "best_rfs_model.joblib"
OUTPUT_PATH = BASE_DIR / "outputs" / "RFSPrediction.csv"

def main():
    test_df = pd.read_excel(TEST_PATH)

    patient_ids, X_test = prepare_test_data(test_df)

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    output_df = pd.DataFrame({
        "ID": patient_ids,
        "RFS": predictions
    })

    OUTPUT_PATH.parent.mkdir(exist_ok = True)
    output_df.to_csv(OUTPUT_PATH, index = False)

    print(f"Saved RFS predictions to {OUTPUT_PATH}")
    print(output_df)

if __name__ == "__main__":
    main()