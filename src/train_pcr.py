import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from pathlib import Path
from preprocessing import load_data, prepare_pcr_data, build_preprocessing_pipeline

TRAIN_PATH = "data/TrainDataset2025.xls"
MODEL_PATH = "models/pcr_logistic_regression.joblib"

def main():
    train_df = load_data(TRAIN_PATH)

    X, y = prepare_pcr_data(train_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 42,
        stratify = y
    )

    model = Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("classifier", LogisticRegression(class_weight = "balanced", max_iter = 1000, random_state = 42))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    balanced_acc = balanced_accuracy_score(y_val, y_pred)

    print("PCR Logistic Regression Baseline")
    print("Balanced accuracy:", round(balanced_acc, 4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\nClassification report:")
    print(classification_report(y_val, y_pred))
    
    Path("models").mkdir(exist_ok = True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()