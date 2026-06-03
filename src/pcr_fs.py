import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from pathlib import Path
from feature_selection import MandatorySelectKBest
from preprocessing import (
    IMPORTANT_FEATURES,
    load_data,
    prepare_pcr_data,
    build_preprocessing_pipeline
)

TRAIN_PATH = "data/TrainDataset2025.xls"
RESULTS_PATH = "outputs/pcr_fs.csv"

def build_models():
    return {
        "Logistic Regression + FS": LogisticRegression(class_weight = "balanced", max_iter = 1000, random_state = 42),
        "SVM + FS": SVC(class_weight = "balanced", kernel = "rbf", random_state = 42)
    }

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

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    results = []

    for model_name, classifier in build_models().items():
        pipeline = Pipeline([
            ("preprocessing", build_preprocessing_pipeline()),
            ("feature_selection", MandatorySelectKBest(
                score_func = f_classif,
                k = 30,
                feature_names = X.columns,
                mandatory_features = IMPORTANT_FEATURES
            )),
            ("classifier", classifier)
        ])

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv = cv,
            scoring = "balanced_accuracy"
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        val_score = balanced_accuracy_score(y_val, y_pred)

        selected_features = pipeline.named_steps["feature_selection"].get_selected_features()

        results.append({
            "model": model_name,
            "selected_feature_count": len(selected_features),
            "cv_balanced_accuracy_mean": cv_scores.mean(),
            "cv_balanced_accuracy_std": cv_scores.std(),
            "validation_balanced_accuracy": val_score,
            "selected_features": ", ".join(selected_features)
        })

        print("\n" + "=" * 50)
        print(model_name)
        print("Selected feature count:", len(selected_features))
        print("Required features included:", all(feature in selected_features for feature in IMPORTANT_FEATURES))
        print("CV balanced accuracy:", round(cv_scores.mean(), 4), "+/-", round(cv_scores.std(), 4))
        print("Validation balanced accuracy:", round(val_score, 4))
        print("\nConfusion matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("\nClassification report:")
        print(classification_report(y_val, y_pred))

    results_df = pd.DataFrame(results).sort_values(
        by="validation_balanced_accuracy",
        ascending = False
    )

    Path("outputs").mkdir(exist_ok = True)
    results_df.to_csv(RESULTS_PATH, index = False)

    print("\nFinal PCR feature selection comparison:")
    print(results_df[[
        "model",
        "selected_feature_count",
        "cv_balanced_accuracy_mean",
        "cv_balanced_accuracy_std",
        "validation_balanced_accuracy"
    ]])
    print(f"\nSaved feature selection results to {RESULTS_PATH}")

if __name__ == "__main__":
    main()