import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from pathlib import Path
from preprocessing import load_data, prepare_pcr_data, build_preprocessing_pipeline

TRAIN_PATH = "data/TrainDataset2025.xls"
RESULTS_PATH = "outputs/pcr_model_comparison.csv"

def build_models():
    return {
        "Logistic Regression": LogisticRegression(class_weight = "balanced", max_iter = 1000, random_state = 42),
        "Random Forest": RandomForestClassifier(class_weight = "balanced", n_estimators = 200, random_state = 42),
        "Gradient Boosting": GradientBoostingClassifier(random_state = 42),
        "SVM": SVC(class_weight = "balanced", kernel = "rbf", random_state = 42)
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

        results.append({
            "model": model_name,
            "cv_balanced_accuracy_mean": cv_scores.mean(),
            "cv_balanced_accuracy_std": cv_scores.std(),
            "validation_balanced_accuracy": val_score
        })

        print("\n" + "=" * 50)
        print(model_name)
        print("CV balanced accuracy:", round(cv_scores.mean(), 4), "+/-", round(cv_scores.std(), 4))
        print("Validation balanced accuracy:", round(val_score, 4))
        print("\nConfusion matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("\nClassification report:")
        print(classification_report(y_val, y_pred))

    results_df = pd.DataFrame(results).sort_values(
        by = "validation_balanced_accuracy",
        ascending = False
    )

    Path("outputs").mkdir(exist_ok = True)
    results_df.to_csv(RESULTS_PATH, index = False)

    print("\nFinal PCR model comparison:")
    print(results_df)
    print(f"\nSaved comparison results to {RESULTS_PATH}")

if __name__ == "__main__":
    main()