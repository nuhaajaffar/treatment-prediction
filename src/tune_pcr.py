import joblib
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
MODEL_PATH = "models/best_pcr_model.joblib"
RESULTS_PATH = "outputs/pcr_tuning.csv"

def main():
    train_df = load_data(TRAIN_PATH)
    X, y = prepare_pcr_data(train_df)

    pipeline = Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("feature_selection", MandatorySelectKBest(
            score_func = f_classif,
            feature_names = X.columns,
            mandatory_features = IMPORTANT_FEATURES
        )),
        ("classifier", LogisticRegression())
    ])

    param_grid = [
        {
            "feature_selection__k": [15, 20, 25, 30, 40, 60],
            "classifier": [
                LogisticRegression(
                    class_weight = "balanced",
                    max_iter = 5000,
                    random_state = 42,
                    solver = "liblinear"
                )
            ],
            "classifier__C": [0.01, 0.03, 0.1, 0.3, 1]
        },
        {
            "feature_selection__k": [15, 20, 25, 30, 40, 60],
            "classifier": [
                SVC(
                    class_weight = "balanced",
                    kernel = "rbf",
                    random_state = 42
                )
            ],
            "classifier__C": [0.1, 0.3, 1, 3, 10],
            "classifier__gamma": [0.03, 0.01, 0.003, 0.001, 0.0003]
        }
    ]

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

    grid_search = GridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        scoring = "balanced_accuracy",
        cv = cv,
        n_jobs = -1,
        return_train_score = True
    )

    grid_search.fit(X, y)

    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values(by = "rank_test_score")

    Path("outputs").mkdir(exist_ok = True)
    Path("models").mkdir(exist_ok = True)

    results_df.to_csv(RESULTS_PATH, index = False)
    joblib.dump(grid_search.best_estimator_, MODEL_PATH)

    print("Best PCR model:")
    print(grid_search.best_estimator_)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nBest CV balanced accuracy:")
    print(round(grid_search.best_score_, 4))

    selected_features = grid_search.best_estimator_.named_steps["feature_selection"].get_selected_features()

    print("\nSelected feature count:")
    print(len(selected_features))

    print("\nRequired features included:")
    print(all(feature in selected_features for feature in IMPORTANT_FEATURES))

    print(f"\nSaved tuning results to {RESULTS_PATH}")
    print(f"Saved best PCR model to {MODEL_PATH}")

if __name__ == "__main__":
    main()