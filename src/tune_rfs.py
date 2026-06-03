import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from pathlib import Path
from feature_selection import MandatorySelectKBest
from preprocessing import (
    IMPORTANT_FEATURES,
    load_data,
    prepare_rfs_data,
    build_preprocessing_pipeline
)

TRAIN_PATH = "data/TrainDataset2025.xls"
MODEL_PATH = "models/best_rfs_model.joblib"
RESULTS_PATH = "outputs/rfs_tuning.csv"

def main():
    train_df = load_data(TRAIN_PATH)
    X, y = prepare_rfs_data(train_df)

    pipeline = Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("feature_selection", MandatorySelectKBest(
            score_func = f_regression,
            feature_names = X.columns,
            mandatory_features = IMPORTANT_FEATURES
        )),
        ("regressor", SVR())
    ])

    param_grid = [
        {
            "feature_selection__k": [20, 30, 40, 60],
            "regressor": [SVR(kernel = "rbf")],
            "regressor__C": [0.1, 1, 10, 100],
            "regressor__gamma": ["scale", 0.01, 0.001],
            "regressor__epsilon": [0.1, 1, 5]
        },
        {
            "feature_selection__k": [20, 30, 40, 60],
            "regressor": [RandomForestRegressor(random_state = 42)],
            "regressor__n_estimators": [200, 500],
            "regressor__max_depth": [None, 5, 10],
            "regressor__min_samples_leaf": [1, 3, 5]
        },
        {
            "feature_selection__k": [20, 30, 40, 60],
            "regressor": [GradientBoostingRegressor(random_state = 42)],
            "regressor__n_estimators": [100, 200],
            "regressor__learning_rate": [0.03, 0.05, 0.1],
            "regressor__max_depth": [2, 3]
        }
    ]

    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)

    grid_search = GridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        scoring = "neg_mean_absolute_error",
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

    print("Best RFS model:")
    print(grid_search.best_estimator_)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nBest CV MAE:")
    print(round(-grid_search.best_score_, 4))

    selected_features = grid_search.best_estimator_.named_steps["feature_selection"].get_selected_features()

    print("\nSelected feature count:")
    print(len(selected_features))

    print("\nRequired features included:")
    print(all(feature in selected_features for feature in IMPORTANT_FEATURES))

    print(f"\nSaved tuning results to {RESULTS_PATH}")
    print(f"Saved best RFS model to {MODEL_PATH}")

if __name__ == "__main__":
    main()