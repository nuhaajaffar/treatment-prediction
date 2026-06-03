import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from pathlib import Path
from preprocessing import load_data, prepare_rfs_data, build_preprocessing_pipeline

TRAIN_PATH = "data/TrainDataset2025.xls"
RESULTS_PATH = "outputs/rfs_model_comparison.csv"

def build_models():
    return {
        "Ridge Regression": Ridge(alpha = 1.0),
        "Random Forest": RandomForestRegressor(n_estimators = 200, random_state = 42),
        "Gradient Boosting": GradientBoostingRegressor(random_state = 42),
        "SVR": SVR(kernel = "rbf")
    }

def main():
    train_df = load_data(TRAIN_PATH)
    X, y = prepare_rfs_data(train_df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 42
    )

    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    results = []

    for model_name, regressor in build_models().items():
        pipeline = Pipeline([
            ("preprocessing", build_preprocessing_pipeline()),
            ("regressor", regressor)
        ])

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv = cv,
            scoring = "neg_mean_absolute_error"
        )

        cv_mae_scores = -cv_scores

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        val_mae = mean_absolute_error(y_val, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        val_r2 = r2_score(y_val, y_pred)

        results.append({
            "model": model_name,
            "cv_mae_mean": cv_mae_scores.mean(),
            "cv_mae_std": cv_mae_scores.std(),
            "validation_mae": val_mae,
            "validation_rmse": val_rmse,
            "validation_r2": val_r2
        })

        print("\n" + "=" * 50)
        print(model_name)
        print("CV MAE:", round(cv_mae_scores.mean(), 4), "+/-", round(cv_mae_scores.std(), 4))
        print("Validation MAE:", round(val_mae, 4))
        print("Validation RMSE:", round(val_rmse, 4))
        print("Validation R2:", round(val_r2, 4))

    results_df = pd.DataFrame(results).sort_values(
        by = "validation_mae",
        ascending = True
    )

    Path("outputs").mkdir(exist_ok = True)
    results_df.to_csv(RESULTS_PATH, index = False)

    print("\nFinal RFS model comparison:")
    print(results_df)
    print(f"\nSaved comparison results to {RESULTS_PATH}")

if __name__ == "__main__":
    main()