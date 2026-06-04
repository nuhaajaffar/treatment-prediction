# Breast Cancer Treatment Response Prediction

This project implements an end-to-end machine learning pipeline for predicting breast cancer treatment response using clinical and MRI-derived radiomics features.

The work focuses on two prediction tasks:

* **pCR prediction**: a binary classification task predicting whether a patient achieves pathological complete response.
* **RFS prediction**: a regression task predicting relapse-free survival time.

The project covers data preprocessing, missing value handling, feature selection, model comparison, hyperparameter tuning and prediction generation using reproducible Scikit-learn pipelines.

## Project Overview

Breast cancer patients can respond differently to chemotherapy. Predicting treatment response before treatment may help support better patient stratification and clinical decision-making.

This project uses tabular clinical features and MRI-derived radiomics features to train machine learning models for early treatment response prediction. The dataset contains clinical biomarkers, tumour-related measurements and radiomics descriptors extracted from MRI scans.

The main goal is not to claim clinical readiness, but to build a structured and reproducible machine learning workflow for healthcare-related prediction tasks.

## Objectives

* Build a complete machine learning pipeline for pCR classification and RFS regression.
* Handle missing values encoded as `999`.
* Apply imputation and feature scaling.
* Perform feature selection while retaining clinically important features.
* Compare multiple machine learning models.
* Tune the best-performing models using cross-validation.
* Generate prediction CSV files for unseen test data.

## Dataset

The dataset contains:

* 11 clinical features
* 107 MRI-derived radiomics features
* pCR outcome label for classification
* RFS outcome value for regression

Missing values are encoded as `999` and are replaced with missing values during preprocessing.

The dataset files are not included in this repository because they may be subject to access restrictions. To run the project locally, place the required Excel files inside the `data/` folder.

Expected local files:

```text
data/
├── TrainDataset2025.xls
├── TestDatasetExample.xls
└── FinalTestDataset2025.xls
```

## Methodology

### 1. Data Preprocessing

The preprocessing stage includes:

* Loading Excel datasets
* Replacing `999` values with missing values
* Removing rows with missing pCR labels for classification
* Separating features and target variables
* Median imputation for missing feature values
* Standard scaling for numerical features

### 2. Feature Selection

Feature selection is applied to reduce dimensionality and limit overfitting.

The project uses a custom feature selector that selects the top features using statistical scoring while ensuring that the following clinically important features are always retained:

* `ER`
* `HER2`
* `Gene`

This ensures that clinically relevant predictors remain part of the modelling process.

### 3. Model Development

Two machine learning tasks are implemented.

#### pCR Classification

Models compared:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* Support Vector Machine
* Extra Trees Classifier

Final selected model:

* **Support Vector Machine with RBF kernel and mandatory feature selection**

Best tuned configuration:

```text
Model: SVM
Selected features: 25
C: 0.3
Gamma: 0.01
CV Balanced Accuracy: 0.6608
```

#### RFS Regression

Models compared:

* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* Support Vector Regressor

Final selected model:

* **Random Forest Regressor with mandatory feature selection**

Best tuned configuration:

```text
Model: Random Forest Regressor
Selected features: 20
n_estimators: 200
min_samples_leaf: 3
CV MAE: 20.6832
```

## Results Summary

### pCR Classification

| Model               | CV Balanced Accuracy |
| ------------------- | -------------------: |
| SVM                 |               0.6760 |
| Logistic Regression |               0.6643 |
| Extra Trees         |               0.6253 |
| Random Forest       |               0.5693 |
| Gradient Boosting   |               0.5501 |

After feature selection and hyperparameter tuning, the final pCR model achieved:

| Final pCR Model         | CV Balanced Accuracy |
| ----------------------- | -------------------: |
| SVM + Feature Selection |               0.6608 |

### RFS Regression

| Final RFS Model                             |  CV MAE |
| ------------------------------------------- | ------: |
| Random Forest Regressor + Feature Selection | 20.6832 |

RFS prediction remained more challenging than pCR classification. The regression results suggest that relapse-free survival may depend on additional biological, treatment-related or follow-up factors that are not fully captured by pre-treatment clinical and radiomics features alone.

## Project Structure

```text
treatment-prediction/
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── outputs/
│   └── README.md
│
├── src/
│   ├── explore_data.py
│   ├── preprocessing.py
│   ├── test_preprocessing.py
│   ├── baseline_pcr.py
│   ├── feature_selection.py
│   ├── pcr_feature_selection.py
│   ├── rfs_feature_selection.py
│   ├── pcr_models.py
│   ├── rfs_models.py
│   ├── tune_pcr.py
│   ├── tune_rfs.py
│   ├── predict_pcr.py
│   └── predict_rfs.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/nuhaajaffar/treatment-prediction.git
cd treatment-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the dataset files

Place the dataset files in the `data/` folder:

```text
data/TrainDataset2025.xls
data/TestDatasetExample.xls
data/FinalTestDataset2025.xls
```

### 6. Run data exploration

```bash
python src/explore_data.py
```

### 7. Run model comparison

```bash
python src/pcr_models.py
python src/rfs_models.py
```

### 8. Run hyperparameter tuning

```bash
python src/tune_pcr.py
python src/tune_rfs.py
```

### 9. Generate predictions

```bash
python src/predict_pcr.py
python src/predict_rfs.py
```

The prediction files will be saved in the `outputs/` folder.

Example output files:

```text
outputs/pcr_prediction.csv
outputs/rfs_prediction.csv
```

## Data Availability

The dataset files are not included in this repository because they may be subject to access restrictions.

To run the project locally, place the required Excel files inside the `data/` folder using the following names:

```text
data/
├── TrainDataset2025.xls
├── TestDatasetExample.xls
└── FinalTestDataset2025.xls
```

The code expects the training dataset to contain the feature columns, `pCR (outcome)` and `RelapseFreeSurvival (outcome)`.
The test datasets should contain the same feature columns and patient `ID`, but should not contain the outcome columns.

If the original dataset is unavailable, the repository can still be reviewed for its machine learning pipeline, code structure and methodology, but the scripts will not run end-to-end without compatible data files.

## Important Running Note

Run the scripts from the **project root directory** using the terminal.

Example:

```bash
python src/tune_pcr.py
python src/tune_rfs.py
python src/predict_pcr.py
python src/predict_rfs.py
```

On Windows, `py` can also be used:

```bash
py src/tune_pcr.py
py src/tune_rfs.py
py src/predict_pcr.py
py src/predict_rfs.py
```

Avoid running individual files using the editor “Run” button if it changes the working directory to `src/`. Some scripts use paths such as `data/TrainDataset2025.xls`, which are relative to the project root. Running from the wrong directory may cause file path errors.

## Skills Demonstrated

* Python
* Pandas
* NumPy
* Scikit-learn
* Data preprocessing
* Missing value handling
* Feature scaling
* Feature selection
* Classification
* Regression
* Cross-validation
* Hyperparameter tuning
* Model evaluation
* Healthcare AI
* Radiomics-based machine learning
* Reproducible ML workflow

## Key Takeaways

The pCR classification task showed moderate predictive performance, with the tuned SVM model achieving the best balanced accuracy. This suggests that the selected clinical and radiomics features contain useful signal for treatment response prediction.

The RFS regression task was more difficult, with limited predictive strength. This is expected because relapse-free survival is a long-term outcome that may depend on factors beyond pre-treatment imaging and clinical variables.

Overall, the project demonstrates a complete and reproducible machine learning workflow for healthcare-related prediction using tabular clinical and radiomics data.

## Limitations

* The dataset is relatively small for high-dimensional radiomics modelling.
* RFS prediction showed limited explanatory power.
* The models are not externally validated on an independent public dataset.
* The project is intended as a machine learning case study, not a clinically deployable system.

## Future Improvements

Potential improvements include:

* Adding stronger baseline comparisons.
* Testing repeated cross-validation.
* Applying more robust outlier handling.
* Exploring survival-specific models.
* Comparing clinical-only, radiomics-only and combined feature sets.
* Performing feature importance analysis for better interpretability.
* Validating the approach on an external dataset.

## Disclaimer

This project is for educational and portfolio purposes only. It is not intended for clinical diagnosis, treatment planning or medical decision-making.
