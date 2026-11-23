# HomeValue AI – Smart House Price Estimator

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b)](https://streamlit.io/) [![scikit--learn](https://img.shields.io/badge/ML-scikit--learn-f7931e)](https://scikit-learn.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An educational regression project that estimates historical residential sale prices in **Ames, Iowa, USA**, in **USD**. It compares Linear Regression, Decision Tree, and Random Forest using one held-out test set and chooses the model with the lowest RMSE. Predictions are not appraisals or guaranteed market prices.

## Objectives and features

- Learn a practical regression workflow, preprocessing pipelines, and test-set evaluation.
- Estimate a price from living area, bedrooms, bathrooms, floors/areas, quality, condition, year, garage, lot, and neighborhood.
- Explore price distributions, property relationships, neighborhood averages, and model metrics.
- Save preprocessing and estimator together so the app applies the same transformations used for training.

## Technology

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Joblib, Streamlit, and Jupyter Notebook.

## Dataset

The source is the [Kaggle House Prices: Advanced Regression Techniques competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), fetched from its public [OpenML mirror (dataset 42165)](https://www.openml.org/search?id=42165&status=active&type=data). The Kaggle training data describes **1,460 house sales** in Ames, Iowa from 2006–2010, with **79 explanatory variables** and `SalePrice` as the target (plus an identifier). Each price is nominal USD. The script downloads the data on first run; raw data and trained model artifacts are excluded from Git.

Important selected columns: `GrLivArea` (above-ground living area, sq ft), `BedroomAbvGr`, `FullBath`, `HalfBath`, `TotRmsAbvGrd`, `GarageCars`, `YearBuilt`, `OverallQual`, `OverallCond`, `LotArea`, `1stFlrSF`, `2ndFlrSF`, `Neighborhood`, `HouseStyle`, and `ExterQual`. Missing values occur in optional fields; numeric values are median-imputed and categorical values most-frequent-imputed. Duplicates are removed. We do not discard high sale prices as outliers automatically because they may represent real homes; the dataset's unusually large values can still influence error metrics.

## Architecture

```text
data/train.csv (downloaded)
        ↓
src/preprocessing.py → imputation + scaling + one-hot encoding
        ↓
src/train_model.py → 80/20 split → Linear / Tree / Forest → test metrics
        ↓
models/best_model.joblib + reports/model_metrics.*
        ↓
app.py → Home / Price Estimator / Analytics / About
```

- `src/preprocessing.py`: dataset download, selected features, reusable preprocessing.
- `src/train_model.py`: reproducible split, trains and compares three regressors, saves best pipeline and metrics.
- `src/evaluate_model.py`: displays saved metrics.
- `src/predict.py`: minimal programmatic prediction example.
- `app.py`: Streamlit application and analytics.
- `notebooks/exploratory_analysis.ipynb`: guided EDA and model training walkthrough.

## Machine learning workflow

1. Download and inspect the Ames dataset.
2. Drop exact duplicate rows and records without a valid positive target.
3. Split 80% for training and 20% for final evaluation (`random_state=42`).
4. Fit preprocessing only within each model's training pipeline, preventing test-data leakage.
5. Compare MAE, MSE, RMSE, and R² on the same test records. Prices remain in original USD for evaluation; no target log transform is used.
6. Select lowest RMSE, save its full pipeline, and load it in Streamlit without retraining.

Linear Regression is a simple baseline that estimates additive feature effects. A Decision Tree learns feature-based splits, allowing nonlinear patterns. Random Forest averages many trees to reduce the instability of a single tree.

### Evaluation results

Run the training command below to generate actual metrics in `reports/model_metrics.csv` and view them in the app. **No scores are hardcoded in this README:** results depend on execution and are reported by the script after it evaluates the test split. MAE is the typical absolute miss in dollars; MSE is squared error; RMSE is in dollars and penalizes large misses more; R² compares against predicting the test-set mean. A score is not a guarantee for an individual house.

## Setup and run

```bash
cd "E:/Learn/project 3"
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.train_model
streamlit run app.py
```

The first training run needs an internet connection to retrieve the OpenML dataset. The application needs the generated model. Dataset and artifacts are kept local and are not committed. To show saved metrics in a terminal, run `python -m src.evaluate_model`.

## Notebook

After downloading data with `python -m src.train_model`, open `notebooks/exploratory_analysis.ipynb` with Jupyter Notebook or VS Code. It covers target and living-area histograms/scatter, bedroom and bathroom comparisons, neighborhood averages, numerical correlations, a price box plot, data quality, model comparison, and the reasoning behind each plot.

## Screenshots

Suggested GitHub screenshots: Home page, completed price estimate with disclaimer, Analytics dashboard, and held-out model comparison. Save your own screenshots in a `screenshots/` folder after running locally.

## Limitations and disclaimer

The training data is small, historical, and limited to Ames, Iowa. Neighborhood labels and property characteristics are specific to this dataset. This is an educational model, not a valuation service, appraisal, financial recommendation, or statement of current market prices. Never interpret its output as a guaranteed sale price. Validate with current local data before practical use.

## Future improvements

- Add a date-aware split and compare prices adjusted for inflation.
- Tune models with cross-validation, inspect residuals, and quantify prediction intervals.
- Add a carefully maintained feature dictionary and validate user input combinations.
- Evaluate on recent, representative local data before expanding geography.

## License and attribution

MIT license for this project code. Dataset attribution: Kaggle House Prices competition; data originally documented by Dean De Cock, “Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project,” *Journal of Statistics Education* (2011). Dataset terms remain those of the source.
"# House-Price-Prediction" 
