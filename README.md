# 🚢 Titanic Survival Prediction

An end-to-end machine learning project predicting passenger survival on the Titanic, built with a fully custom feature engineering pipeline, compared across four different models, and deployed as an interactive Streamlit web app.

## 🌐 Live Demo

**Try it here:** [https://titanicsurvivalprediction1234.streamlit.app/](https://titanicsurvivalprediction1234.streamlit.app/)

Enter a passenger's details (class, age, fare, family size, etc.) and get a live survival prediction from the trained Decision Tree model.

---

## 📁 Project Structure

```
Titanic_Project/
│
├── Data/                       # Raw datasets (gitignored — not pushed to GitHub)
│   ├── train (2).csv           # Original Kaggle training data
│   ├── test (2).csv            # Original Kaggle test data
│   └── Finalprediction.csv     # Generated Kaggle submission file
│
├── Reports/
│   └── raw_report.html         # Exploratory data profiling report
│
├── model_exploration.ipynb     # Notebook: EDA, feature engineering iteration,
│                                # model comparison, hyperparameter tuning
├── final_pipeline.py           # Clean, final training script — builds the
│                                # feature-engineered pipeline, trains the
│                                # final model, and saves it with joblib
├── transformer.py              # Custom TitanicTransformer class used inside
│                                # the sklearn pipeline (feature engineering
│                                # logic, picklable for deployment)
├── app.py                      # Streamlit app — loads the saved model and
│                                # serves the interactive prediction UI
├── tree_pipeline.joblib         # Serialized final trained pipeline (model +
│                                # preprocessing + feature engineering, all in one)
│
├── pyproject.toml               # Project dependencies (for uv)
├── uv.lock                      # Locked dependency versions
├── .python-version              # Python version pin (3.12)
├── .gitignore                   # Excludes Data/ folder from version control
└── README.md                    # This file
```

> **Note on data:** The `Data/` folder (raw CSVs, predictions) is excluded via `.gitignore` and is **not** pushed to GitHub. To run this project locally, download `train.csv` and `test.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data) and place them in a local `Data/` folder.

---

## 🗂️ What Each File Does

### `model_exploration.ipynb`
The working notebook where all the experimentation happened:
- Exploratory data analysis on the raw Titanic dataset
- Iterative feature engineering (family size, marital status, cabin presence, age/fare brackets)
- Building and debugging the sklearn `Pipeline` + `ColumnTransformer`
- Training and cross-validating 4 different models
- Hyperparameter tuning with `RandomizedSearchCV` for each model
- Comparing CV scores vs. held-out test scores across all models

### `transformer.py`
Contains the `TitanicTransformer` class — a callable class (not a plain function) so it can be pickled and reloaded outside the training script. It encapsulates all feature engineering logic and holds the **fixed statistics computed only from the training set** (age median, embarked mode, fare median, fare quantile bin edges), so the exact same transformation is applied consistently to training data, test data, and any new single-passenger input at prediction time — with no data leakage.

### `final_pipeline.py`
The clean, production version of the pipeline:
1. Loads `train.csv`, splits into train/test
2. Computes fixed imputation/binning statistics from the training set only
3. Builds the full `Pipeline`: feature engineering → preprocessing (scaling, one-hot, ordinal encoding) → model
4. Fits the final chosen model (Decision Tree)
5. Predicts on Kaggle's `test.csv`
6. Saves the submission CSV and serializes the trained pipeline with `joblib` for deployment

### `app.py`
The Streamlit web app. Loads `tree_pipeline.joblib`, presents a form for entering passenger details, and returns a live survival prediction with a visual (GIF) result.

---

## 🧠 Feature Engineering

Beyond the raw Titanic columns, the following engineered features were built into the pipeline:

| Feature | Description |
|---|---|
| `Is_married` | Extracted from `Name` — flags passengers with the title "Mrs" |
| `Family_size` | `SibSp + Parch + 1`, bucketed into `Small` / `Large` |
| `Is_alone` | Binary flag for passengers traveling with no family aboard |
| `Age_bracket` | `Age` binned into `child` / `teen` / `adult` / `senior` |
| `Has_cabin` | Binary flag for whether the `Cabin` field was recorded |
| `Fare_bracket` | `Fare` binned into quartiles: `Low` / `Medium` / `High` / `Very High` |

**Preprocessing:**
- **Numeric columns** (`Pclass`, `Is_married`, `Is_alone`, `Has_cabin`) → `StandardScaler`
- **Nominal categorical** (`Sex`, `Embarked`) → `OneHotEncoder`
- **Ordinal categorical** (`Family_size`, `Age_bracket`, `Fare_bracket`) → `OrdinalEncoder` with explicitly defined category order

All missing-value imputation (`Age`, `Embarked`, `Fare`) and bin-edge calculations are computed **once from the training set** and reused at inference time — avoiding leakage and ensuring the pipeline works correctly on a single new passenger, not just batches.

---

## 🤖 Models Compared

Four models were trained through the identical feature engineering + preprocessing pipeline, each tuned with `RandomizedSearchCV`:

| Model | CV Accuracy | Local Test Accuracy | Kaggle Public Score |
|---|---|---|---|
| Logistic Regression | ~0.820 | ~0.795 | 0.76794 |
| **Decision Tree (final)** | **~0.830** | **~0.806** | **0.78229** ⭐ |
| Random Forest | ~0.830 | ~0.802 | 0.75837 |
| XGBoost | ~0.830 – 0.838 | ~0.787 – 0.791 | 0.76076 |

**Winner: Decision Tree** — best performer on both the local held-out test set and Kaggle's actual hidden test set, and was selected as the final deployed model.

**Final Decision Tree hyperparameters** (found via `RandomizedSearchCV`):
```python
DecisionTreeClassifier(
    max_depth=3,
    criterion="gini",
    min_samples_split=15,
    min_samples_leaf=5,
    random_state=42
)
```

### Notes on the results
- Ensembles (Random Forest, XGBoost) did **not** outperform a single well-tuned Decision Tree on Kaggle's leaderboard, despite similar or better cross-validation scores. This is a known effect on the Titanic dataset it's small (~891 rows) and heavily analyzed, so extra model complexity doesn't always translate into better generalization, and CV scores computed on a small training split don't always predict leaderboard performance perfectly.
- Every model showed a consistent **CV-score vs. test-score gap** (~2-5%), which reflects natural variance from evaluating on a relatively small hold-out set rather than any error in the pipeline.

---

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.12
- `uv` package manager (or `pip`)

### Setup
```bash
git clone <repository-url>
cd Titanic_Project

# Download train.csv and test.csv from Kaggle's Titanic competition
# and place them inside a local Data/ folder (not included in this repo)

uv sync
```

### Train the model
```bash
uv run python final_pipeline.py
```
This regenerates `tree_pipeline.joblib` and a Kaggle submission CSV.

### Run the Streamlit app
```bash
uv run streamlit run app.py
```
Then open the URL shown in your terminal (usually `http://localhost:8501`).

---

## 🛠️ Technologies Used

- **[scikit-learn](https://scikit-learn.org/)** — pipelines, preprocessing, models, cross-validation, hyperparameter search
- **[XGBoost](https://xgboost.readthedocs.io/)** — gradient-boosted tree model
- **[Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/)** — data manipulation
- **[Streamlit](https://streamlit.io/)** — interactive web app deployment
- **[joblib](https://joblib.readthedocs.io/)** — model serialization

---

## 📌 Key Learnings from This Project

- Building a leak-free sklearn `Pipeline`/`ColumnTransformer` from raw data to model, including custom feature engineering via a picklable transformer class
- Correctly distinguishing between cross-validation scores, held-out test scores, and true leaderboard performance
- Diagnosing overfitting through CV-vs-test score gaps (especially visible with XGBoost's deeper trees)
- Hyperparameter tuning with `RandomizedSearchCV` across linear, tree-based, and boosted models
- Serializing a full pipeline (feature engineering + preprocessing + model) for deployment, including handling the "closures aren't picklable" limitation with a custom class
- Deploying a trained pipeline behind a live Streamlit interface

---

## 👤 Author

Muhammad Awais Tariq

## 📚 References

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

If you found this project useful, consider giving it a ⭐