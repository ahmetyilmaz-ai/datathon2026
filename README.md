# Datathon 2026 - Appointment No-Show Prediction

Hi! This repository contains my solution work for **Datathon 2026**. The project is about predicting whether a patient will miss an appointment. The target column is `label_noshow`, and the final output is a probability for each `appointment_id`.

I worked on this project like a competition notebook/codebase: trying different features, checking validation scores, tuning models, and blending predictions to get a better final submission.

## What this project does

The main idea is to build a machine learning model for appointment no-show prediction. The pipeline roughly does this:

1. Loads appointment, patient, clinic, and sample submission data.
2. Merges appointment rows with patient and clinic information.
3. Creates extra features from dates, SMS status, patient history, clinic load, waiting time, and distance.
4. Uses a time-based validation split with the last 45 days as validation.
5. Trains CatBoost models and checks the validation PR-AUC / Average Precision score.
6. Blends multiple model predictions and creates a submission file.

## Repository structure

```text
.
├── main.py                         # Main CatBoost blend pipeline
├── main_optuna.py                  # Optuna + CatBoost GPU experiment
├── cold_start_backtest.py          # Pseudo-cold clinic validation experiment
├── export_compare_bridge.py        # Exports validation/test predictions for comparison
├── patients.csv                    # Patient information
├── clinics.csv                     # Clinic information
├── sample_submission.csv           # Submission format
├── blend_metadata*.json            # Saved experiment / blend metadata
├── compare_bridge/                 # Exported comparison files
└── catboost_info/                  # CatBoost training outputs
```

Some input files like `appointments_train.csv` and `appointments_test.csv` are expected to be in the project root when running the scripts.

## Main files

### `main.py`

This is the main script I used for the final blended model. It trains several CatBoost models with slightly different settings and feature groups, then searches blend weights on the validation set.

The script saves:

- `submission_raw_blend.csv`
- `feature_importance_blend_models.csv`

### `main_optuna.py`

This script is for hyperparameter tuning with Optuna. It runs a CatBoost GPU search and then retrains the best model on the full training data.

It saves:

- `submission_optuna_best.csv`
- `feature_importance_optuna_best.csv`
- `optuna_trials.csv`

Note: this file uses `task_type="GPU"`. If you do not have a GPU, you may need to change it to CPU settings.

### `cold_start_backtest.py`

This script checks how the models behave when some clinics are treated like unseen clinics. I added this because a model can look good on normal validation but still struggle with cold-start clinic cases.

### `export_compare_bridge.py`

This is a small helper script for exporting predictions in a cleaner format so I can compare results with other experiments.

## Features used

Some of the features I tried include:

- appointment month, day, week, and hour-based buckets
- booking month, booking hour, and booking day of week
- SMS lead time and missing SMS indicators
- prior appointment count, prior no-show count, and prior no-show rate
- clinic capacity / load based features
- estimated waiting time interactions
- distance between patient residence and clinic using the haversine formula
- categorical features like clinic, specialty, booking channel, appointment type, sex, and area

## How to run

First, clone the repository:

```bash
git clone https://github.com/ahmetyilmaz-ai/datathon2026.git
cd datathon2026
```

Create an environment and install the main dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn catboost optuna
```

Make sure these data files are in the root folder:

```text
appointments_train.csv
appointments_test.csv
patients.csv
clinics.csv
sample_submission.csv
```

Run the main blend pipeline:

```bash
python main.py
```

Run the Optuna experiment if you want to tune CatBoost:

```bash
python main_optuna.py
```

Run the cold-start experiment:

```bash
python cold_start_backtest.py
```

## Validation approach

I used a time-based split instead of a random split. The last 45 days are used as validation. I think this is more realistic because in real usage we usually train on past appointments and predict future appointments.

The main metric used in the scripts is **Average Precision / PR-AUC**, which is useful for this task because no-show prediction can be imbalanced.

## Results and experiments

The repository includes metadata files from my experiments. Some of the saved results are around **0.489 PR-AUC** on out-of-fold / validation style checks. I also tried different model combinations, including clinic-aware and clinic-agnostic versions, because clinic information can be helpful but may also overfit in cold-start cases.

## Things I would improve next

This project is still a work in progress. If I had more time, I would improve these parts:

- add a `requirements.txt` file
- clean hard-coded local paths in helper scripts
- make all scripts use one shared config file
- add more cross-validation experiments
- test probability calibration
- compare CatBoost with more LightGBM/XGBoost models
- write a cleaner final training notebook

## Final note

This repo is not a perfect production project; it is more like my competition workspace. I kept the experiments and helper files because they show how I tried to improve the model step by step.