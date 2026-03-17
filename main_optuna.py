"""
CatBoost Hyperparameter Optimization with Optuna (GPU)
======================================================
30-trial Bayesian search  →  full retrain  →  submission.csv
"""

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

# =========================================================
# CONFIG
# =========================================================
TRAIN_PATH = "appointments_train.csv"
TEST_PATH = "appointments_test.csv"
PATIENTS_PATH = "patients.csv"
CLINICS_PATH = "clinics.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"

TARGET = "label_noshow"
ID_COL = "appointment_id"
TIME_COL = "appointment_datetime"

VALID_DAYS = 45
RANDOM_SEED = 42
N_TRIALS = 30


# =========================================================
# LOAD
# =========================================================
def load_data():
    train = pd.read_csv(TRAIN_PATH, parse_dates=["appointment_datetime", "booking_datetime"])
    test = pd.read_csv(TEST_PATH, parse_dates=["appointment_datetime", "booking_datetime"])
    patients = pd.read_csv(PATIENTS_PATH)
    clinics = pd.read_csv(CLINICS_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    if "specialty" in clinics.columns:
        clinics = clinics.drop(columns=["specialty"])

    return train, test, patients, clinics, sample_sub


# =========================================================
# FEATURES
# =========================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_features(df):
    df = df.copy()

    df["appt_date_only"] = df["appointment_datetime"].dt.normalize()
    df["appt_month"] = df["appointment_datetime"].dt.month
    df["appt_day"] = df["appointment_datetime"].dt.day
    df["appt_week"] = df["appointment_datetime"].dt.isocalendar().week.astype(int)

    df["booking_month"] = df["booking_datetime"].dt.month
    df["booking_hour"] = df["booking_datetime"].dt.hour
    df["booking_dow"] = df["booking_datetime"].dt.dayofweek

    df["sms_lead_missing"] = df["sms_lead_hours"].isna().astype(int)
    df["sms_lead_hours_filled"] = df["sms_lead_hours"].fillna(-1)
    df["sms_possible_but_not_sent"] = ((df["has_phone"] == 1) & (df["sms_sent"] == 0)).astype(int)

    df["prior_show_count"] = (df["prior_appt_count"] - df["prior_noshow_count"]).clip(lower=0)
    df["has_prior_history"] = (df["prior_appt_count"] > 0).astype(int)
    df["had_prior_noshow"] = (df["prior_noshow_count"] > 0).astype(int)

    df["appt_to_capacity"] = df["clinic_day_appt_count"] / np.maximum(df["capacity_daily"], 1)
    df["appt_minus_capacity"] = df["clinic_day_appt_count"] - df["capacity_daily"]
    df["clinic_load_x_wait"] = df["clinic_load_ratio"] * df["wait_mins_est"]

    # --- new features ---
    df["distance_km"] = haversine(
        df["residence_lat"], df["residence_lon"],
        df["clinic_lat"], df["clinic_lon"]
    )
    df["lead_days"] = (df["appointment_datetime"] - df["booking_datetime"]).dt.days
    df["prior_noshow_rate"] = df["prior_noshow_count"] / df["prior_appt_count"].clip(lower=1)

    cat_cols = ["clinic_id", "specialty", "booking_channel", "appointment_type", "sex", "area_id"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def build_feature_lists(df):
    drop_cols = [
        TARGET, ID_COL, "patient_id", TIME_COL, "booking_datetime",
        "appt_date_only", "sms_lead_hours",
    ]
    features = [c for c in df.columns if c not in drop_cols]

    cat_features = ["clinic_id", "specialty", "booking_channel", "appointment_type", "sex", "area_id"]
    cat_features = [c for c in cat_features if c in features]
    return features, cat_features


# =========================================================
# SPLIT
# =========================================================
def make_time_split(df):
    max_date = df["appt_date_only"].max()
    valid_start = max_date - pd.Timedelta(days=VALID_DAYS - 1)

    train_part = df[df["appt_date_only"] < valid_start].copy()
    valid_part = df[df["appt_date_only"] >= valid_start].copy()

    print(f"Train period: {train_part[TIME_COL].min()} -> {train_part[TIME_COL].max()} | n={len(train_part):,}")
    print(f"Valid period: {valid_part[TIME_COL].min()} -> {valid_part[TIME_COL].max()} | n={len(valid_part):,}")
    print(f"Valid target rate: {valid_part[TARGET].mean():.4f}")

    return train_part, valid_part


# =========================================================
# OPTUNA OBJECTIVE
# =========================================================
def create_objective(train_pool, valid_pool, y_valid):
    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "depth": trial.suggest_int("depth", 6, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        }

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="PRAUC:type=Classic",
            task_type="GPU",
            devices="0",
            bootstrap_type="Bayesian",
            has_time=True,
            random_seed=RANDOM_SEED,
            od_type="Iter",
            od_wait=200,
            verbose=0,
            allow_writing_files=False,
            **params,
        )

        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        preds = model.predict_proba(valid_pool)[:, 1]
        ap = average_precision_score(y_valid, preds)

        best_iter = model.get_best_iteration()
        if best_iter is None or best_iter <= 0:
            best_iter = model.tree_count_

        trial.set_user_attr("best_iteration", int(best_iter))

        print(f"  Trial {trial.number:>2d} | AP={ap:.6f} | iter={best_iter} | "
              f"lr={params['learning_rate']:.4f} depth={params['depth']} "
              f"l2={params['l2_leaf_reg']:.2f} bag_t={params['bagging_temperature']:.3f}")

        return ap

    return objective


# =========================================================
# FULL RETRAIN
# =========================================================
def fit_full_model(train_df, test_df, best_params, best_iter):
    features, cat_features = build_feature_lists(train_df)

    train_df = train_df.sort_values(TIME_COL).reset_index(drop=True)
    test_df = test_df.sort_values(TIME_COL).reset_index(drop=True)

    train_pool = Pool(train_df[features], train_df[TARGET], cat_features=cat_features)
    test_pool = Pool(test_df[features], cat_features=cat_features)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        task_type="GPU",
        devices="0",
        iterations=best_iter,
        learning_rate=best_params["learning_rate"],
        depth=best_params["depth"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        bagging_temperature=best_params["bagging_temperature"],
        min_data_in_leaf=best_params["min_data_in_leaf"],
        bootstrap_type="Bayesian",
        has_time=True,
        random_seed=RANDOM_SEED,
        verbose=200,
        allow_writing_files=False,
    )

    model.fit(train_pool)
    test_pred = model.predict_proba(test_pool)[:, 1]
    return model, test_df, test_pred, train_pool, features


# =========================================================
# MAIN
# =========================================================
def main():
    # --- load ---
    train, test, patients, clinics, sample_sub = load_data()
    print("Raw shapes")
    print("train appointments:", train.shape)
    print("test appointments :", test.shape)
    print("patients          :", patients.shape)
    print("clinics           :", clinics.shape)
    print(f"Train no-show rate: {train[TARGET].mean():.4f}")

    train = train.merge(patients, on="patient_id", how="left", validate="m:1")
    train = train.merge(clinics, on="clinic_id", how="left", validate="m:1")
    test = test.merge(patients, on="patient_id", how="left", validate="m:1")
    test = test.merge(clinics, on="clinic_id", how="left", validate="m:1")

    # --- feature engineering ---
    train_fe = add_features(train)
    test_fe = add_features(test)
    features, cat_features = build_feature_lists(train_fe)

    # --- time split ---
    train_part, valid_part = make_time_split(train_fe)
    train_pool = Pool(train_part[features], train_part[TARGET], cat_features=cat_features)
    valid_pool = Pool(valid_part[features], valid_part[TARGET], cat_features=cat_features)
    y_valid = valid_part[TARGET].values

    # --- optuna ---
    print(f"\n{'='*60}")
    print(f"  OPTUNA SEARCH — {N_TRIALS} trials (CatBoost GPU)")
    print(f"{'='*60}")

    study = optuna.create_study(direction="maximize", study_name="catboost_prauc")
    study.optimize(create_objective(train_pool, valid_pool, y_valid), n_trials=N_TRIALS)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_ap = best_trial.value
    best_iter = best_trial.user_attrs.get("best_iteration", best_params["iterations"])

    print(f"\n{'='*60}")
    print(f"  BEST TRIAL #{best_trial.number}")
    print(f"  Val PR-AUC  : {best_ap:.6f}")
    print(f"  Best iter   : {best_iter}")
    print(f"  Params      : {best_params}")
    print(f"{'='*60}")

    # --- full retrain ---
    print("\n>>> Full retrain with best params on entire train set...")
    model, test_sorted, test_pred, full_train_pool, feat_list = fit_full_model(
        train_fe, test_fe, best_params, best_iter
    )

    # --- submission ---
    submission = sample_sub[[ID_COL]].merge(
        test_sorted[[ID_COL]].assign(label_noshow=test_pred),
        on=ID_COL,
        how="left",
        validate="1:1",
    )
    submission["label_noshow"] = submission["label_noshow"].clip(0, 1)
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved: submission.csv  ({len(submission)} rows)")
    print(submission.head())

    # --- feature importance ---
    fi = pd.DataFrame({
        "feature": feat_list,
        "importance": model.get_feature_importance(full_train_pool),
    }).sort_values("importance", ascending=False)
    fi.to_csv("feature_importance_optuna_best.csv", index=False)
    print("\nTop-10 features:")
    print(fi.head(10).to_string(index=False))

    # --- save study results ---
    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_trials.csv", index=False)
    print(f"\nAll {N_TRIALS} trial results saved: optuna_trials.csv")


if __name__ == "__main__":
    main()
