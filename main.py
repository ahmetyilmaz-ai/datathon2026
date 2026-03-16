import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

# =========================================================
# PATHS / CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "appointments_train.csv"
TEST_PATH = BASE_DIR / "appointments_test.csv"
PATIENTS_PATH = BASE_DIR / "patients.csv"
CLINICS_PATH = BASE_DIR / "clinics.csv"
SAMPLE_SUB_PATH = BASE_DIR / "sample_submission.csv"

TARGET = "label_noshow"
ID_COL = "appointment_id"
TIME_COL = "appointment_datetime"
BOOKING_TIME_COL = "booking_datetime"
DATE_COL = "appt_date_only"
RANDOM_SEED = 42

ROLLING_VALID_DAYS = 28
N_FOLDS = 4
ROLLING_STEP_DAYS = 28
TE_SMOOTH = 25.0

OUTPUT_SUB = BASE_DIR / "submission_improved_stacked.csv"
OUTPUT_OOF = BASE_DIR / "oof_base_predictions.csv"
OUTPUT_CV = BASE_DIR / "cv_summary.csv"
OUTPUT_IMPORTANCE = BASE_DIR / "feature_importance_full_models.csv"
OUTPUT_META = BASE_DIR / "blend_metadata.json"

MODEL_CONFIGS = [
    {
        "name": "clinic_aware",
        "use_clinic_id": True,
        "params": {
            "iterations": 4000,
            "learning_rate": 0.015,
            "depth": 6,
            "l2_leaf_reg": 25,
            "min_data_in_leaf": 40,
            "random_strength": 1.5,
            "subsample": 0.80,
        },
    },
    {
        "name": "clinic_agnostic",
        "use_clinic_id": False,
        "params": {
            "iterations": 4000,
            "learning_rate": 0.015,
            "depth": 6,
            "l2_leaf_reg": 25,
            "min_data_in_leaf": 60,
            "random_strength": 1.5,
            "subsample": 0.78,
        },
    },
]

# =========================================================
# LOAD / MERGE
# =========================================================
def load_data():
    train = pd.read_csv(TRAIN_PATH, parse_dates=[TIME_COL, BOOKING_TIME_COL])
    test = pd.read_csv(TEST_PATH, parse_dates=[TIME_COL, BOOKING_TIME_COL])
    patients = pd.read_csv(PATIENTS_PATH)
    clinics = pd.read_csv(CLINICS_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    if "specialty" in clinics.columns:
        clinics = clinics.drop(columns=["specialty"])

    train = train.merge(patients, on="patient_id", how="left", validate="m:1")
    train = train.merge(clinics, on="clinic_id", how="left", validate="m:1")

    test = test.merge(patients, on="patient_id", how="left", validate="m:1")
    test = test.merge(clinics, on="clinic_id", how="left", validate="m:1")

    train = train.sort_values([TIME_COL, ID_COL]).reset_index(drop=True)
    test = test.sort_values([TIME_COL, ID_COL]).reset_index(drop=True)
    return train, test, sample_sub


# =========================================================
# LEAKAGE-SAFE TEMPORAL PATIENT HISTORY
# - 2025 cumulative appointment count
# - 2025 cumulative known no-show count (train labels only)
# - days since previous appointment
# =========================================================
def add_temporal_patient_history(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["_is_train"] = 1
    test_df["_is_train"] = 0

    if TARGET not in test_df.columns:
        test_df[TARGET] = np.nan

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=False)
    combined = combined.sort_values(["patient_id", TIME_COL, ID_COL]).reset_index(drop=True)

    # 1) Cumulative 2025 appointment count: strictly previous rows only
    combined["patient_2025_appt_count"] = combined.groupby("patient_id").cumcount()

    # 2) Cumulative 2025 no-show count:
    # Only TRAIN rows contribute label information.
    # TEST labels are unknown, so they contribute 0.
    combined["_known_noshow"] = np.where(
        combined["_is_train"] == 1,
        combined[TARGET].fillna(0),
        0,
    )

    combined["patient_2025_noshow_count"] = (
        combined.groupby("patient_id")["_known_noshow"].cumsum() - combined["_known_noshow"]
    )

    # 3) Recency: days since last appointment, only previous appointment per patient
    combined["prev_appt_datetime"] = combined.groupby("patient_id")[TIME_COL].shift(1)
    combined["days_since_last_appt"] = (
        (combined[TIME_COL] - combined["prev_appt_datetime"]).dt.total_seconds() / 86400.0
    )
    combined["days_since_last_appt"] = combined["days_since_last_appt"].fillna(-1.0)

    combined["patient_2025_appt_count"] = combined["patient_2025_appt_count"].astype(np.int32)
    combined["patient_2025_noshow_count"] = combined["patient_2025_noshow_count"].astype(np.int32)
    combined["days_since_last_appt"] = combined["days_since_last_appt"].astype(float)

    train_out = combined.loc[combined["_is_train"] == 1].drop(
        columns=["_is_train", "_known_noshow", "prev_appt_datetime"]
    )
    test_out = combined.loc[combined["_is_train"] == 0].drop(
        columns=["_is_train", "_known_noshow", "prev_appt_datetime", TARGET]
    )

    train_out = train_out.sort_values([TIME_COL, ID_COL]).reset_index(drop=True)
    test_out = test_out.sort_values([TIME_COL, ID_COL]).reset_index(drop=True)
    return train_out, test_out


# =========================================================
# FEATURE ENGINEERING
# =========================================================
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def add_features(df):
    df = df.copy()

    # ------------------------------
    # datetime features
    # ------------------------------
    df[DATE_COL] = df[TIME_COL].dt.normalize()
    df["appt_month"] = df[TIME_COL].dt.month
    df["appt_day"] = df[TIME_COL].dt.day
    df["appt_week"] = df[TIME_COL].dt.isocalendar().week.astype(int)
    df["appt_quarter"] = df[TIME_COL].dt.quarter
    df["is_month_start"] = df[TIME_COL].dt.is_month_start.astype(int)
    df["is_month_end"] = df[TIME_COL].dt.is_month_end.astype(int)

    df["booking_month"] = df[BOOKING_TIME_COL].dt.month
    df["booking_hour"] = df[BOOKING_TIME_COL].dt.hour
    df["booking_dow"] = df[BOOKING_TIME_COL].dt.dayofweek

    df["appointment_hour_sin"] = np.sin(2 * np.pi * df["appointment_hour"] / 24.0)
    df["appointment_hour_cos"] = np.cos(2 * np.pi * df["appointment_hour"] / 24.0)
    df["appointment_dow_sin"] = np.sin(2 * np.pi * df["appointment_dow"] / 7.0)
    df["appointment_dow_cos"] = np.cos(2 * np.pi * df["appointment_dow"] / 7.0)

    # ------------------------------
    # lead / booking features
    # ------------------------------
    df["lead_time_days"] = df["lead_time_hours"] / 24.0
    df["lead_time_log1p"] = np.log1p(df["lead_time_hours"].clip(lower=0))
    df["same_day_booking"] = (df["lead_time_hours"] <= 24).astype(int)
    df["very_short_lead"] = (df["lead_time_hours"] <= 6).astype(int)
    df["long_lead"] = (df["lead_time_hours"] >= 24 * 7).astype(int)
    df["very_long_lead"] = (df["lead_time_hours"] >= 24 * 14).astype(int)

    # ------------------------------
    # sms / phone features
    # ------------------------------
    df["sms_lead_missing"] = df["sms_lead_hours"].isna().astype(int)
    df["sms_lead_hours_filled"] = df["sms_lead_hours"].fillna(-1)
    df["sms_possible_but_not_sent"] = ((df["has_phone"] == 1) & (df["sms_sent"] == 0)).astype(int)
    df["sms_sent_x_has_phone"] = (df["sms_sent"] * df["has_phone"]).astype(int)

    # ------------------------------
    # historical patient features (pre-2025 from patients.csv)
    # ------------------------------
    df["prior_show_count"] = (df["prior_appt_count"] - df["prior_noshow_count"]).clip(lower=0)
    df["has_prior_history"] = (df["prior_appt_count"] > 0).astype(int)
    df["had_prior_noshow"] = (df["prior_noshow_count"] > 0).astype(int)

    df["prior_noshow_rate_safe"] = np.where(
        df["prior_appt_count"] > 0,
        df["prior_noshow_count"] / np.maximum(df["prior_appt_count"], 1),
        0.0,
    )
    df["prior_noshow_ratio_safe"] = df["prior_noshow_rate_safe"]

    # ------------------------------
    # 2025 cumulative patient history
    # ------------------------------
    df["patient_2025_has_history"] = (df["patient_2025_appt_count"] > 0).astype(int)
    df["patient_2025_show_count"] = (
        df["patient_2025_appt_count"] - df["patient_2025_noshow_count"]
    ).clip(lower=0)

    # İstenen dinamik oran
    df["patient_2025_noshow_rate"] = (
        df["patient_2025_noshow_count"] / np.maximum(df["patient_2025_appt_count"], 1)
    )
    df["patient_2025_noshow_rate_safe"] = df["patient_2025_noshow_rate"]

    # Recency
    df["has_prev_appt_2025"] = (df["days_since_last_appt"] >= 0).astype(int)
    df["days_since_last_appt_log1p"] = np.where(
        df["days_since_last_appt"] >= 0,
        np.log1p(df["days_since_last_appt"]),
        -1.0,
    )
    df["recent_return_7d"] = (
        (df["days_since_last_appt"] >= 0) & (df["days_since_last_appt"] <= 7)
    ).astype(int)
    df["recent_return_30d"] = (
        (df["days_since_last_appt"] >= 0) & (df["days_since_last_appt"] <= 30)
    ).astype(int)

    # ------------------------------
    # strong numeric interactions
    # ------------------------------
    df["prior_noshow_rate_squared"] = df["prior_noshow_rate_safe"] ** 2
    df["patient_age_x_chronic"] = df["age"] * df["chronic_count"]
    df["distance_x_age"] = df["distance_km"] * df["age"]
    df["ses_x_has_phone"] = df["ses_score"] * df["has_phone"]

    df["lead_time_x_no_sms"] = df["lead_time_days"] * (1 - df["sms_sent"])
    df["noshow_rate_x_lead_time"] = df["prior_noshow_rate_safe"] * df["lead_time_days"]
    df["distance_x_noshow_rate"] = df["distance_km"] * df["prior_noshow_rate_safe"]
    df["wait_time_x_noshow"] = df["wait_mins_est"] * df["prior_noshow_rate_safe"]

    # 2025 dynamic history interactions
    df["patient2025_noshow_x_lead"] = df["patient_2025_noshow_rate"] * df["lead_time_days"]
    df["patient2025_noshow_x_distance"] = df["patient_2025_noshow_rate"] * df["distance_km"]
    df["patient2025_noshow_x_wait"] = df["patient_2025_noshow_rate"] * df["wait_mins_est"]

    # Recency interactions
    df["recency_x_prior_noshow"] = df["days_since_last_appt_log1p"] * df["prior_noshow_rate_safe"]
    df["recency_x_2025_noshow"] = df["days_since_last_appt_log1p"] * df["patient_2025_noshow_rate"]
    df["recency_x_distance"] = df["days_since_last_appt_log1p"] * df["distance_km"]

    # ------------------------------
    # clinic / capacity features
    # ------------------------------
    df["appt_to_capacity"] = df["clinic_day_appt_count"] / np.maximum(df["capacity_daily"], 1)
    df["appt_minus_capacity"] = df["clinic_day_appt_count"] - df["capacity_daily"]
    df["clinic_load_x_wait"] = df["clinic_load_ratio"] * df["wait_mins_est"]
    df["wait_minus_base"] = df["wait_mins_est"] - df["base_wait_mins_est"]
    df["wait_to_base_ratio"] = df["wait_mins_est"] / np.maximum(df["base_wait_mins_est"], 1)
    df["capacity_x_open_sat"] = df["capacity_daily"] * (1 + df["open_on_saturday"])
    df["high_load_x_wait"] = (df["clinic_load_ratio"] > 1.0).astype(int) * df["wait_mins_est"]

    # ------------------------------
    # distance / geo features
    # ------------------------------
    df["distance_log1p"] = np.log1p(df["distance_km"].clip(lower=0))
    df["distance_x_lead"] = df["distance_km"] * df["lead_time_days"]
    df["distance_x_sms"] = df["distance_km"] * (1 + df["sms_sent"])

    df["geo_haversine_km"] = haversine_km(
        df["residence_lat"], df["residence_lon"], df["clinic_lat"], df["clinic_lon"]
    )
    df["distance_gap_vs_geo"] = df["distance_km"] - df["geo_haversine_km"]
    df["abs_lat_gap"] = (df["residence_lat"] - df["clinic_lat"]).abs()
    df["abs_lon_gap"] = (df["residence_lon"] - df["clinic_lon"]).abs()

    # ------------------------------
    # buckets
    # ------------------------------
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[-1, 17, 29, 44, 59, 74, 120],
        labels=["0_17", "18_29", "30_44", "45_59", "60_74", "75_plus"],
    ).astype(str)

    df["lead_time_bucket"] = pd.cut(
        df["lead_time_hours"],
        bins=[-1, 6, 24, 72, 168, 336, 24 * 60],
        labels=["lt_6h", "6_24h", "1_3d", "3_7d", "7_14d", "14d_plus"],
    ).astype(str)

    df["hour_bucket"] = pd.cut(
        df["appointment_hour"],
        bins=[-1, 8, 11, 14, 17, 23],
        labels=["very_early", "morning", "noon", "afternoon", "late"],
    ).astype(str)

    base_cats = [
        "clinic_id",
        "specialty",
        "booking_channel",
        "appointment_type",
        "sex",
        "area_id",
        "distance_bucket",
        "age_bucket",
        "lead_time_bucket",
        "hour_bucket",
    ]

    for col in base_cats:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ------------------------------
    # crossed categoricals
    # ------------------------------
    df["specialty_x_channel"] = df["specialty"].astype(str) + "__" + df["booking_channel"].astype(str)
    df["specialty_x_type"] = df["specialty"].astype(str) + "__" + df["appointment_type"].astype(str)
    df["specialty_x_hour"] = df["specialty"].astype(str) + "__" + df["hour_bucket"].astype(str)
    df["area_x_specialty"] = df["area_id"].astype(str) + "__" + df["specialty"].astype(str)
    df["area_x_channel"] = df["area_id"].astype(str) + "__" + df["booking_channel"].astype(str)
    df["clinic_x_hour"] = df["clinic_id"].astype(str) + "__" + df["hour_bucket"].astype(str)
    df["clinic_x_type"] = df["clinic_id"].astype(str) + "__" + df["appointment_type"].astype(str)

    return df


# =========================================================
# TARGET ENCODING
# =========================================================
def get_te_cols(use_clinic_id=True):
    te_cols = [
        "specialty",
        "booking_channel",
        "appointment_type",
        "distance_bucket",
        "lead_time_bucket",
        "hour_bucket",
        "specialty_x_channel",
        "specialty_x_type",
        "specialty_x_hour",
        "area_x_specialty",
        "area_x_channel",
    ]
    if use_clinic_id:
        te_cols += ["clinic_id", "clinic_x_hour", "clinic_x_type"]
    return te_cols


def apply_target_encoding(train_df, other_df, cols, target=TARGET, smooth=TE_SMOOTH):
    train_df = train_df.copy()
    other_df = other_df.copy()
    global_mean = train_df[target].mean()

    te_feature_names = []
    for col in cols:
        if col not in train_df.columns:
            continue

        stats = train_df.groupby(col, dropna=False)[target].agg(["sum", "count"])
        stats["te"] = (stats["sum"] + global_mean * smooth) / (stats["count"] + smooth)
        mapper = stats["te"]

        new_col = f"te_{col}"
        train_df[new_col] = train_df[col].map(mapper).fillna(global_mean).astype(float)
        other_df[new_col] = other_df[col].map(mapper).fillna(global_mean).astype(float)
        te_feature_names.append(new_col)

    return train_df, other_df, te_feature_names


# =========================================================
# FEATURE LISTS
# =========================================================
def build_feature_lists(df, use_clinic_id=True):
    drop_cols = [
        TARGET,
        ID_COL,
        "patient_id",
        TIME_COL,
        BOOKING_TIME_COL,
        DATE_COL,
        "residence_lat",
        "residence_lon",
        "clinic_lat",
        "clinic_lon",
        "sms_lead_hours",
        "is_cold_start_clinic",
    ]

    if not use_clinic_id:
        drop_cols.append("clinic_id")
        drop_cols += ["clinic_x_hour", "clinic_x_type"]

    features = [c for c in df.columns if c not in drop_cols]

    cat_features = [
        "specialty",
        "booking_channel",
        "appointment_type",
        "sex",
        "area_id",
        "distance_bucket",
        "age_bucket",
        "lead_time_bucket",
        "hour_bucket",
        "specialty_x_channel",
        "specialty_x_type",
        "specialty_x_hour",
        "area_x_specialty",
        "area_x_channel",
    ]
    if use_clinic_id:
        cat_features = ["clinic_id", "clinic_x_hour", "clinic_x_type"] + cat_features

    cat_features = [c for c in cat_features if c in features]
    return features, cat_features


# =========================================================
# ROLLING FOLDS
# =========================================================
def build_rolling_folds(df, n_folds=N_FOLDS, valid_days=ROLLING_VALID_DAYS, step_days=ROLLING_STEP_DAYS):
    max_date = df[DATE_COL].max()
    folds = []

    for i in range(n_folds):
        valid_end = max_date - pd.Timedelta(days=i * step_days)
        valid_start = valid_end - pd.Timedelta(days=valid_days - 1)

        train_mask = df[DATE_COL] < valid_start
        valid_mask = (df[DATE_COL] >= valid_start) & (df[DATE_COL] <= valid_end)

        tr_idx = df.index[train_mask].to_numpy()
        va_idx = df.index[valid_mask].to_numpy()

        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue

        folds.append(
            {
                "fold": i + 1,
                "train_idx": tr_idx,
                "valid_idx": va_idx,
                "valid_start": valid_start,
                "valid_end": valid_end,
            }
        )

    folds = list(reversed(folds))
    return folds


# =========================================================
# MODEL HELPERS
# =========================================================
def build_catboost(params, random_seed=RANDOM_SEED):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        min_data_in_leaf=params["min_data_in_leaf"],
        bootstrap_type="Bernoulli",
        subsample=params.get("subsample", 0.8),
        random_strength=params.get("random_strength", 1.0),
        has_time=True,
        random_seed=random_seed,
        od_type="Iter",
        od_wait=200,
        verbose=False,
        allow_writing_files=False,
    )


def train_predict_single_model(train_fold, valid_fold, cfg):
    te_cols = get_te_cols(use_clinic_id=cfg["use_clinic_id"])
    train_enc, valid_enc, _ = apply_target_encoding(train_fold, valid_fold, te_cols)
    features, cat_features = build_feature_lists(train_enc, use_clinic_id=cfg["use_clinic_id"])

    train_sorted = train_enc.sort_values(TIME_COL)
    valid_sorted = valid_enc.sort_values(TIME_COL)

    train_pool = Pool(train_sorted[features], train_sorted[TARGET], cat_features=cat_features)
    valid_pool = Pool(valid_sorted[features], valid_sorted[TARGET], cat_features=cat_features)

    model = build_catboost(cfg["params"])
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    preds = model.predict_proba(valid_pool)[:, 1]
    pred_s = pd.Series(preds, index=valid_sorted.index).sort_index()

    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter <= 0:
        best_iter = model.tree_count_

    return pred_s, int(best_iter), model, features, cat_features


# =========================================================
# OOF TRAINING
# =========================================================
def run_oof_training(train_df, folds, model_configs):
    oof_frame = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET: train_df[TARGET]})
    cv_rows = []
    model_artifacts = {}

    for cfg in model_configs:
        print(f"\n===== OOF training: {cfg['name']} =====")
        oof_pred = pd.Series(index=train_df.index, dtype=float)
        fold_scores = []
        best_iters = []

        for fold_info in folds:
            fold_no = fold_info["fold"]
            tr_idx = fold_info["train_idx"]
            va_idx = fold_info["valid_idx"]

            tr_fold = train_df.loc[tr_idx].copy()
            va_fold = train_df.loc[va_idx].copy()

            pred_s, best_iter, _, _, _ = train_predict_single_model(tr_fold, va_fold, cfg)
            best_iters.append(best_iter)

            oof_pred.loc[pred_s.index] = pred_s.values
            fold_ap = average_precision_score(va_fold[TARGET], pred_s.loc[va_fold.index])
            fold_scores.append(fold_ap)

            cv_rows.append(
                {
                    "model": cfg["name"],
                    "fold": fold_no,
                    "valid_start": str(fold_info["valid_start"].date()),
                    "valid_end": str(fold_info["valid_end"].date()),
                    "fold_ap": fold_ap,
                    "n_train": len(tr_fold),
                    "n_valid": len(va_fold),
                }
            )

            print(
                f"Fold {fold_no}: AP={fold_ap:.6f} | "
                f"valid={fold_info['valid_start'].date()} -> {fold_info['valid_end'].date()} | "
                f"n_train={len(tr_fold):,} n_valid={len(va_fold):,}"
            )

        overall_mask = oof_pred.notna()
        overall_ap = average_precision_score(train_df.loc[overall_mask, TARGET], oof_pred.loc[overall_mask])
        median_best_iter = int(np.median(best_iters)) if best_iters else cfg["params"]["iterations"]

        oof_frame[f"pred_{cfg['name']}"] = oof_pred
        model_artifacts[cfg["name"]] = {
            "config": cfg,
            "median_best_iter": median_best_iter,
            "oof_ap": overall_ap,
            "fold_ap_mean": float(np.mean(fold_scores)),
        }

        print(
            f"{cfg['name']} overall OOF AP={overall_ap:.6f} | "
            f"fold mean={np.mean(fold_scores):.6f} | median_best_iter={median_best_iter}"
        )

    cv_summary = pd.DataFrame(cv_rows)
    return oof_frame, cv_summary, model_artifacts


# =========================================================
# STACKERS
# =========================================================
def fit_stackers(oof_frame):
    base_cols_all = [
        "pred_clinic_aware",
        "pred_clinic_agnostic",
    ]
    base_cols_no_clinic = [
        "pred_clinic_agnostic",
    ]

    mask_all = oof_frame[base_cols_all].notna().all(axis=1)
    X_all = oof_frame.loc[mask_all, base_cols_all].values
    y_all = oof_frame.loc[mask_all, TARGET].values

    stacker_all = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")
    stacker_all.fit(X_all, y_all)

    X_no = oof_frame.loc[mask_all, base_cols_no_clinic].values
    stacker_no_clinic = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")
    stacker_no_clinic.fit(X_no, y_all)

    stacked_all_oof = stacker_all.predict_proba(X_all)[:, 1]
    stacked_no_oof = stacker_no_clinic.predict_proba(X_no)[:, 1]

    ap_all = average_precision_score(y_all, stacked_all_oof)
    ap_no = average_precision_score(y_all, stacked_no_oof)

    coef_all = dict(zip(base_cols_all, stacker_all.coef_[0]))
    coef_no = dict(zip(base_cols_no_clinic, stacker_no_clinic.coef_[0]))

    print(f"\nStacker(all models) OOF AP: {ap_all:.6f}")
    print(f"Stacker(no clinic-aware) OOF AP: {ap_no:.6f}")
    print("All-model stacker coefficients:", coef_all)
    print("No-clinic stacker coefficients:", coef_no)

    return {
        "stacker_all": stacker_all,
        "stacker_no_clinic": stacker_no_clinic,
        "base_cols_all": base_cols_all,
        "base_cols_no_clinic": base_cols_no_clinic,
        "oof_ap_all": ap_all,
        "oof_ap_no_clinic": ap_no,
        "coef_all": coef_all,
        "coef_no_clinic": coef_no,
    }


# =========================================================
# FULL TRAIN / TEST PREDICTIONS
# =========================================================
def fit_full_single_model(train_df, test_df, cfg, best_iter):
    te_cols = get_te_cols(use_clinic_id=cfg["use_clinic_id"])
    train_enc, test_enc, _ = apply_target_encoding(train_df, test_df, te_cols)
    features, cat_features = build_feature_lists(train_enc, use_clinic_id=cfg["use_clinic_id"])

    train_sorted = train_enc.sort_values(TIME_COL)
    test_sorted = test_enc.sort_values(TIME_COL)

    model_params = dict(cfg["params"])
    model_params["iterations"] = max(best_iter, 200)
    model = build_catboost(model_params)

    train_pool = Pool(train_sorted[features], train_sorted[TARGET], cat_features=cat_features)
    test_pool = Pool(test_sorted[features], cat_features=cat_features)
    model.fit(train_pool)

    test_pred = pd.Series(model.predict_proba(test_pool)[:, 1], index=test_sorted.index).sort_index()

    fi = pd.DataFrame(
        {
            "model": cfg["name"],
            "feature": features,
            "importance": model.get_feature_importance(train_pool),
        }
    ).sort_values(["importance", "feature"], ascending=[False, True])

    return test_pred, model, fi


def fit_all_full_models(train_df, test_df, artifacts):
    pred_frame = pd.DataFrame({ID_COL: test_df[ID_COL]})
    importance_frames = []

    for model_name, meta in artifacts.items():
        cfg = meta["config"]
        best_iter = meta["median_best_iter"]
        print(f"\n===== Full fit: {model_name} | iterations={best_iter} =====")

        pred_s, _, fi = fit_full_single_model(train_df, test_df, cfg, best_iter)
        pred_frame[f"pred_{model_name}"] = pred_s.values
        importance_frames.append(fi)

    fi_all = pd.concat(importance_frames, ignore_index=True)
    return pred_frame, fi_all


# =========================================================
# FINAL SUBMISSION
# =========================================================
def make_final_submission(test_df, sample_sub, pred_frame, stacker_meta):
    all_cols = stacker_meta["base_cols_all"]
    no_cols = stacker_meta["base_cols_no_clinic"]

    X_all = pred_frame[all_cols].values
    X_no = pred_frame[no_cols].values

    pred_known = stacker_meta["stacker_all"].predict_proba(X_all)[:, 1]
    pred_cold = stacker_meta["stacker_no_clinic"].predict_proba(X_no)[:, 1]

    cold_mask = test_df["clinic_id"].astype(str).isin(
        set(test_df.loc[test_df["is_cold_start_clinic"] == 1, "clinic_id"].astype(str).unique())
    )

    final_pred = np.where(cold_mask.values, pred_cold, pred_known)
    final_pred = np.clip(final_pred, 0.0, 1.0)

    submission = sample_sub[[ID_COL]].merge(
        pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: final_pred}),
        on=ID_COL,
        how="left",
        validate="1:1",
    )
    submission[TARGET] = submission[TARGET].clip(0, 1)

    return submission, pd.DataFrame(
        {
            ID_COL: test_df[ID_COL],
            "pred_stacker_all": pred_known,
            "pred_stacker_no_clinic": pred_cold,
            "is_cold_start_clinic": cold_mask.astype(int),
            "final_pred": final_pred,
        }
    )


# =========================================================
# MAIN
# =========================================================
def main():
    train_raw, test_raw, sample_sub = load_data()

    # Leakage-safe temporal patient history
    train_hist, test_hist = add_temporal_patient_history(train_raw, test_raw)

    train_df = add_features(train_hist)
    test_df = add_features(test_hist)

    print("Raw shapes")
    print("train:", train_df.shape)
    print("test :", test_df.shape)
    print(f"Train no-show rate: {train_df[TARGET].mean():.4f}")
    print(
        "Test cold-start clinic share: "
        f"{test_df['is_cold_start_clinic'].mean():.4%} | "
        f"clinics={sorted(test_df.loc[test_df['is_cold_start_clinic'] == 1, 'clinic_id'].unique().tolist())}"
    )
    print(
        "2025 history coverage | "
        f"train has_history={train_df['patient_2025_has_history'].mean():.4%}, "
        f"test has_history={test_df['patient_2025_has_history'].mean():.4%}"
    )
    print(
        "recency coverage | "
        f"train has_prev={train_df['has_prev_appt_2025'].mean():.4%}, "
        f"test has_prev={test_df['has_prev_appt_2025'].mean():.4%}"
    )

    folds = build_rolling_folds(train_df)
    print("\nRolling folds")
    for f in folds:
        print(
            f"Fold {f['fold']}: {f['valid_start'].date()} -> {f['valid_end'].date()} | "
            f"n_train={len(f['train_idx']):,} n_valid={len(f['valid_idx']):,}"
        )

    oof_frame, cv_summary, artifacts = run_oof_training(train_df, folds, MODEL_CONFIGS)
    stacker_meta = fit_stackers(oof_frame)

    pred_frame, fi_all = fit_all_full_models(train_df, test_df, artifacts)
    submission, extra_pred_frame = make_final_submission(test_df, sample_sub, pred_frame, stacker_meta)

    oof_frame.to_csv(OUTPUT_OOF, index=False)
    cv_summary.to_csv(OUTPUT_CV, index=False)
    fi_all.to_csv(OUTPUT_IMPORTANCE, index=False)
    submission.to_csv(OUTPUT_SUB, index=False)

    blend_meta = {
        "stacker_all_oof_ap": stacker_meta["oof_ap_all"],
        "stacker_no_clinic_oof_ap": stacker_meta["oof_ap_no_clinic"],
        "stacker_all_coefficients": stacker_meta["coef_all"],
        "stacker_no_clinic_coefficients": stacker_meta["coef_no_clinic"],
        "model_artifacts": {
            k: {
                "median_best_iter": int(v["median_best_iter"]),
                "oof_ap": float(v["oof_ap"]),
                "fold_ap_mean": float(v["fold_ap_mean"]),
            }
            for k, v in artifacts.items()
        },
    }
    OUTPUT_META.write_text(json.dumps(blend_meta, indent=2, ensure_ascii=False))

    final_pred_summary = extra_pred_frame["final_pred"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    print("\nFinal prediction summary")
    print(final_pred_summary)
    print(f"\nSubmission saved to: {OUTPUT_SUB}")
    print(f"OOF predictions saved to: {OUTPUT_OOF}")
    print(f"CV summary saved to: {OUTPUT_CV}")
    print(f"Feature importance saved to: {OUTPUT_IMPORTANCE}")
    print(f"Blend metadata saved to: {OUTPUT_META}")


if __name__ == "__main__":
    main()