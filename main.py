import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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

ROLLING_VALID_DAYS = 28
N_FOLDS = 4
ROLLING_STEP_DAYS = 28
TE_SMOOTH = 25.0

SEED1 = 42
SEED2 = 2026

OUTPUT_SUB = BASE_DIR / "submission_catboost_mean_blend_gpu.csv"
OUTPUT_OOF = BASE_DIR / "oof_catboost_mean_blend_gpu.csv"
OUTPUT_CV = BASE_DIR / "cv_summary_catboost_mean_blend_gpu.csv"
OUTPUT_IMPORTANCE = BASE_DIR / "feature_importance_catboost_mean_blend_gpu.csv"
OUTPUT_META = BASE_DIR / "blend_metadata_catboost_mean_blend_gpu.json"

# =========================================================
# 4 MODEL = 2 BASE MODEL x 2 SEED
# - Only CatBoost
# - Simple mean blend
# =========================================================
MODEL_CONFIGS = [
    {
        "name": "clinic_aware_seed1",
        "use_clinic_id": True,
        "params": {
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 25,
            "min_data_in_leaf": 40,
            "random_strength": 1.5,
            "subsample": 0.80,
            "random_seed": SEED1,
        },
    },
    {
        "name": "clinic_aware_seed2",
        "use_clinic_id": True,
        "params": {
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 25,
            "min_data_in_leaf": 40,
            "random_strength": 1.5,
            "subsample": 0.80,
            "random_seed": SEED2,
        },
    },
    {
        "name": "clinic_agnostic_seed1",
        "use_clinic_id": False,
        "params": {
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 25,
            "min_data_in_leaf": 60,
            "random_strength": 1.5,
            "subsample": 0.78,
            "random_seed": SEED1,
        },
    },
    {
        "name": "clinic_agnostic_seed2",
        "use_clinic_id": False,
        "params": {
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 25,
            "min_data_in_leaf": 60,
            "random_strength": 1.5,
            "subsample": 0.78,
            "random_seed": SEED2,
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
# LEAKAGE-SAFE TEMPORAL HISTORIES
# - Patient 2025 cumulative history
# - Patient recency
# - Clinic 2025 cumulative history
# =========================================================
def add_temporal_histories(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["_is_train"] = 1
    test_df["_is_train"] = 0

    if TARGET not in test_df.columns:
        test_df[TARGET] = np.nan

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=False)
    combined = combined.sort_values([TIME_COL, ID_COL]).reset_index(drop=True)

    combined["_known_noshow"] = np.where(
        combined["_is_train"] == 1,
        combined[TARGET].fillna(0),
        0,
    )

    # ------------------------------
    # Patient rolling history + recency
    # ------------------------------
    combined = combined.sort_values(["patient_id", TIME_COL, ID_COL]).reset_index(drop=True)

    combined["patient_2025_appt_count"] = combined.groupby("patient_id").cumcount()

    combined["patient_2025_noshow_count"] = (
        combined.groupby("patient_id")["_known_noshow"].cumsum() - combined["_known_noshow"]
    )

    combined["prev_appt_datetime"] = combined.groupby("patient_id")[TIME_COL].shift(1)
    combined["days_since_last_appt"] = (
        (combined[TIME_COL] - combined["prev_appt_datetime"]).dt.total_seconds() / 86400.0
    )
    combined["days_since_last_appt"] = combined["days_since_last_appt"].fillna(-1.0)

    # ------------------------------
    # Clinic rolling history
    # ------------------------------
    combined = combined.sort_values(["clinic_id", TIME_COL, ID_COL]).reset_index(drop=True)

    combined["clinic_2025_appt_count"] = combined.groupby("clinic_id").cumcount()

    combined["clinic_2025_noshow_count"] = (
        combined.groupby("clinic_id")["_known_noshow"].cumsum() - combined["_known_noshow"]
    )

    # Back to chronological order
    combined = combined.sort_values([TIME_COL, ID_COL]).reset_index(drop=True)

    combined["patient_2025_appt_count"] = combined["patient_2025_appt_count"].astype(np.int32)
    combined["patient_2025_noshow_count"] = combined["patient_2025_noshow_count"].astype(np.int32)
    combined["clinic_2025_appt_count"] = combined["clinic_2025_appt_count"].astype(np.int32)
    combined["clinic_2025_noshow_count"] = combined["clinic_2025_noshow_count"].astype(np.int32)
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

    df["booking_month"] = df[BOOKING_TIME_COL].dt.month

    df["appointment_hour_sin"] = np.sin(2 * np.pi * df["appointment_hour"] / 24.0)
    df["appointment_hour_cos"] = np.cos(2 * np.pi * df["appointment_hour"] / 24.0)
    df["appointment_dow_sin"] = np.sin(2 * np.pi * df["appointment_dow"] / 7.0)

    # ------------------------------
    # lead / booking features
    # ------------------------------
    df["lead_time_days"] = df["lead_time_hours"] / 24.0
    df["lead_time_log1p"] = np.log1p(df["lead_time_hours"].clip(lower=0))
    df["same_day_booking"] = (df["lead_time_hours"] <= 24).astype(int)

    # ------------------------------
    # sms / phone features
    # ------------------------------
    df["sms_lead_missing"] = df["sms_lead_hours"].isna().astype(int)
    df["sms_lead_hours_filled"] = df["sms_lead_hours"].fillna(-1)
    df["sms_possible_but_not_sent"] = ((df["has_phone"] == 1) & (df["sms_sent"] == 0)).astype(int)
    df["sms_sent_x_has_phone"] = (df["sms_sent"] * df["has_phone"]).astype(int)

    # ------------------------------
    # pre-2025 patient history
    # Kullanıcı istediği formül:
    # prior_noshow_count / prior_appt_count.clip(lower=1)
    # ------------------------------
    df["prior_show_count"] = (df["prior_appt_count"] - df["prior_noshow_count"]).clip(lower=0)
    df["prior_noshow_rate"] = (
        df["prior_noshow_count"].fillna(0) / df["prior_appt_count"].fillna(0).clip(lower=1)
    )
    df["prior_noshow_rate_safe"] = df["prior_noshow_rate"]
    df["prior_noshow_ratio_safe"] = df["prior_noshow_rate"]

    # ------------------------------
    # patient 2025 history + recency
    # ------------------------------
    df["patient_2025_noshow_rate"] = (
        df["patient_2025_noshow_count"] / np.maximum(df["patient_2025_appt_count"], 1)
    )
    df["patient_2025_noshow_rate_safe"] = df["patient_2025_noshow_rate"]

    df["days_since_last_appt_log1p"] = np.where(
        df["days_since_last_appt"] >= 0,
        np.log1p(df["days_since_last_appt"]),
        -1.0,
    )

    # ------------------------------
    # clinic 2025 history
    # ------------------------------
    df["clinic_2025_show_count"] = (
        df["clinic_2025_appt_count"] - df["clinic_2025_noshow_count"]
    ).clip(lower=0)

    df["clinic_2025_noshow_rate"] = (
        df["clinic_2025_noshow_count"] / np.maximum(df["clinic_2025_appt_count"], 1)
    )
    df["clinic_2025_noshow_rate_safe"] = df["clinic_2025_noshow_rate"]

    # ------------------------------
    # numeric interactions
    # ------------------------------
    df["prior_noshow_rate_squared"] = df["prior_noshow_rate"] ** 2
    df["patient_age_x_chronic"] = df["age"] * df["chronic_count"]
    df["distance_x_age"] = df["distance_km"] * df["age"]
    df["ses_x_has_phone"] = df["ses_score"] * df["has_phone"]

    df["lead_time_x_no_sms"] = df["lead_time_days"] * (1 - df["sms_sent"])
    df["noshow_rate_x_lead_time"] = df["prior_noshow_rate"] * df["lead_time_days"]
    df["distance_x_noshow_rate"] = df["distance_km"] * df["prior_noshow_rate"]
    df["wait_time_x_noshow"] = df["wait_mins_est"] * df["prior_noshow_rate"]

    df["patient2025_noshow_x_lead"] = df["patient_2025_noshow_rate"] * df["lead_time_days"]
    df["patient2025_noshow_x_distance"] = df["patient_2025_noshow_rate"] * df["distance_km"]
    df["patient2025_noshow_x_wait"] = df["patient_2025_noshow_rate"] * df["wait_mins_est"]

    df["clinic2025_noshow_x_wait"] = df["clinic_2025_noshow_rate"] * df["wait_mins_est"]
    df["clinic2025_noshow_x_load"] = df["clinic_2025_noshow_rate"] * df["clinic_load_ratio"]
    df["clinic2025_noshow_x_capacity"] = df["clinic_2025_noshow_rate"] * df["capacity_daily"]

    df["recency_x_prior_noshow"] = df["days_since_last_appt_log1p"] * df["prior_noshow_rate"]
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

    for col in cols:
        if col not in train_df.columns:
            continue

        stats = train_df.groupby(col, dropna=False)[target].agg(["sum", "count"])
        stats["te"] = (stats["sum"] + global_mean * smooth) / (stats["count"] + smooth)
        mapper = stats["te"]

        new_col = f"te_{col}"
        train_df[new_col] = train_df[col].map(mapper).fillna(global_mean).astype(float)
        other_df[new_col] = other_df[col].map(mapper).fillna(global_mean).astype(float)

    return train_df, other_df


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
        # removed noisy features
        "is_month_start",
        "is_month_end",
        "clinic_2025_has_history",
        "patient_2025_has_history",
        "had_prior_noshow",
        "has_prior_history",
        "very_short_lead",
        "recent_return_7d",
        "recent_return_30d",
        "open_on_saturday",
        "recency_x_2025_noshow",
        "patient_2025_show_count",
        "patient_2025_noshow_count",
        "patient_2025_appt_count",
        "has_prev_appt_2025",
        "long_lead",
        "very_long_lead",
        "appointment_dow_cos",
        "booking_hour",
        "booking_dow",
        "sex",
    ]

    if not use_clinic_id:
        drop_cols.append("clinic_id")
        drop_cols += ["clinic_x_hour", "clinic_x_type"]

    features = [c for c in df.columns if c not in drop_cols]

    cat_features = [
        "specialty",
        "booking_channel",
        "appointment_type",
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
# FOLDS
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
def build_catboost(params):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        min_data_in_leaf=params["min_data_in_leaf"],
        bootstrap_type="Poisson",
        subsample=params.get("subsample", 0.8),
        random_strength=params.get("random_strength", 1.0),
        has_time=True,
        random_seed=params.get("random_seed", SEED1),
        od_type="Iter",
        od_wait=300,
        task_type="GPU",
        devices="0",
        verbose=False,
        allow_writing_files=False,
    )


def train_predict_single_model(train_fold, valid_fold, cfg):
    te_cols = get_te_cols(use_clinic_id=cfg["use_clinic_id"])
    train_enc, valid_enc = apply_target_encoding(train_fold, valid_fold, te_cols)
    features, cat_features = build_feature_lists(train_enc, use_clinic_id=cfg["use_clinic_id"])

    train_sorted = train_enc.sort_values(TIME_COL)
    valid_sorted = valid_enc.sort_values(TIME_COL)

    train_pool = Pool(train_sorted[features], train_sorted[TARGET], cat_features=cat_features)
    valid_pool = Pool(valid_sorted[features], valid_sorted[TARGET], cat_features=cat_features)

    model = build_catboost(cfg["params"])
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    preds = model.predict_proba(valid_pool)[:, 1]
    pred_s = pd.Series(preds, index=valid_sorted.index).sort_index()

    raw_best_iter = model.get_best_iteration()
    if raw_best_iter is None or raw_best_iter <= 0:
        raw_best_iter = model.tree_count_
    else:
        raw_best_iter = raw_best_iter + 1

    used_iter = max(int(raw_best_iter), 500)

    fi = pd.DataFrame(
        {
            "model": cfg["name"],
            "feature": features,
            "importance": model.get_feature_importance(train_pool),
        }
    ).sort_values(["importance", "feature"], ascending=[False, True])

    return pred_s, int(raw_best_iter), int(used_iter), model, fi


# =========================================================
# OOF TRAINING
# =========================================================
def run_oof_training(train_df, folds, model_configs):
    oof_frame = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET: train_df[TARGET]})
    cv_rows = []
    model_artifacts = {}
    fi_frames = []

    for cfg in model_configs:
        print(f"\n===== OOF training: {cfg['name']} =====")
        oof_pred = pd.Series(index=train_df.index, dtype=float)
        fold_scores = []
        raw_best_iters = []
        used_iters = []

        for fold_info in folds:
            fold_no = fold_info["fold"]
            tr_idx = fold_info["train_idx"]
            va_idx = fold_info["valid_idx"]

            tr_fold = train_df.loc[tr_idx].copy()
            va_fold = train_df.loc[va_idx].copy()

            pred_s, raw_best_iter, used_iter, _, fi = train_predict_single_model(tr_fold, va_fold, cfg)
            raw_best_iters.append(raw_best_iter)
            used_iters.append(used_iter)
            fi["fold"] = fold_no
            fi_frames.append(fi)

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
                    "raw_best_iter": int(raw_best_iter),
                    "used_iter": int(used_iter),
                }
            )

            print(
                f"Fold {fold_no}: AP={fold_ap:.6f} | "
                f"valid={fold_info['valid_start'].date()} -> {fold_info['valid_end'].date()} | "
                f"n_train={len(tr_fold):,} n_valid={len(va_fold):,} | "
                f"raw_best_iter={raw_best_iter} | used_iter={used_iter}"
            )

        overall_mask = oof_pred.notna()
        overall_ap = average_precision_score(train_df.loc[overall_mask, TARGET], oof_pred.loc[overall_mask])

        median_raw_iter = int(np.median(raw_best_iters)) if raw_best_iters else 0
        median_used_iter = max(int(np.median(used_iters)), 500) if used_iters else 500

        oof_frame[f"pred_{cfg['name']}"] = oof_pred
        model_artifacts[cfg["name"]] = {
            "config": cfg,
            "median_best_iter": median_used_iter,
            "median_raw_best_iter": median_raw_iter,
            "oof_ap": overall_ap,
            "fold_ap_mean": float(np.mean(fold_scores)),
        }

        print(
            f"{cfg['name']} overall OOF AP={overall_ap:.6f} | "
            f"fold mean={np.mean(fold_scores):.6f} | "
            f"median_raw_best_iter={median_raw_iter} | "
            f"median_used_iter={median_used_iter}"
        )

    cv_summary = pd.DataFrame(cv_rows)

    fi_all = (
        pd.concat(fi_frames, ignore_index=True)
        .groupby(["model", "feature"], as_index=False)["importance"]
        .mean()
        .sort_values(["model", "importance", "feature"], ascending=[True, False, True])
    )

    return oof_frame, cv_summary, model_artifacts, fi_all


# =========================================================
# SIMPLE MEAN BLEND
# =========================================================
def get_blend_cols():
    all_cols = [
        "pred_clinic_aware_seed1",
        "pred_clinic_aware_seed2",
        "pred_clinic_agnostic_seed1",
        "pred_clinic_agnostic_seed2",
    ]
    agnostic_cols = [
        "pred_clinic_agnostic_seed1",
        "pred_clinic_agnostic_seed2",
    ]
    return all_cols, agnostic_cols


def evaluate_mean_blend_oof(oof_frame):
    all_cols, agnostic_cols = get_blend_cols()

    mask_all = oof_frame[all_cols].notna().all(axis=1)
    pred_all = oof_frame.loc[mask_all, all_cols].mean(axis=1)
    ap_all = average_precision_score(oof_frame.loc[mask_all, TARGET], pred_all)

    mask_no = oof_frame[agnostic_cols].notna().all(axis=1)
    pred_no = oof_frame.loc[mask_no, agnostic_cols].mean(axis=1)
    ap_no = average_precision_score(oof_frame.loc[mask_no, TARGET], pred_no)

    oof_out = oof_frame.copy()
    oof_out["pred_mean_all"] = np.nan
    oof_out.loc[mask_all, "pred_mean_all"] = pred_all.values

    oof_out["pred_mean_agnostic"] = np.nan
    oof_out.loc[mask_no, "pred_mean_agnostic"] = pred_no.values

    print(f"\nMean blend OOF AP (all 4 models): {ap_all:.6f}")
    print(f"Mean blend OOF AP (agnostic 2 models): {ap_no:.6f}")

    return {
        "all_cols": all_cols,
        "agnostic_cols": agnostic_cols,
        "oof_ap_all": float(ap_all),
        "oof_ap_agnostic": float(ap_no),
        "oof_frame": oof_out,
    }


# =========================================================
# FULL TRAIN / TEST PREDICTIONS
# =========================================================
def fit_full_single_model(train_df, test_df, cfg, best_iter):
    te_cols = get_te_cols(use_clinic_id=cfg["use_clinic_id"])
    train_enc, test_enc = apply_target_encoding(train_df, test_df, te_cols)
    features, cat_features = build_feature_lists(train_enc, use_clinic_id=cfg["use_clinic_id"])

    train_sorted = train_enc.sort_values(TIME_COL)
    test_sorted = test_enc.sort_values(TIME_COL)

    model_params = dict(cfg["params"])
    model_params["iterations"] = max(int(best_iter), 500)
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
def make_final_submission(test_df, sample_sub, pred_frame, blend_meta):
    all_cols = blend_meta["all_cols"]
    agnostic_cols = blend_meta["agnostic_cols"]

    pred_known = pred_frame[all_cols].mean(axis=1).values
    pred_cold = pred_frame[agnostic_cols].mean(axis=1).values

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

    extra = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL],
            "pred_mean_all": pred_known,
            "pred_mean_agnostic": pred_cold,
            "is_cold_start_clinic": cold_mask.astype(int),
            "final_pred": final_pred,
        }
    )
    return submission, extra


# =========================================================
# MAIN
# =========================================================
def main():
    train_raw, test_raw, sample_sub = load_data()

    # Leakage-safe temporal histories
    train_hist, test_hist = add_temporal_histories(train_raw, test_raw)

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
        "2025 patient history coverage | "
        f"train has_history={(train_hist['patient_2025_appt_count'] > 0).mean():.4%}, "
        f"test has_history={(test_hist['patient_2025_appt_count'] > 0).mean():.4%}"
    )
    print(
        "2025 clinic history coverage | "
        f"train has_history={(train_hist['clinic_2025_appt_count'] > 0).mean():.4%}, "
        f"test has_history={(test_hist['clinic_2025_appt_count'] > 0).mean():.4%}"
    )
    print(
        "recency coverage | "
        f"train has_prev={(train_hist['days_since_last_appt'] >= 0).mean():.4%}, "
        f"test has_prev={(test_hist['days_since_last_appt'] >= 0).mean():.4%}"
    )

    folds = build_rolling_folds(train_df)
    print("\nRolling folds")
    for f in folds:
        print(
            f"Fold {f['fold']}: {f['valid_start'].date()} -> {f['valid_end'].date()} | "
            f"n_train={len(f['train_idx']):,} n_valid={len(f['valid_idx']):,}"
        )

    oof_frame, cv_summary, artifacts, oof_importance = run_oof_training(train_df, folds, MODEL_CONFIGS)
    blend_meta = evaluate_mean_blend_oof(oof_frame)

    pred_frame, fi_full = fit_all_full_models(train_df, test_df, artifacts)
    submission, extra_pred_frame = make_final_submission(
        test_df=test_df,
        sample_sub=sample_sub,
        pred_frame=pred_frame,
        blend_meta=blend_meta,
    )

    # Save outputs
    oof_save = blend_meta["oof_frame"]
    oof_save.to_csv(OUTPUT_OOF, index=False)
    cv_summary.to_csv(OUTPUT_CV, index=False)

    fi_all = pd.concat(
        [
            oof_importance.assign(source="oof_mean_over_folds"),
            fi_full.assign(source="full_fit"),
        ],
        ignore_index=True,
    )
    fi_all.to_csv(OUTPUT_IMPORTANCE, index=False)

    submission.to_csv(OUTPUT_SUB, index=False)

    meta_out = {
        "blend_type": "simple_mean",
        "task_type": "GPU",
        "devices": "0",
        "bootstrap_type": "Poisson",
        "eval_metric_for_early_stopping": "Logloss",
        "min_forced_iterations": 500,
        "all_model_columns": blend_meta["all_cols"],
        "agnostic_model_columns": blend_meta["agnostic_cols"],
        "oof_ap_all_4_mean": blend_meta["oof_ap_all"],
        "oof_ap_agnostic_2_mean": blend_meta["oof_ap_agnostic"],
        "model_artifacts": {
            k: {
                "median_best_iter": int(v["median_best_iter"]),
                "median_raw_best_iter": int(v["median_raw_best_iter"]),
                "oof_ap": float(v["oof_ap"]),
                "fold_ap_mean": float(v["fold_ap_mean"]),
                "use_clinic_id": bool(v["config"]["use_clinic_id"]),
                "random_seed": int(v["config"]["params"]["random_seed"]),
                "max_iterations": int(v["config"]["params"]["iterations"]),
                "learning_rate": float(v["config"]["params"]["learning_rate"]),
            }
            for k, v in artifacts.items()
        },
    }
    OUTPUT_META.write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))

    final_pred_summary = extra_pred_frame["final_pred"].describe(
        percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
    )

    print("\nFinal prediction summary")
    print(final_pred_summary)
    print(f"\nSubmission saved to: {OUTPUT_SUB}")
    print(f"OOF predictions saved to: {OUTPUT_OOF}")
    print(f"CV summary saved to: {OUTPUT_CV}")
    print(f"Feature importance saved to: {OUTPUT_IMPORTANCE}")
    print(f"Blend metadata saved to: {OUTPUT_META}")


if __name__ == "__main__":
    main()