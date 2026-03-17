import numpy as np
import pandas as pd
from itertools import product
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
def add_features(df, use_clinic_id=True, add_hour_bucket=False):
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

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
    df["prior_noshow_rate"] = df["prior_noshow_count"] / df["prior_appt_count"].clip(lower=1)

    df["lead_days"] = (df["appointment_datetime"] - df["booking_datetime"]).dt.days
    df["distance_km"] = haversine(
        df["residence_lat"], df["residence_lon"],
        df["clinic_lat"], df["clinic_lon"]
    )

    df["appt_to_capacity"] = df["clinic_day_appt_count"] / np.maximum(df["capacity_daily"], 1)
    df["appt_minus_capacity"] = df["clinic_day_appt_count"] - df["capacity_daily"]
    df["clinic_load_x_wait"] = df["clinic_load_ratio"] * df["wait_mins_est"]

    if add_hour_bucket:
        df["appt_hour_bucket"] = pd.cut(
            df["appointment_hour"],
            bins=[-1, 8, 11, 14, 17, 23],
            labels=["very_early", "morning", "noon", "afternoon", "late"]
        ).astype(str)

    cat_cols = ["specialty", "booking_channel", "appointment_type", "sex", "area_id"]
    if use_clinic_id:
        cat_cols = ["clinic_id"] + cat_cols
    if add_hour_bucket:
        cat_cols.append("appt_hour_bucket")

    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def build_feature_lists(df, use_clinic_id=True):
    drop_cols = [
        TARGET, ID_COL, "patient_id", TIME_COL, "booking_datetime",
        "appt_date_only", "sms_lead_hours",
    ]

    if not use_clinic_id:
        drop_cols.append("clinic_id")

    features = [c for c in df.columns if c not in drop_cols]

    cat_features = ["specialty", "booking_channel", "appointment_type", "sex", "area_id"]
    if use_clinic_id:
        cat_features = ["clinic_id"] + cat_features
    if "appt_hour_bucket" in features:
        cat_features.append("appt_hour_bucket")

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
# MODEL CONFIGS
# =========================================================
MODEL_CONFIGS = [
    {
        "name": "baseline_4954",
        "use_clinic_id": True,
        "add_hour_bucket": False,
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 10,
        "min_data_in_leaf": 40,
    },
    {
        "name": "regularized_off",
        "use_clinic_id": False,
        "add_hour_bucket": False,
        "iterations": 2400,
        "learning_rate": 0.025,
        "depth": 5,
        "l2_leaf_reg": 12,
        "min_data_in_leaf": 60,
    },
    {
        "name": "hour_bucket_reg_off",
        "use_clinic_id": False,
        "add_hour_bucket": True,
        "iterations": 2400,
        "learning_rate": 0.025,
        "depth": 5,
        "l2_leaf_reg": 12,
        "min_data_in_leaf": 60,
    },
]


# =========================================================
# TRAIN ONE MODEL ON VALIDATION SPLIT
# =========================================================
def train_and_eval_one(train_base, cfg):
    df = add_features(train_base, use_clinic_id=cfg["use_clinic_id"], add_hour_bucket=cfg["add_hour_bucket"])
    features, cat_features = build_feature_lists(df, use_clinic_id=cfg["use_clinic_id"])

    train_part, valid_part = make_time_split(df)

    train_pool = Pool(train_part[features], train_part[TARGET], cat_features=cat_features)
    valid_pool = Pool(valid_part[features], valid_part[TARGET], cat_features=cat_features)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        iterations=cfg["iterations"],
        learning_rate=cfg["learning_rate"],
        depth=cfg["depth"],
        l2_leaf_reg=cfg["l2_leaf_reg"],
        min_data_in_leaf=cfg["min_data_in_leaf"],
        bootstrap_type="Bernoulli",
        subsample=0.8,
        has_time=True,
        task_type="GPU",
        devices="0",
        random_seed=RANDOM_SEED,
        od_type="Iter",
        od_wait=200,
        verbose=200,
        allow_writing_files=False
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    valid_pred = model.predict_proba(valid_pool)[:, 1]
    ap = average_precision_score(valid_part[TARGET], valid_pred)

    best_iter = model.get_best_iteration()
    if best_iter is None or best_iter <= 0:
        best_iter = model.tree_count_

    return {
        "name": cfg["name"],
        "config": cfg,
        "ap": ap,
        "best_iter": int(best_iter),
        "valid_pred": valid_pred,
        "y_valid": valid_part[TARGET].values,
        "features": features,
        "cat_features": cat_features,
    }


# =========================================================
# SEARCH BLEND WEIGHTS
# =========================================================
def search_blend(results):
    y = results[0]["y_valid"]
    pred_map = {r["name"]: r["valid_pred"] for r in results}

    best = None
    best_ap = -1

    grid = np.arange(0, 1.01, 0.1)

    for w1, w2, w3 in product(grid, grid, grid):
        s = w1 + w2 + w3
        if abs(s - 1.0) > 1e-9:
            continue

        blend = (
            w1 * pred_map["baseline_4954"] +
            w2 * pred_map["regularized_off"] +
            w3 * pred_map["hour_bucket_reg_off"]
        )
        ap = average_precision_score(y, blend)

        if ap > best_ap:
            best_ap = ap
            best = {
                "baseline_4954": round(w1, 2),
                "regularized_off": round(w2, 2),
                "hour_bucket_reg_off": round(w3, 2),
            }

    return best, best_ap


# =========================================================
# FIT FULL MODEL
# =========================================================
def fit_full_model(train_base, test_base, cfg, best_iter):
    train_df = add_features(train_base, use_clinic_id=cfg["use_clinic_id"], add_hour_bucket=cfg["add_hour_bucket"])
    test_df = add_features(test_base, use_clinic_id=cfg["use_clinic_id"], add_hour_bucket=cfg["add_hour_bucket"])

    features, cat_features = build_feature_lists(train_df, use_clinic_id=cfg["use_clinic_id"])

    train_df = train_df.sort_values(TIME_COL).reset_index(drop=True)
    test_df = test_df.sort_values(TIME_COL).reset_index(drop=True)

    train_pool = Pool(train_df[features], train_df[TARGET], cat_features=cat_features)
    test_pool = Pool(test_df[features], cat_features=cat_features)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC:type=Classic",
        iterations=best_iter,
        learning_rate=cfg["learning_rate"],
        depth=cfg["depth"],
        l2_leaf_reg=cfg["l2_leaf_reg"],
        min_data_in_leaf=cfg["min_data_in_leaf"],
        bootstrap_type="Bernoulli",
        subsample=0.8,
        has_time=True,
        task_type="GPU",
        devices="0",
        random_seed=RANDOM_SEED,
        verbose=200,
        allow_writing_files=False
    )

    model.fit(train_pool)
    test_pred = model.predict_proba(test_pool)[:, 1]
    return model, test_df, test_pred, train_pool, features


# =========================================================
# MAIN
# =========================================================
def main():
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

    # validation train
    results = []
    for cfg in MODEL_CONFIGS:
        print(f"\n===== Training {cfg['name']} =====")
        res = train_and_eval_one(train, cfg)
        results.append(res)
        print(f"{res['name']} AP: {res['ap']:.6f} | best_iter={res['best_iter']}")

    # blend search
    best_weights, best_blend_ap = search_blend(results)

    print("\n================ BLEND SEARCH ================")
    for r in results:
        print(f"{r['name']}: {r['ap']:.6f}")
    print("Best blend weights:", best_weights)
    print(f"Best blend AP: {best_blend_ap:.6f}")

    # full fit
    pred_dict = {}
    test_sorted_ref = None
    fi_frames = []

    for res in results:
        print(f"\n===== Full fit {res['name']} =====")
        model, test_sorted, test_pred, train_pool, features = fit_full_model(
            train_base=train,
            test_base=test,
            cfg=res["config"],
            best_iter=res["best_iter"]
        )
        pred_dict[res["name"]] = test_pred
        test_sorted_ref = test_sorted  # hepsi aynı sıralamada

        fi = pd.DataFrame({
            "model": res["name"],
            "feature": features,
            "importance": model.get_feature_importance(train_pool)
        }).sort_values("importance", ascending=False)
        fi_frames.append(fi)

    # final blend
    final_test_pred = (
        best_weights["baseline_4954"] * pred_dict["baseline_4954"] +
        best_weights["regularized_off"] * pred_dict["regularized_off"] +
        best_weights["hour_bucket_reg_off"] * pred_dict["hour_bucket_reg_off"]
    )

    submission = sample_sub[[ID_COL]].merge(
        test_sorted_ref[[ID_COL]].assign(label_noshow=final_test_pred),
        on=ID_COL,
        how="left",
        validate="1:1"
    )
    submission["label_noshow"] = submission["label_noshow"].clip(0, 1)
    submission.to_csv("submission_blended_best.csv", index=False)

    print("\nSubmission kaydedildi: submission_blended_best.csv")
    print(submission.head())

    fi_all = pd.concat(fi_frames, axis=0, ignore_index=True)
    fi_all.to_csv("feature_importance_blend_models.csv", index=False)
    print("Feature importances kaydedildi: feature_importance_blend_models.csv")


if __name__ == "__main__":
    main()