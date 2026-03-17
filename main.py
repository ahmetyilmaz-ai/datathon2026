import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool, CatBoostError
import optuna


# =========================================================
# AYARLAR
# =========================================================
SEED = 42
DATE_COL = "appointment_datetime"
TARGET = "label_noshow"
CUTOFF = pd.to_datetime("2025-10-01")

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================
def resolve_specialty_col(df):
    for c in ["specialty_x", "specialty", "specialty_y"]:
        if c in df.columns:
            return c
    raise ValueError("specialty kolonu bulunamadı.")

def kfold_target_encode(train, test, cols, target, n_splits=5, smoothing=20.0):
    train = train.copy()
    test = test.copy()

    kf = KFold(n_splits=n_splits, shuffle=False)
    global_mean = train[target].mean()

    for col in cols:
        te_col = f"{col}_te"
        train[te_col] = np.nan

        for tr_idx, val_idx in kf.split(train):
            fold_tr = train.iloc[tr_idx]
            stats = fold_tr.groupby(col)[target].agg(["mean", "count"])
            stats["smooth"] = (
                (stats["count"] * stats["mean"] + smoothing * global_mean)
                / (stats["count"] + smoothing)
            )
            train.loc[train.index[val_idx], te_col] = (
                train.iloc[val_idx][col].map(stats["smooth"])
            )

        train[te_col] = train[te_col].fillna(global_mean)

        full = train.groupby(col)[target].agg(["mean", "count"])
        full["smooth"] = (
            (full["count"] * full["mean"] + smoothing * global_mean)
            / (full["count"] + smoothing)
        )
        test[te_col] = test[col].map(full["smooth"]).fillna(global_mean)

    return train, test


def check_catboost_gpu(train_pool, val_pool):
    try:
        probe = CatBoostClassifier(
            iterations=5,
            learning_rate=0.1,
            depth=4,
            verbose=False,
            random_seed=SEED,
            task_type="GPU",
            devices="0",
        )
        probe.fit(train_pool, eval_set=val_pool, use_best_model=False)
        return True, None
    except Exception as e:
        return False, str(e)


def build_cat_model(base_params, use_gpu=True):
    params = dict(base_params)
    if use_gpu:
        params["task_type"] = "GPU"
        params["devices"] = "0"
    return CatBoostClassifier(**params)


# =========================================================
# VERİYİ OKU
# =========================================================
train_raw = pd.read_csv("appointments_train.csv")
test_raw = pd.read_csv("appointments_test.csv")
patients = pd.read_csv("patients.csv")
clinics = pd.read_csv("clinics.csv")

print("Raw shapes")
print("train:", train_raw.shape)
print("test :", test_raw.shape)

# Merge
train = train_raw.merge(patients, on="patient_id", how="left")
train = train.merge(clinics, on="clinic_id", how="left")
test = test_raw.merge(patients, on="patient_id", how="left")
test = test.merge(clinics, on="clinic_id", how="left")

print(f"Train no-show rate: {train[TARGET].mean():.4f}")

# =========================================================
# TEMİZLİK + FEATURE ENGINEERING
# =========================================================
sms_median = train.loc[train["sms_sent"] == 1, "sms_lead_hours"].median()

for df in [train, test]:
    df["sms_lead_hours"] = np.where(
        df["sms_sent"] == 0,
        -1,
        df["sms_lead_hours"].fillna(sms_median),
    )
    df["wait_mins_est"] = df["wait_mins_est"].fillna(df["base_wait_mins_est"])

for df in [train, test]:
    df["appointment_datetime"] = pd.to_datetime(df["appointment_datetime"])
    df["booking_datetime"] = pd.to_datetime(df["booking_datetime"])

    spec_col_local = resolve_specialty_col(df)

    # Zaman
    df["appt_hour"] = df["appointment_datetime"].dt.hour
    df["appt_dow"] = df["appointment_datetime"].dt.dayofweek
    df["appt_day"] = df["appointment_datetime"].dt.day
    df["appt_month"] = df["appointment_datetime"].dt.month
    df["book_month"] = df["booking_datetime"].dt.month
    df["is_weekend"] = (df["appt_dow"] >= 5).astype(int)
    df["is_monday"] = (df["appt_dow"] == 0).astype(int)
    df["is_friday"] = (df["appt_dow"] == 4).astype(int)
    df["is_early"] = (df["appt_hour"] <= 8).astype(int)
    df["is_late"] = (df["appt_hour"] >= 17).astype(int)

    # Klinik kapalı mı
    df["clinic_closed_day"] = (
        ((df["is_weekend"] == 1) & (df["open_on_saturday"] == 0))
    ).astype(int)

    # Lead time
    df["lead_time_hours"] = df["lead_time_hours"].fillna(0).clip(lower=0)
    df["lead_log"] = np.log1p(df["lead_time_hours"])
    df["lead_days"] = df["lead_time_hours"] / 24.0
    df["lead_long"] = (df["lead_days"] > 30).astype(int)
    df["lead_vlong"] = (df["lead_days"] > 60).astype(int)

    df["lead_time_bucket"] = pd.cut(
        df["lead_time_hours"],
        bins=[-0.001, 48, 168, 480, 960, 999999],
        labels=[0, 1, 2, 3, 4],
        include_lowest=True,
    ).astype(int)

    # SMS
    df["sms_effective"] = np.where(
        df["sms_sent"] == 0,
        0,
        np.where(df["sms_lead_hours"] <= 48, 2, np.where(df["sms_lead_hours"] <= 72, 1, 0)),
    )
    df["sms_policy_x_sent"] = df["sms_policy_prob"] * df["sms_sent"]
    df["sms_expected_not_sent"] = (
        ((df["sms_sent"] == 0) & (df["sms_policy_prob"] > 0.7))
    ).astype(int)
    df["sms_x_lead"] = df["sms_sent"] * df["lead_log"]
    df["sms_too_early"] = (
        ((df["sms_sent"] == 1) & (df["sms_lead_hours"] > 72))
    ).astype(int)

    # Bekleme
    df["wait_above"] = df["wait_mins_est"] - df["base_wait_mins_est"]
    df["wait_ratio"] = df["wait_mins_est"] / (df["base_wait_mins_est"] + 1)
    df["wait_log"] = np.log1p(df["wait_mins_est"].clip(lower=0))

    # Mesafe
    df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())
    df["dist_log"] = np.log1p(df["distance_km"].clip(lower=0))
    df["is_far"] = (df["distance_km"] > 25).astype(int)

    # Klinik yoğunluk
    df["load_x_lead"] = df["clinic_load_ratio"] * df["lead_log"]
    df["load_x_dist"] = df["clinic_load_ratio"] * df["dist_log"]
    df["dist_load"] = df["distance_km"] * df["clinic_load_ratio"]
    df["wait_x_dist"] = df["wait_log"] * df["distance_km"]

    # SES
    df["low_ses"] = (df["ses_score"] < 0.3).astype(int)
    df["high_ses"] = (df["ses_score"] > 0.7).astype(int)
    df["ses_x_dist"] = df["ses_score"] * df["distance_km"]
    df["ses_x_lead"] = df["ses_score"] * df["lead_log"]
    df["ses_x_wait"] = df["ses_score"] * df["wait_log"]

    # Hasta geçmişi
    df["prior_noshow_rate"] = df["prior_noshow_rate"].fillna(0)
    df["is_new_patient"] = (df["prior_appt_count"] == 0).astype(int)
    df["prior_ns_log"] = np.log1p(df["prior_noshow_count"].fillna(0))
    df["prior_appt_log"] = np.log1p(df["prior_appt_count"].fillna(0))
    df["noshow_reliability"] = (
        (1 - df["prior_noshow_rate"]) * np.log1p(df["prior_appt_count"].fillna(0))
    )
    df["chronic_noshow"] = (
        ((df["prior_noshow_count"].fillna(0) >= 3) & (df["prior_noshow_rate"] > 0.5))
    ).astype(int)
    df["noshow_x_lead"] = df["prior_noshow_rate"] * df["lead_log"]

    # Randevu tipi
    df["is_followup"] = (df["appointment_type"] == "followup").astype(int)
    df["loyalty_score"] = df["is_followup"] * np.log1p(df["prior_appt_count"].fillna(0))

    # Kronik & iletişim
    df["chronic_x_phone"] = df["chronic_count"] * df["has_phone"]
    df["high_chronic"] = (df["chronic_count"] >= 3).astype(int)
    df["no_contact"] = (((df["has_phone"] == 0) & (df["sms_sent"] == 0))).astype(int)
    df["chronic_x_ses"] = df["chronic_count"] * df["ses_score"]

    # Yaş
    df["age_pediatric"] = (df["age"] <= 18).astype(int)
    df["age_elderly"] = (df["age"] >= 65).astype(int)
    df["age_working"] = (((df["age"] >= 25) & (df["age"] <= 55))).astype(int)
    df["age_x_dist"] = df["age"] * df["dist_log"]


# =========================================================
# TARGET ENCODING
# =========================================================
spec_col = resolve_specialty_col(train)
enc_cols = ["booking_channel", "appointment_type", "sex", spec_col]
train, test = kfold_target_encode(train, test, enc_cols, TARGET)

# =========================================================
# TIME SPLIT
# =========================================================
tr = train[train[DATE_COL] < CUTOFF].copy()
val = train[train[DATE_COL] >= CUTOFF].copy()

print(f"Train: {tr.shape[0]:,} | Val: {val.shape[0]:,}")
print(f"Val no-show: {val[TARGET].mean():.4f}")

# =========================================================
# ROLLING FEATURES (TR + VAL İÇİN ZAMAN SIRALI)
# =========================================================
tr["_split"] = "tr"
val["_split"] = "val"

combined = pd.concat([tr, val], axis=0).sort_values(DATE_COL).reset_index(drop=True)
spec_col_combined = resolve_specialty_col(combined)

# 1) Hasta expanding no-show
combined["_cn"] = (
    combined.groupby("patient_id")[TARGET]
    .transform(lambda x: x.shift(1).expanding().sum())
    .fillna(0)
)
combined["_ca"] = combined.groupby("patient_id").cumcount()
combined["dynamic_noshow_rate"] = (
    combined["_cn"] / combined["_ca"].replace(0, np.nan)
).fillna(0)

# 2) Son 3 randevu
combined["patient_roll3"] = (
    combined.groupby("patient_id")[TARGET]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    .fillna(combined["dynamic_noshow_rate"])
)

# 3) Son 5 randevu
combined["patient_roll5"] = (
    combined.groupby("patient_id")[TARGET]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    .fillna(combined["dynamic_noshow_rate"])
)

# 4) Klinik expanding
combined["_cli_cn"] = (
    combined.groupby("clinic_id")[TARGET]
    .transform(lambda x: x.shift(1).expanding().sum())
    .fillna(0)
)
combined["_cli_ca"] = combined.groupby("clinic_id").cumcount()
combined["clinic_noshow_rate"] = (
    combined["_cli_cn"] / combined["_cli_ca"].replace(0, np.nan)
).fillna(train[TARGET].mean())

# 5) Specialty expanding
combined["_sp_cn"] = (
    combined.groupby(spec_col_combined)[TARGET]
    .transform(lambda x: x.shift(1).expanding().sum())
    .fillna(0)
)
combined["_sp_ca"] = combined.groupby(spec_col_combined).cumcount()
combined["specialty_roll_rate"] = (
    combined["_sp_cn"] / combined["_sp_ca"].replace(0, np.nan)
).fillna(train[TARGET].mean())

# 6) Hasta son randevusundan beri gün
combined["days_since_last_appt"] = (
    combined.groupby("patient_id")[DATE_COL].diff().dt.days
).fillna(999)

# 7) Hasta bu ay kaçıncı randevusu
combined["patient_monthly_appt_count"] = (
    combined.groupby(["patient_id", combined[DATE_COL].dt.to_period("M")]).cumcount()
)

# 8) Klinik o gün kaçıncı randevu
combined["clinic_daily_rank"] = (
    combined.groupby(["clinic_id", combined[DATE_COL].dt.date]).cumcount()
)

combined = combined.drop(
    columns=["_cn", "_ca", "_cli_cn", "_cli_ca", "_sp_cn", "_sp_ca"]
)

tr = combined[combined["_split"] == "tr"].drop(columns=["_split"]).copy()
val = combined[combined["_split"] == "val"].drop(columns=["_split"]).copy()

# =========================================================
# TEST'E ROLLING FEATURE TAŞIMA
# =========================================================
spec_col_tr = resolve_specialty_col(tr)

ROLL_PATIENT_COLS = [
    "dynamic_noshow_rate",
    "patient_roll3",
    "patient_roll5",
]

global_means = {c: tr[c].mean() for c in [
    "dynamic_noshow_rate",
    "patient_roll3",
    "patient_roll5",
    "clinic_noshow_rate",
    "specialty_roll_rate",
]}

# patient-based rolling
patient_last = (
    tr.sort_values(DATE_COL)
    .groupby("patient_id")[ROLL_PATIENT_COLS]
    .last()
    .reset_index()
)
test = test.merge(patient_last, on="patient_id", how="left")
for c in ROLL_PATIENT_COLS:
    test[c] = test[c].fillna(global_means[c])

# patient last appointment date -> test gap
patient_last_date = (
    tr.sort_values(DATE_COL)
    .groupby("patient_id")[DATE_COL]
    .last()
)
test["days_since_last_appt"] = (
    test[DATE_COL] - test["patient_id"].map(patient_last_date)
).dt.days
test["days_since_last_appt"] = test["days_since_last_appt"].fillna(999).clip(lower=0)

# clinic-based rolling
clinic_last = (
    tr.sort_values(DATE_COL)
    .groupby("clinic_id")["clinic_noshow_rate"]
    .last()
)
test["clinic_noshow_rate"] = test["clinic_id"].map(clinic_last).fillna(global_means["clinic_noshow_rate"])

# specialty-based rolling
spec_last = (
    tr.sort_values(DATE_COL)
    .groupby(spec_col_tr)["specialty_roll_rate"]
    .last()
)
test["specialty_roll_rate"] = test[spec_col_tr].map(spec_last).fillna(global_means["specialty_roll_rate"])

# test içi bilinebilen sayaçlar
test["patient_monthly_appt_count"] = (
    test.sort_values(DATE_COL)
    .groupby(["patient_id", test[DATE_COL].dt.to_period("M")])
    .cumcount()
)

test["clinic_daily_rank"] = (
    test.sort_values(DATE_COL)
    .groupby(["clinic_id", test[DATE_COL].dt.date])
    .cumcount()
)

print("\nRolling features eklendi:")
for c in ["dynamic_noshow_rate", "patient_roll3", "patient_roll5", "clinic_noshow_rate", "specialty_roll_rate"]:
    print(f"  {c:<25} tr={tr[c].mean():.3f} | val={val[c].mean():.3f} | test={test[c].mean():.3f}")

# =========================================================
# FEATURE SET
# =========================================================
DROP_CANDIDATES = [
    "appointment_id",
    "patient_id",
    "clinic_id",
    DATE_COL,
    "booking_datetime",
    TARGET,
    "specialty_x",
    "specialty_y",
    "specialty",
    "residence_lat",
    "residence_lon",
    "clinic_lat",
    "clinic_lon",
    "area_id",
    "lead_time_days",
    "prior_noshow_count",
    "distance_bucket",
    "lead_time_log",
    "is_overloaded",
    "_split",
]

DROP_COLS = [c for c in DROP_CANDIDATES if c in tr.columns]

CAT_COLS = ["booking_channel", "appointment_type", "sex", "lead_time_bucket"]

features = [c for c in tr.columns if c not in DROP_COLS]
cat_cols = [c for c in CAT_COLS if c in features]

# Test'te eksik feature varsa ekle
for c in features:
    if c not in test.columns:
        test[c] = 0

X_tr = tr[features].copy()
y_tr = tr[TARGET].copy()
X_val = val[features].copy()
y_val = val[TARGET].copy()
X_test = test[features].copy()

# categorical cast
for c in cat_cols:
    X_tr[c] = X_tr[c].astype("category")
    X_val[c] = X_val[c].astype("category")
    X_test[c] = X_test[c].astype("category")

print(f"\nFeature count: {len(features)}")
print(f"Categorical  : {cat_cols}")

# =========================================================
# LIGHTGBM
# =========================================================
model_lgb = lgb.LGBMClassifier(
    n_estimators=10000,
    learning_rate=0.02,
    max_depth=8,
    num_leaves=127,
    min_child_samples=30,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)

model_lgb.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="average_precision",
    categorical_feature=cat_cols,
    callbacks=[
        lgb.early_stopping(200, verbose=False),
        lgb.log_evaluation(200),
    ],
)

lgb_val = model_lgb.predict_proba(X_val)[:, 1]
lgb_test = model_lgb.predict_proba(X_test)[:, 1]
lgb_auc = average_precision_score(y_val, lgb_val)

print(f"\nLGB PR-AUC : {lgb_auc:.5f}")
print(f"Best iter  : {model_lgb.best_iteration_}")
print(f"Val mean   : {lgb_val.mean():.4f}")
print(f"Test mean  : {lgb_test.mean():.4f}")

# =========================================================
# XGBOOST
# =========================================================
X_tr_x = X_tr.copy()
X_val_x = X_val.copy()
X_test_x = X_test.copy()

for c in cat_cols:
    X_tr_x[c] = X_tr_x[c].cat.codes.astype("int32")
    X_val_x[c] = X_val_x[c].cat.codes.astype("int32")
    X_test_x[c] = X_test_x[c].cat.codes.astype("int32")

scale = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
print(f"\nscale_pos_weight: {scale:.2f}")

model_xgb = xgb.XGBClassifier(
    n_estimators=10000,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
    eval_metric="aucpr",
    early_stopping_rounds=200,
    tree_method="hist",
    scale_pos_weight=scale,
)

model_xgb.fit(
    X_tr_x,
    y_tr,
    eval_set=[(X_val_x, y_val)],
    verbose=200,
)

xgb_val = model_xgb.predict_proba(X_val_x)[:, 1]
xgb_test = model_xgb.predict_proba(X_test_x)[:, 1]
xgb_auc = average_precision_score(y_val, xgb_val)

print(f"\nXGB PR-AUC : {xgb_auc:.5f}")
print(f"Best iter  : {model_xgb.best_iteration}")
print(f"Val mean   : {xgb_val.mean():.4f}")
print(f"Test mean  : {xgb_test.mean():.4f}")

# =========================================================
# CATBOOST POOL
# =========================================================
X_tr_c = X_tr.copy()
X_val_c = X_val.copy()
X_test_c = X_test.copy()

for c in cat_cols:
    X_tr_c[c] = X_tr_c[c].astype(str)
    X_val_c[c] = X_val_c[c].astype(str)
    X_test_c[c] = X_test_c[c].astype(str)

cat_idx = [X_tr_c.columns.get_loc(c) for c in cat_cols]

train_pool = Pool(X_tr_c, y_tr, cat_features=cat_idx)
val_pool = Pool(X_val_c, y_val, cat_features=cat_idx)
test_pool = Pool(X_test_c, cat_features=cat_idx)

print(f"\nCatBoost pool hazır: train={X_tr_c.shape}, val={X_val_c.shape}, test={X_test_c.shape}")

# GPU kontrolü
CAT_USE_GPU, gpu_err = check_catboost_gpu(train_pool, val_pool)
if CAT_USE_GPU:
    print("CatBoost device: GPU (devices='0')")
else:
    print("CatBoost device: CPU fallback")
    print("GPU error:", gpu_err)

# =========================================================
# CATBOOST ANA MODEL
# =========================================================
cat_base_params = dict(
    iterations=10000,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=3,
    random_seed=SEED,
    verbose=200,
    eval_metric="PRAUC",
    od_type="Iter",
    od_wait=200,
)

model_cat = build_cat_model(cat_base_params, use_gpu=CAT_USE_GPU)
model_cat.fit(train_pool, eval_set=val_pool, use_best_model=True)

cat_val = model_cat.predict_proba(val_pool)[:, 1]
cat_test = model_cat.predict_proba(test_pool)[:, 1]
cat_auc = average_precision_score(y_val, cat_val)

print(f"\nCAT PR-AUC : {cat_auc:.5f}")
print(f"Best iter  : {model_cat.best_iteration_}")
print(f"Val mean   : {cat_val.mean():.4f}")
print(f"Test mean  : {cat_test.mean():.4f}")

# =========================================================
# ENSEMBLE (DİREKT ORTALAMA AĞIRLIK TARAMA)
# =========================================================
print("\nDirekt ortalama ensemble:")
best_auc = -1
best_w = None
best_pred = None

for w1 in np.arange(0.1, 0.9, 0.1):
    for w2 in np.arange(0.1, 1.0 - w1, 0.1):
        w3 = round(1.0 - w1 - w2, 2)
        if w3 <= 0:
            continue

        ens_val = w1 * lgb_val + w2 * xgb_val + w3 * cat_val
        auc = average_precision_score(y_val, ens_val)

        if auc > best_auc:
            best_auc = auc
            best_w = (w1, w2, w3)
            best_pred = w1 * lgb_test + w2 * xgb_test + w3 * cat_test

print(f"  En iyi ağırlık: LGB({best_w[0]}) + XGB({best_w[1]}) + CAT({best_w[2]})")
print(f"  En iyi Val AP : {best_auc:.5f}")
print(f"  Test mean     : {best_pred.mean():.4f}")
print(f"  Cat tek farkı : {best_auc - cat_auc:+.5f}")

# =========================================================
# OPTUNA - 30 TRIAL
# =========================================================
print("\n===== OPTUNA BAŞLIYOR (30 trial) =====")

def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("lr", 0.005, 0.05, log=True),
        "depth": trial.suggest_int("depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging", 0.0, 1.0),
        "random_strength": trial.suggest_float("rs", 0.0, 2.0),
        "border_count": trial.suggest_categorical("border", [64, 128, 254]),
        "random_seed": SEED,
        "verbose": False,
        "eval_metric": "PRAUC",
        "od_type": "Iter",
        "od_wait": 100,
    }

    m = build_cat_model(params, use_gpu=CAT_USE_GPU)
    m.fit(train_pool, eval_set=val_pool, use_best_model=True)

    preds = m.predict_proba(val_pool)[:, 1]
    score = average_precision_score(y_val, preds)

    print(
        f"Trial {trial.number:02d} | "
        f"AP={score:.5f} | "
        f"best_iter={m.best_iteration_} | "
        f"params={{"
        f"'iterations': {params['iterations']}, "
        f"'learning_rate': {params['learning_rate']:.5f}, "
        f"'depth': {params['depth']}, "
        f"'l2_leaf_reg': {params['l2_leaf_reg']:.4f}, "
        f"'bagging_temperature': {params['bagging_temperature']:.4f}, "
        f"'random_strength': {params['random_strength']:.4f}, "
        f"'border_count': {params['border_count']}"
        f"}}"
    )

    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\n===== OPTUNA SONUÇLARI =====")
print(f"En iyi skor   : {study.best_value:.6f}")
print(f"En iyi params : {study.best_params}")

print("\nTop 10 trial:")
valid_trials = [t for t in study.trials if t.value is not None]
valid_trials = sorted(valid_trials, key=lambda t: t.value, reverse=True)

for i, t in enumerate(valid_trials[:10], 1):
    print(
        f"{i:02d}. trial={t.number:02d} | "
        f"value={t.value:.6f} | "
        f"params={t.params}"
    )

# =========================================================
# DOSYA ÇIKTILARI
# =========================================================
submission_cat = pd.DataFrame({
    "appointment_id": test_raw["appointment_id"].values,
    "label_noshow": cat_test,
})
submission_cat.to_csv("submission_catboost_gpu.csv", index=False)

submission_ens = pd.DataFrame({
    "appointment_id": test_raw["appointment_id"].values,
    "label_noshow": best_pred,
})
submission_ens.to_csv("submission_ensemble.csv", index=False)

print("\nKaydedildi:")
print(" - submission_catboost_gpu.csv")
print(" - submission_ensemble.csv")

print("\nCatBoost test dağılımı")
print(f"   min  : {cat_test.min():.4f}")
print(f"   mean : {cat_test.mean():.4f}")
print(f"   max  : {cat_test.max():.4f}")
print(f"   <0.3 : {(cat_test < 0.3).mean() * 100:.1f}%")
print(f"   >0.7 : {(cat_test > 0.7).mean() * 100:.1f}%")