"""
Pseudo-cold clinic backtest.
Simulates unseen clinics by holding out the top-5 largest clinics from training,
then measuring per-model and per-blend AP on those held-out clinic rows only.

Does NOT modify main.py or any submission pipeline.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score

# Import shared logic from main.py
from main import (
    load_data,
    add_features,
    build_feature_lists,
    MODEL_CONFIGS,
    TARGET,
    ID_COL,
    TIME_COL,
    VALID_DAYS,
    RANDOM_SEED,
)

# =========================================================
# BLEND CANDIDATES TO EVALUATE
# =========================================================
BLEND_CANDIDATES = {
    "normal_blend_0.30_0.60_0.10":  {"baseline_4954": 0.30, "regularized_off": 0.60, "hour_bucket_reg_off": 0.10},
    "fine_blend_0.28_0.64_0.08":    {"baseline_4954": 0.28, "regularized_off": 0.64, "hour_bucket_reg_off": 0.08},
    "cold_A_0.00_1.00_0.00":        {"baseline_4954": 0.00, "regularized_off": 1.00, "hour_bucket_reg_off": 0.00},
    "cold_B_0.00_0.90_0.10":        {"baseline_4954": 0.00, "regularized_off": 0.90, "hour_bucket_reg_off": 0.10},
    "cold_C_0.00_0.80_0.20":        {"baseline_4954": 0.00, "regularized_off": 0.80, "hour_bucket_reg_off": 0.20},
    "cold_D_0.10_0.80_0.10":        {"baseline_4954": 0.10, "regularized_off": 0.80, "hour_bucket_reg_off": 0.10},
}


def make_time_split(df):
    """Same split logic as main.py — last 45 days = validation."""
    max_date = df["appt_date_only"].max()
    valid_start = max_date - pd.Timedelta(days=VALID_DAYS - 1)
    train_part = df[df["appt_date_only"] < valid_start].copy()
    valid_part = df[df["appt_date_only"] >= valid_start].copy()
    return train_part, valid_part


def select_pseudo_cold_clinics(train_part, valid_part, min_train=1000, min_valid=150, top_k=5):
    """Pick the top-k clinics that have enough rows in both splits."""
    train_counts = train_part.groupby("clinic_id").size().rename("train_n")
    valid_counts = valid_part.groupby("clinic_id").size().rename("valid_n")

    stats = pd.concat([train_counts, valid_counts], axis=1).dropna()
    stats = stats[(stats["train_n"] >= min_train) & (stats["valid_n"] >= min_valid)]
    stats = stats.sort_values("valid_n", ascending=False)

    selected = stats.head(top_k)
    return selected


def train_cold_model(train_part, valid_cold, cfg):
    """Train one CatBoost model and return predictions on valid_cold rows."""
    df_train = add_features(train_part, use_clinic_id=cfg["use_clinic_id"], add_hour_bucket=cfg["add_hour_bucket"])
    df_valid = add_features(valid_cold, use_clinic_id=cfg["use_clinic_id"], add_hour_bucket=cfg["add_hour_bucket"])

    features, cat_features = build_feature_lists(df_train, use_clinic_id=cfg["use_clinic_id"])

    train_pool = Pool(df_train[features], df_train[TARGET], cat_features=cat_features)
    valid_pool = Pool(df_valid[features], df_valid[TARGET], cat_features=cat_features)

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
        random_seed=RANDOM_SEED,
        od_type="Iter",
        od_wait=cfg.get("od_wait", 200),
        verbose=200,
        allow_writing_files=False,
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    preds = model.predict_proba(valid_pool)[:, 1]
    return preds


def main():
    # --- Load & merge ---
    train, _test, patients, clinics, _sample_sub = load_data()
    train = train.merge(patients, on="patient_id", how="left", validate="m:1")
    train = train.merge(clinics, on="clinic_id", how="left", validate="m:1")

    # --- Time split on raw train (before feature engineering) ---
    # We need appt_date_only for splitting
    train["appt_date_only"] = train["appointment_datetime"].dt.normalize()
    max_date = train["appt_date_only"].max()
    valid_start = max_date - pd.Timedelta(days=VALID_DAYS - 1)

    train_part = train[train["appt_date_only"] < valid_start].copy()
    valid_part = train[train["appt_date_only"] >= valid_start].copy()

    print(f"Train rows: {len(train_part):,}  |  Valid rows: {len(valid_part):,}")

    # --- Select pseudo-cold clinics (using original numeric clinic_id) ---
    selected = select_pseudo_cold_clinics(train_part, valid_part)

    if len(selected) == 0:
        print("ERROR: No clinics meet the minimum row criteria. Aborting.")
        return

    print("\n========== SELECTED PSEUDO-COLD CLINICS ==========")
    print(selected.to_string())
    print(f"\nTotal pseudo-cold valid rows: {int(selected['valid_n'].sum()):,}")

    cold_clinic_ids = selected.index.tolist()

    # --- Build cold train & valid sets ---
    # Train: remove ALL rows from cold clinics (from both train and valid periods)
    train_cold = train_part[~train_part["clinic_id"].isin(cold_clinic_ids)].copy()

    # Valid: keep ONLY cold clinic rows
    valid_cold = valid_part[valid_part["clinic_id"].isin(cold_clinic_ids)].copy()

    print(f"\nCold-train rows (cold clinics removed): {len(train_cold):,}")
    print(f"Cold-valid rows (cold clinics only):    {len(valid_cold):,}")
    print(f"Cold-valid no-show rate:                {valid_cold[TARGET].mean():.4f}")

    # --- Train each model on cold-train, predict cold-valid ---
    y_cold = valid_cold[TARGET].values
    pred_map = {}

    for cfg in MODEL_CONFIGS:
        name = cfg["name"]
        print(f"\n===== Training {name} (pseudo-cold) =====")
        preds = train_cold_model(train_cold, valid_cold, cfg)
        pred_map[name] = preds

        ap = average_precision_score(y_cold, preds)
        print(f"  {name} pseudo-cold AP: {ap:.6f}")

    # --- Evaluate all blend candidates ---
    print("\n\n========== PSEUDO-COLD BLEND RESULTS ==========")

    results = []

    # Individual models
    for name in ["baseline_4954", "regularized_off", "hour_bucket_reg_off"]:
        ap = average_precision_score(y_cold, pred_map[name])
        results.append((f"solo_{name}", ap))

    # Blend candidates
    for blend_name, weights in BLEND_CANDIDATES.items():
        blend_pred = (
            weights["baseline_4954"] * pred_map["baseline_4954"] +
            weights["regularized_off"] * pred_map["regularized_off"] +
            weights["hour_bucket_reg_off"] * pred_map["hour_bucket_reg_off"]
        )
        ap = average_precision_score(y_cold, blend_pred)
        results.append((blend_name, ap))

    # Sort best to worst
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Blend/Model':<40} {'Pseudo-Cold AP':>14}")
    print("-" * 62)
    for i, (name, ap) in enumerate(results, 1):
        print(f"{i:<6} {name:<40} {ap:>14.6f}")

    # --- Summary ---
    winner_name, winner_ap = results[0]
    normal_ap = dict(results).get("normal_blend_0.30_0.60_0.10", 0)
    fine_ap = dict(results).get("fine_blend_0.28_0.64_0.08", 0)

    print("\n========== SUMMARY ==========")
    print(f"Pseudo-cold winner:       {winner_name}  (AP={winner_ap:.6f})")
    print(f"Normal blend (0.3/0.6/0.1) cold AP:  {normal_ap:.6f}")
    print(f"Fine blend (0.28/0.64/0.08) cold AP:  {fine_ap:.6f}")

    if winner_name.startswith("normal_blend") or winner_name.startswith("fine_blend"):
        print(">> Normal validation winner = pseudo-cold winner. No cold-start adjustment needed.")
    else:
        print(f">> Pseudo-cold winner differs from normal blend!")
        print(f"   Recommended cold-start blend: {winner_name}")
        diff = winner_ap - max(normal_ap, fine_ap)
        print(f"   Cold AP uplift vs best normal blend: {diff:+.6f}")


if __name__ == "__main__":
    main()
