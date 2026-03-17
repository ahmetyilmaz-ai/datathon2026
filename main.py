from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score
import optuna

# -----------------------------
# Normal CatBoost model
# -----------------------------
model_cat = CatBoostClassifier(
    iterations          = 10000,
    learning_rate       = 0.02,
    depth               = 8,
    l2_leaf_reg         = 3,
    random_seed         = 42,
    verbose             = 200,
    eval_metric         = 'PRAUC',
    od_type             = 'Iter',
    od_wait             = 200,
    task_type           = "GPU",
    devices             = "0",
)

model_cat.fit(train_pool, eval_set=val_pool, use_best_model=True)

cat_val  = model_cat.predict_proba(val_pool)[:, 1]
cat_test = model_cat.predict_proba(test_pool)[:, 1]
cat_auc  = average_precision_score(y_val, cat_val)

print(f"CAT PR-AUC : {cat_auc:.4f}")
print(f"Best iter  : {model_cat.best_iteration_}")
print(f"Val mean   : {cat_val.mean():.4f}")
print(f"Test mean  : {cat_test.mean():.4f}")


# -----------------------------
# Optuna tuning (30 trial)
# -----------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        'iterations':          trial.suggest_int('iterations', 500, 2000),
        'learning_rate':       trial.suggest_float('lr', 0.005, 0.05, log=True),
        'depth':               trial.suggest_int('depth', 6, 10),
        'l2_leaf_reg':         trial.suggest_float('l2', 1.0, 10.0),
        'bagging_temperature': trial.suggest_float('bagging', 0.0, 1.0),
        'random_strength':     trial.suggest_float('rs', 0.0, 2.0),
        'border_count':        trial.suggest_categorical('border', [64, 128, 254]),
    }

    m = CatBoostClassifier(
        **params,
        random_seed = 42,
        verbose     = False,
        eval_metric = 'PRAUC',
        od_type     = 'Iter',
        od_wait     = 100,
        task_type   = "GPU",
        devices     = "0",
    )

    m.fit(train_pool, eval_set=val_pool, use_best_model=True)
    preds = m.predict_proba(val_pool)[:, 1]
    score = average_precision_score(y_val, preds)

    print(
        f"Trial {trial.number:02d} | "
        f"AP={score:.5f} | "
        f"best_iter={m.best_iteration_} | "
        f"params={params}"
    )

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\n===== OPTUNA SONUÇLARI =====")
print(f"En iyi skor   : {study.best_value:.6f}")
print(f"En iyi params : {study.best_params}")

print("\nTop 10 trial:")
trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)

for i, t in enumerate(trials_sorted[:10], 1):
    print(
        f"{i:02d}. "
        f"trial={t.number} | "
        f"value={t.value:.6f} | "
        f"params={t.params}"
    )