import os
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = r"C:\Users\Ahmet\Desktop\data-league26"
OUT_DIR = os.path.join(ROOT, "compare_bridge")

TRAIN_PATH = os.path.join(ROOT, "appointments_train.csv")
OOF_PATH = os.path.join(ROOT, "oof_catboost_mean_blend_gpu.csv")
TEST_META_PATH = os.path.join(ROOT, "submission_meta_blend.csv")

VAL_OUT = os.path.join(OUT_DIR, "my_val_predictions.csv")
TEST_OUT = os.path.join(OUT_DIR, "my_test_predictions.csv")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH, usecols=["appointment_id", "label_noshow"])
    oof = pd.read_csv(OOF_PATH, usecols=["appointment_id", "pred_mean_all"])

    val = train.merge(oof, on="appointment_id", how="inner", validate="1:1")
    val = val.dropna(subset=["pred_mean_all"]).rename(
        columns={"label_noshow": "y_true", "pred_mean_all": "pred"}
    )
    val = val[["appointment_id", "y_true", "pred"]]

    ap = average_precision_score(val["y_true"], val["pred"])

    test = pd.read_csv(TEST_META_PATH, usecols=["appointment_id", "label_noshow"])

    val.to_csv(VAL_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    print(f"validation_ap={ap:.15f}")
    print(f"val_rows={len(val)}")
    print(f"test_rows={len(test)}")
    print(f"val_cols={list(val.columns)}")
    print(f"test_cols={list(test.columns)}")
    print(f"val_out={VAL_OUT}")
    print(f"test_out={TEST_OUT}")


if __name__ == "__main__":
    main()
