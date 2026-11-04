"""Final XGBoost training with chosen settings. Step 4.

Trains on as much of the train split as you ask for, scores on the VALIDATION split
(the test split is never touched), and saves the model, predictions and metrics.

Colab:  python -m complaintops.train_xgb --n-train 235000 --out /content/drive/MyDrive/xgb_final
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from complaintops.train_baseline import prepare_xgb_data, train_xgb

# Trial 3 of the Optuna search (depth 4: tested, cheap), with more rounds because
# nearly every tuning trial hit the 300-round cap before converging.
CHOSEN = dict(max_depth=4, learning_rate=0.06, subsample=0.76,
              colsample_bytree=0.60, min_child_weight=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=235_000)
    ap.add_argument("--max-tfidf", type=int, default=20_000)
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="data/xgb_final")
    args = ap.parse_args()

    data = prepare_xgb_data(args.n_train, args.max_tfidf)
    print(f"features ready in {data['prep_time']:.0f}s: {data['X_tr'].shape}", flush=True)

    f1, info, clf = train_xgb(data, CHOSEN, args.n_estimators, args.device)
    print(f"XGBoost val macro-F1 {f1:.4f}  {info}", flush=True)

    pred = clf.predict(data["X_va"])
    le = data["le"]
    y_true, y_pred = le.inverse_transform(data["y_va"]), le.inverse_transform(pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(out / "xgb_model.json"))
    pd.DataFrame({"y": y_true, "pred": y_pred}).to_parquet(out / "xgb_val_preds.parquet")
    (out / "xgb_report.txt").write_text(report)
    imp = pd.Series(clf.feature_importances_).sort_values(ascending=False).head(15)
    names = list(data["tfidf"].get_feature_names_out()) + [
        "n_chars", "n_words", "n_xxxx", "n_dollar", "day_of_week"]
    imp.index = [names[i] for i in imp.index]
    print("top features:\n", imp.round(4).to_string())
    json.dump(dict(val_macro_f1=f1, val_accuracy=accuracy_score(y_true, y_pred),
                   params=CHOSEN, n_estimators_cap=args.n_estimators, **info,
                   top_features=imp.round(4).to_dict()),
              open(out / "xgb_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
