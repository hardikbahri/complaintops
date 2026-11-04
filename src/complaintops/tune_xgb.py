"""Optuna search for XGBoost settings. Step 4.

Tunes on a train subsample, scoring on the validation split (never the test split).
Run (Colab or local):  python -m complaintops.tune_xgb --n-train 40000 --trials 30
"""
import argparse
import json
import time

import optuna

from complaintops.train_baseline import prepare_xgb_data, train_xgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40_000)
    ap.add_argument("--max-tfidf", type=int, default=20_000)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--timeout-min", type=float, default=None, help="stop the search after N minutes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="data/optuna_xgb")
    ap.add_argument("--storage", default=None,
                    help="e.g. sqlite:////content/drive/MyDrive/optuna.db - saves every trial "
                         "so a disconnected run can be resumed by running the same command again")
    args = ap.parse_args()

    t0 = time.time()
    data = prepare_xgb_data(args.n_train, args.max_tfidf)
    print(f"features ready in {data['prep_time']:.0f}s: {data['X_tr'].shape}", flush=True)

    def objective(trial):
        params = dict(
            max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        )
        f1, info, _ = train_xgb(data, params, args.n_estimators, args.device)
        trial.set_user_attr("info", info)
        print(f"trial {trial.number}: val macro-F1 {f1:.4f} {params} {info}", flush=True)
        return f1

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        storage=args.storage, study_name="xgb", load_if_exists=True)
    done = len([t for t in study.trials if t.state.is_finished()])
    if done:
        print(f"resuming: {done} trials already finished", flush=True)
    timeout = args.timeout_min * 60 if args.timeout_min else None
    study.optimize(objective, n_trials=max(args.trials - done, 0), timeout=timeout)

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(f"{args.out}_trials.csv", index=False)
    result = dict(best_value=study.best_value, best_params=study.best_params,
                  n_trials=len(study.trials), n_train=args.n_train,
                  max_tfidf=args.max_tfidf, total_minutes=round((time.time() - t0) / 60, 1))
    json.dump(result, open(f"{args.out}_best.json", "w"), indent=2)
    print("BEST:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
