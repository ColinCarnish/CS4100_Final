from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.forest.forest import (load_model_artifact, mae, prepare_features,
                                      r2_score, rmse)


def load_eval_sample(
    csv_path: str,
    sample_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "delay_sec" not in df.columns:
        raise ValueError("Expected 'delay_sec' in evaluation dataset.")

    if sample_rows > 0 and sample_rows < len(df):
        df = df.sample(n=sample_rows, random_state=random_seed)

    y = pd.to_numeric(df["delay_sec"], errors="coerce")
    valid_mask = y.notna()
    cleaned = df.loc[valid_mask].copy()
    cleaned["delay_sec"] = y.loc[valid_mask].astype(np.float64)
    return cleaned


def success_rate_within(y_true: np.ndarray, y_pred: np.ndarray, tolerance_sec: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred) <= tolerance_sec) * 100.0)


def permutation_importance_mae(
    model,
    x_eval: np.ndarray,
    y_true: np.ndarray,
    feature_names: list[str],
    random_seed: int = 42,
) -> pd.Series:
    rng = np.random.default_rng(random_seed)
    base_pred = model.predict(x_eval)
    base_mae = mae(y_true, base_pred)
    importances: list[float] = []

    for col_idx in range(x_eval.shape[1]):
        shuffled = x_eval.copy()
        shuffled[:, col_idx] = rng.permutation(shuffled[:, col_idx])
        shuffled_pred = model.predict(shuffled)
        importances.append(mae(y_true, shuffled_pred) - base_mae)

    return pd.Series(importances, index=feature_names).sort_values()


def save_diagnostics_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    importance_series: pd.Series,
    out_path: str,
    sample_size: int,
    r2_value: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Random Forest Regressor Diagnostics\n"
        f"(evaluation sample n={sample_size:,})",
        fontsize=12,
    )

    colors = ["tomato" if v > 0 else "lightgray" for v in importance_series.values]
    axes[0].barh(importance_series.index, importance_series.values, color=colors)
    axes[0].set_xlabel("Increase in MAE when feature is shuffled (seconds)")
    axes[0].set_title("Feature Importance (permutation)")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].scatter(y_true, y_pred, alpha=0.2, s=8, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
    axes[1].set_xlabel("Actual Delay (sec)")
    axes[1].set_ylabel("Predicted Delay (sec)")
    axes[1].set_title(f"Actual vs Predicted  (R2={r2_value:.3f})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved random forest artifact on sampled rows."
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default="src/models/forest/artifacts/random_forest_delay_model.pkl",
        help="Path to saved model artifact (.pkl).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="Datasets/final_data.csv",
        help="Path to evaluation CSV with delay_sec.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=100_000,
        help="Number of rows to sample for evaluation. Set <=0 to use all rows.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--tolerances",
        type=str,
        default="30,60,120,300",
        help="Comma-separated tolerance thresholds in seconds for success rate.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default="src/models/forest/artifacts/results/forest_eval_plot.png",
        help="Path for diagnostics plot PNG.",
    )
    parser.add_argument(
        "--plot-sample-rows",
        type=int,
        default=10_000,
        help="Rows to use for permutation-importance/plot calculations. Set <=0 to use all eval rows.",
    )
    args = parser.parse_args()

    artifact = load_model_artifact(args.artifact_path)
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    sample_rows = args.sample_rows if args.sample_rows and args.sample_rows > 0 else 0
    eval_df = load_eval_sample(args.data_path, sample_rows, args.random_seed)

    x_eval, _ = prepare_features(eval_df, feature_cols=feature_names)
    y_true = eval_df["delay_sec"].to_numpy(dtype=np.float64)
    y_pred = model.predict(x_eval)
    r2_value = r2_score(y_true, y_pred)
    abs_err = np.abs(y_true - y_pred)

    print(f"Artifact: {Path(args.artifact_path)}")
    print(f"Data: {Path(args.data_path)}")
    print(f"Rows evaluated: {len(eval_df)}")
    print("")
    print(f"MAE (sec):  {mae(y_true, y_pred):.4f}")
    print(f"RMSE (sec): {rmse(y_true, y_pred):.4f}")
    print(f"R2:         {r2_value:.4f}")
    print(f"Median absolute error (sec): {float(np.median(abs_err)):.4f}")
    print(f"Mean absolute error (min):   {float(np.mean(abs_err) / 60.0):.4f}")

    tolerances = [float(v.strip()) for v in args.tolerances.split(",") if v.strip()]
    print("")
    print("Success rates:")
    for tol in tolerances:
        rate = success_rate_within(y_true, y_pred, tol)
        print(f"  within {tol:.0f}s: {rate:.2f}%")

    if args.plot_sample_rows and args.plot_sample_rows > 0 and args.plot_sample_rows < len(eval_df):
        rng = np.random.default_rng(args.random_seed)
        plot_idx = rng.choice(len(eval_df), size=args.plot_sample_rows, replace=False)
        x_plot = x_eval[plot_idx]
        y_plot = y_true[plot_idx]
        pred_plot = y_pred[plot_idx]
    else:
        x_plot = x_eval
        y_plot = y_true
        pred_plot = y_pred

    importance_series = permutation_importance_mae(
        model,
        x_plot,
        y_plot,
        feature_names,
        random_seed=args.random_seed,
    )
    save_diagnostics_plot(
        y_plot,
        pred_plot,
        importance_series,
        args.plot_path,
        sample_size=len(y_plot),
        r2_value=r2_score(y_plot, pred_plot),
    )
    print("")
    print(f"Plot saved: {Path(args.plot_path)}")


if __name__ == "__main__":
    main()
