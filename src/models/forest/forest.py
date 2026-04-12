from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import \
    train_test_split as sklearn_train_test_split

LOGGER = logging.getLogger("forest_regressor")
CANDIDATE_FEATURES = [
    "stop_sequence",
    "direction_id",
    "event_type",
    "stop_id_bucket",
    "route_id_Blue",
    "route_id_Orange",
    "route_id_Red",
    "scheduled_time_sec",
    "hour",
    "hour_sin",
    "hour_cos",
    "is_peak_hour",
    "is_weekend",
    "month",
    "day_of_month",
    "day_of_week_monday",
    "day_of_week_tuesday",
    "day_of_week_wednesday",
    "day_of_week_thursday",
    "day_of_week_friday",
    "day_of_week_saturday",
    "day_of_week_sunday",
]
DAY_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]

DAY_OF_WEEK_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
EVENT_TYPE_MAP = {
    "ARR": 0,
    "DEP": 1,
}


@dataclass
class TreeNode:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[float] = None

    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeRegressorScratch:
    def __init__(
        self,
        max_depth: int = 12,
        min_samples_split: int = 20,
        min_samples_leaf: int = 5,
        max_features: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.root: Optional[TreeNode] = None
        self.feature_use_count: Optional[np.ndarray] = None
        self.feature_gain_sum: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n_features = x.shape[1]
        self.feature_use_count = np.zeros(n_features, dtype=np.int64)
        self.feature_gain_sum = np.zeros(n_features, dtype=np.float64)
        self.root = self._build_tree(x, y, depth=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree must be fit before predict.")
        return np.array([self._predict_row(row, self.root) for row in x], dtype=np.float64)

    def _predict_row(self, row: np.ndarray, node: TreeNode) -> float:
        while not node.is_leaf():
            if row[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n_samples = x.shape[0]
        leaf_value = float(np.mean(y))
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.all(y == y[0])
        ):
            return TreeNode(value=leaf_value)

        split = self._best_split(x, y)
        if split is None:
            return TreeNode(value=leaf_value)

        feature_idx, threshold, gain = split
        left_mask = x[:, feature_idx] <= threshold
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return TreeNode(value=leaf_value)

        if self.feature_use_count is not None:
            self.feature_use_count[feature_idx] += 1
        if self.feature_gain_sum is not None:
            self.feature_gain_sum[feature_idx] += gain

        left_node = self._build_tree(x[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(x[right_mask], y[right_mask], depth + 1)
        return TreeNode(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left_node,
            right=right_node,
        )

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> Optional[tuple[int, float, float]]:
        n_samples, n_features = x.shape
        max_features = self.max_features or int(np.sqrt(n_features))
        max_features = max(1, min(max_features, n_features))
        feature_candidates = self.rng.choice(n_features, size=max_features, replace=False)

        parent_var = np.var(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_idx in feature_candidates:
            values = x[:, feature_idx]
            unique_vals = np.unique(values)
            if unique_vals.shape[0] < 2:
                continue

            if unique_vals.shape[0] > 128:
                quantiles = np.linspace(0.05, 0.95, num=20)
                thresholds = np.quantile(values, quantiles)
                thresholds = np.unique(thresholds)
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for threshold in thresholds:
                left_mask = values <= threshold
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]
                left_var = np.var(y_left)
                right_var = np.var(y_right)
                weighted_var = (n_left / n_samples) * left_var + (n_right / n_samples) * right_var
                gain = parent_var - weighted_var

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = float(threshold)

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_gain


class RandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 20,
        max_depth: int = 12,
        min_samples_split: int = 20,
        min_samples_leaf: int = 5,
        max_features: Optional[int] = None,
        bootstrap_ratio: float = 1.0,
        random_state: int = 42,
        verbose: bool = True,
        log_every: int = 5,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap_ratio = bootstrap_ratio
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose
        self.log_every = max(1, log_every)
        self.trees: list[DecisionTreeRegressorScratch] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.split_usage_frequency_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = x.shape
        bootstrap_size = max(1, int(n_samples * self.bootstrap_ratio))
        self.trees = []
        feature_counts = np.zeros(n_features, dtype=np.float64)
        feature_gains = np.zeros(n_features, dtype=np.float64)
        if self.verbose:
            LOGGER.info(
                "Training random forest with %d trees | samples=%d | features=%d | bootstrap_size=%d",
                self.n_estimators,
                n_samples,
                n_features,
                bootstrap_size,
            )

        for tree_idx in range(self.n_estimators):
            indices = self.rng.integers(0, n_samples, size=bootstrap_size)
            x_bootstrap = x[indices]
            y_bootstrap = y[indices]

            tree_seed = int(self.rng.integers(0, 1_000_000_000))
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            tree.fit(x_bootstrap, y_bootstrap)
            self.trees.append(tree)

            if tree.feature_use_count is not None:
                feature_counts += tree.feature_use_count
            if tree.feature_gain_sum is not None:
                feature_gains += tree.feature_gain_sum
            if self.verbose and (
                (tree_idx + 1) % self.log_every == 0 or tree_idx == self.n_estimators - 1
            ):
                LOGGER.info("Finished tree %d/%d", tree_idx + 1, self.n_estimators)

        count_total = feature_counts.sum()
        gain_total = feature_gains.sum()
        self.split_usage_frequency_ = (
            feature_counts / count_total if count_total > 0 else np.zeros(n_features, dtype=np.float64)
        )
        self.feature_importances_ = (
            feature_gains / gain_total if gain_total > 0 else np.zeros(n_features, dtype=np.float64)
        )
        if self.verbose:
            LOGGER.info("Random forest training complete.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Forest must be fit before predict.")
        tree_predictions = np.array([tree.predict(x) for tree in self.trees], dtype=np.float64)
        return np.mean(tree_predictions, axis=0)


# Backward-compatible alias for older pickle artifacts.
RandomForestRegressorScratch = RandomForestRegressor


def _encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    if "service_date" in encoded.columns:
        service_dt = pd.to_datetime(encoded["service_date"], errors="coerce")
        encoded["month"] = service_dt.dt.month.fillna(0).astype(np.int32)
        encoded["day_of_month"] = service_dt.dt.day.fillna(0).astype(np.int32)
    if "day_of_week" in encoded.columns:
        day_values = encoded["day_of_week"].astype(str).str.strip().str.lower()
        for day_name in DAY_NAMES:
            encoded[f"day_of_week_{day_name}"] = (day_values == day_name).astype(np.int32)
        encoded["is_weekend"] = day_values.isin(["saturday", "sunday"]).astype(np.int32)
    if "event_type" in encoded.columns:
        event_values = encoded["event_type"].astype(str).str.strip().str.upper()
        encoded["event_type"] = event_values.map(EVENT_TYPE_MAP).fillna(-1).astype(np.int32)
    if "hour" in encoded.columns:
        hour_numeric = pd.to_numeric(encoded["hour"], errors="coerce").fillna(0.0)
    else:
        hour_numeric = pd.Series(np.zeros(len(encoded)), index=encoded.index, dtype=float)
    encoded["hour"] = hour_numeric
    hour_angle = 2.0 * np.pi * (hour_numeric % 24.0) / 24.0
    encoded["hour_sin"] = np.sin(hour_angle)
    encoded["hour_cos"] = np.cos(hour_angle)
    encoded["is_peak_hour"] = hour_numeric.isin([7, 8, 9, 16, 17, 18, 19]).astype(np.int32)
    if "direction_id" in encoded.columns:
        encoded["direction_id"] = pd.to_numeric(
            encoded["direction_id"],
            errors="coerce",
        ).fillna(-1).astype(np.int32)
    if "departure_time_sec" in encoded.columns or "arrival_time_sec" in encoded.columns:
        dep_raw = encoded["departure_time_sec"] if "departure_time_sec" in encoded.columns else np.nan
        arr_raw = encoded["arrival_time_sec"] if "arrival_time_sec" in encoded.columns else np.nan
        dep = pd.to_numeric(dep_raw, errors="coerce")
        arr = pd.to_numeric(arr_raw, errors="coerce")
        encoded["scheduled_time_sec"] = dep.fillna(arr)
    if "stop_id" in encoded.columns:
        stop_str = encoded["stop_id"].astype(str)
        stop_hash = pd.util.hash_pandas_object(stop_str, index=False).to_numpy(dtype=np.uint64)
        encoded["stop_id_bucket"] = (stop_hash % 1024).astype(np.int32)
    if "route_id" in encoded.columns:
        route_values = encoded["route_id"].astype(str).str.strip().str.lower()
        encoded["route_id_Blue"] = (route_values == "blue").astype(np.int32)
        encoded["route_id_Orange"] = (route_values == "orange").astype(np.int32)
        encoded["route_id_Red"] = (route_values == "red").astype(np.int32)
    return encoded


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    working = _encode_categorical_columns(df)
    if feature_cols is None:
        feature_cols = [col for col in CANDIDATE_FEATURES if col in working.columns]
    if not feature_cols:
        raise ValueError("No usable feature columns found in dataset.")

    for col in feature_cols:
        if col not in working.columns:
            working[col] = np.nan

    x = working[feature_cols].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    return x.to_numpy(dtype=np.float64), feature_cols


def load_features_and_target(
    csv_path: str,
    max_rows: Optional[int] = None,
    sample_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], Optional[np.ndarray]]:
    LOGGER.info("Loading dataset from %s", csv_path)
    df = pd.read_csv(csv_path)
    original_rows = df.shape[0]
    if max_rows is not None and 0 < max_rows < original_rows:
        LOGGER.info(
            "Sampling %d rows from %d total rows for faster development.",
            max_rows,
            original_rows,
        )
        df = df.sample(n=max_rows, random_state=sample_seed)
    else:
        LOGGER.info("Using all %d rows.", original_rows)

    if "delay_sec" not in df.columns:
        raise ValueError("Expected target column 'delay_sec' in dataset.")

    x, feature_cols = prepare_features(df)
    y = pd.to_numeric(df["delay_sec"], errors="coerce")
    timestamps = pd.to_datetime(df["service_date"], errors="coerce")

    valid_mask = y.notna()
    invalid_target_rows = int((~valid_mask).sum())
    valid_mask_np = valid_mask.to_numpy(dtype=bool)
    x = x[valid_mask_np]
    y = y.loc[valid_mask]
    timestamp_np = timestamps.loc[valid_mask].to_numpy()
    valid_time_mask = ~pd.isna(timestamp_np)
    x = x[valid_time_mask]
    y = y.to_numpy(dtype=np.float64)[valid_time_mask]
    timestamp_np = timestamp_np[valid_time_mask]

    LOGGER.info(
        "Prepared dataset with %d rows and %d features. Target=delay_sec | dropped_invalid_target_rows=%d",
        x.shape[0],
        x.shape[1],
        invalid_target_rows,
    )
    LOGGER.info(
        "Date range after filtering: %s to %s",
        str(np.min(timestamp_np))[:10],
        str(np.max(timestamp_np))[:10],
    )

    return x, y, feature_cols, timestamp_np


def save_model_artifact(
    model: RandomForestRegressor,
    feature_names: list[str],
    artifact_path: str,
) -> None:
    artifact = {
        "model": model,
        "feature_names": feature_names,
        "target_name": "delay_sec",
        "event_type_map": EVENT_TYPE_MAP,
        "day_of_week_map": DAY_OF_WEEK_MAP,
    }
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        pickle.dump(artifact, file_obj)
    LOGGER.info("Saved model artifact to %s", path)


class _ArtifactUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module in {
            "__main__",
            "forest",
            "models.forest.forest",
            "src.models.forest",
            "src.models.forest.forest",
        } and name in {
            "RandomForestRegressor",
            "RandomForestRegressorScratch",
            "DecisionTreeRegressorScratch",
            "TreeNode",
        }:
            module = __name__
        return super().find_class(module, name)


def load_model_artifact(artifact_path: str) -> dict:
    path = Path(artifact_path)
    with path.open("rb") as file_obj:
        artifact = _ArtifactUnpickler(file_obj).load()
    if "model" not in artifact or "feature_names" not in artifact:
        raise ValueError("Artifact is missing required keys: 'model' and 'feature_names'.")
    return artifact


def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x.shape[0] != y.shape[0]:
        raise ValueError("Feature and target rows must match.")
    x_train, x_test, y_train, y_test = sklearn_train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return x_train, x_test, y_train, y_test


def time_based_train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (x.shape[0] == y.shape[0] == timestamps.shape[0]):
        raise ValueError("Feature, target, and timestamp rows must match.")
    order = np.argsort(timestamps.astype("datetime64[ns]"))
    split_idx = int(x.shape[0] * (1.0 - test_size))
    train_idx = order[:split_idx]
    test_idx = order[split_idx:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scratch random forest regressor.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=300_000,
        help="Max rows to use for development runs. Set <=0 to use full dataset.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for row sampling.",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default="src/models/forest/artifacts/random_forest_delay_model.pkl",
        help="Where to save the trained model artifact pickle file.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=25,
        help="Number of trees in the forest.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of prepared data to hold out for test.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["time", "random"],
        default="time",
        help="Train/test split strategy.",
    )
    parser.add_argument(
        "--max-features-mode",
        type=str,
        choices=["sqrt", "all", "log2"],
        default="sqrt",
        help="How many candidate features each tree split can consider.",
    )
    parser.add_argument(
        "--max-features-count",
        type=int,
        default=0,
        help="If >0, overrides max-features-mode with an explicit feature count.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    script_path = Path(__file__).resolve()
    candidate_csv_paths = [
        script_path.parents[3] / "Datasets" / "final_data.csv",
        script_path.parents[2] / "Datasets" / "final_data.csv",
        Path.cwd() / "Datasets" / "final_data.csv",
    ]
    csv_path = next((p for p in candidate_csv_paths if p.exists()), candidate_csv_paths[0])

    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    x, y, feature_names, timestamps = load_features_and_target(
        str(csv_path),
        max_rows=max_rows,
        sample_seed=args.sample_seed,
    )
    if timestamps is None:
            LOGGER.warning("service_date not found. Falling back to random split.")
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=args.test_size,
                random_state=42,
            )
    else:
        x_train, x_test, y_train, y_test = time_based_train_test_split(
            x,
            y,
            timestamps,
            test_size=args.test_size,
        )
    LOGGER.info(
        "Train/test split complete | mode=%s | train_rows=%d | test_rows=%d",
        args.split_mode,
        x_train.shape[0],
        x_test.shape[0],
    )

    if args.max_features_count > 0:
        tree_max_features = min(args.max_features_count, x_train.shape[1])
    elif args.max_features_mode == "all":
        tree_max_features = x_train.shape[1]
    elif args.max_features_mode == "log2":
        tree_max_features = max(1, int(np.log2(x_train.shape[1])))
    else:
        tree_max_features = max(1, int(np.sqrt(x_train.shape[1])))
    LOGGER.info(
        "Using max_features=%d per split (mode=%s)",
        tree_max_features,
        args.max_features_mode,
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=14,
        min_samples_split=25,
        min_samples_leaf=8,
        max_features=tree_max_features,
        bootstrap_ratio=1.0,
        random_state=42,
        verbose=True,
        log_every=5,
    )
    LOGGER.info("Starting model fit...")
    model.fit(x_train, y_train)
    LOGGER.info("Generating predictions on test set...")
    preds = model.predict(x_test)
    LOGGER.info("Evaluation complete.")

    print(f"Rows: {x.shape[0]}, Features: {x.shape[1]}")
    print(f"MAE:  {mae(y_test, preds):.4f}")
    print(f"RMSE: {rmse(y_test, preds):.4f}")
    print(f"R2:   {r2_score(y_test, preds):.4f}")

    if model.feature_importances_ is not None:
        print("\nFeature importances (variance-reduction):")
        ordered = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda pair: pair[1],
            reverse=True,
        )
        for name, score in ordered:
            print(f"  {name:18s} {score:.4f}")
    if model.split_usage_frequency_ is not None:
        print("\nFeature split usage frequency:")
        ordered_usage = sorted(
            zip(feature_names, model.split_usage_frequency_),
            key=lambda pair: pair[1],
            reverse=True,
        )
        for name, score in ordered_usage:
            print(f"  {name:18s} {score:.4f}")

    save_model_artifact(model, feature_names, args.artifact_path)


if __name__ == "__main__":
    main()
