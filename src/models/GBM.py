import zipfile
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# data container that holds the following:
# which feature to split on
# value to split at
# left child node
# right child node
# leaf prediction
class DecisionTreeNode:

    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    # checks for type
    def is_leaf(self):
        return self.value is not None


# building regression tree for predictions
class DecisionTree:

    def __init__(self, max_depth=4, min_samples_leaf=20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    # entry point
    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        node = DecisionTreeNode()

        # stop conditions:
        # not enough data to split
        # all values are nearly identical
        if (depth >= self.max_depth
                or len(y) < 2 * self.min_samples_leaf
                or np.std(y) < 1e-6):
            node.value = np.mean(y)
            return node

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            node.value = np.mean(y)
            return node

        # splitting the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.value = np.mean(y)
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)

        return node

    # find the split that minimizes the weighted variance of the two groups
    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_loss = np.var(y) * len(y)  # baseline: no split

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            col = X[:, feature_idx]

            if np.nanmax(col) == np.nanmin(col):
                continue

            # sample candidate percentiles
            candidates = np.nanpercentile(col, np.linspace(5, 95, 20))
            candidates = np.unique(candidates)

            for threshold in candidates:
                left_mask = col <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < self.min_samples_leaf:
                    continue
                if right_mask.sum() < self.min_samples_leaf:
                    continue

                left_loss = np.var(y[left_mask]) * left_mask.sum()
                right_loss = np.var(y[right_mask]) * right_mask.sum()
                total_loss = left_loss + right_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._traverse(row, self.root) for row in X])

    def _traverse(self, row, node):
        if node.is_leaf():
            return node.value
        if row[node.feature_index] <= node.threshold:
            return self._traverse(row, node.left)
        else:
            return self._traverse(row, node.right)


# gradient boosting for regression
class GradientBoostingRegressor:
    def __init__(self, n_estimators=50, learning_rate=0.1,
                 max_depth=4, min_samples_leaf=20):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y, X_val=None, y_val=None):

        # start with the mean
        self.initial_pred = np.mean(y)
        y_pred = np.full(len(y), self.initial_pred)

        print(f"  Initial prediction (mean): {self.initial_pred:.2f}s")
        print(f"  {'Round':>6}  {'Train MAE':>10}  {'Val MAE':>10}")
        print(f"  {'-' * 32}")

        for i in range(self.n_estimators):
            residuals = y - y_pred

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X, residuals)

            y_pred += self.learning_rate * tree.predict(X)

            self.trees.append(tree)

            # record every 10 rounds
            if (i + 1) % 10 == 0 or i == 0:
                train_mae = np.mean(np.abs(residuals))
                if X_val is not None:
                    val_pred = self.predict(X_val)
                    val_mae = np.mean(np.abs(y_val - val_pred))
                    print(f"  {i + 1:>6}  {train_mae:>10.2f}  {val_mae:>10.2f}")
                else:
                    print(f"  {i + 1:>6}  {train_mae:>10.2f}")

        return self

    def predict(self, X):
        y_pred = np.full(len(X), self.initial_pred)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


# PICKLE FUNCTIONS

def save_model(model, feature_names, artifact_path):
    artifact = {
        "model": model,  # the trained GradientBoostingRegressor
        "feature_names": feature_names,  # feature list in training order
        "target_name": "delay_sec",  # what the model predicts
    }
    with open(artifact_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"  Model saved → {artifact_path}")


def load_model(artifact_path):
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)
    print(f"  Model loaded ← {artifact_path}")
    return artifact["model"], artifact["feature_names"], artifact["target_name"]


# load data
print("=" * 55)
print("  MBTA Gradient Boosting Delay Model ")
print("=" * 55)

print("\nLoading data...")

ZIP_PATH = r"C:\Users\ryuli\Downloads\final_data.csv.zip"
CSV_NAME = "final_data.csv"
ARTIFACT_PATH = r"C:\Users\ryuli\Downloads\gbm_delay_model.pkl"

with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open(CSV_NAME) as f:
        df = pd.read_csv(f, low_memory=False)

print(f"  Events : {len(df):,} rows")

# filter arrivals only
arr = df[df["event_type"] == "ARR"].copy()
print(f"  ARR rows : {len(arr):,}")

# encode route as integer
arr["route_enc"] = arr["route_id"].map({"Red": 0, "Orange": 1, "Blue": 2}).fillna(-1)

# encode day of week as integer
dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
           "Friday": 4, "Saturday": 5, "Sunday": 6}
arr["dow"] = arr["day_of_week"].map(dow_map)

# peak hour flag
arr["is_peak"] = arr["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

# weekend flag
arr["is_weekend"] = (arr["dow"] >= 5).astype(int)

# lag features: delay at previous stop in the same trip
arr = arr.sort_values(["trip_id", "stop_sequence"])
arr["lag_delay_1"] = arr.groupby("trip_id")["delay_sec"].shift(1)
arr["lag_delay_2"] = arr.groupby("trip_id")["delay_sec"].shift(2)
arr["cum_delay"] = arr.groupby("trip_id")["delay_sec"].cumsum().shift(1)

# normalised stop position within trip (0 = first stop, 1 = last)
arr["stop_seq_norm"] = arr.groupby("trip_id")["stop_sequence"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
)

# encode stop name as integer category
arr["stop_enc"] = arr["stop_name"].astype("category").cat.codes

# build feature list — direction_id added only if present in dataset
FEATURES = [
    "route_enc",  # line (Red/Orange/Blue)
    "stop_sequence",  # position along route
    "stop_seq_norm",  # normalised position
    "stop_enc",  # which stop
    "hour",  # hour of day
    "dow",  # day of week
    "is_weekend",  # weekend flag
    "is_peak",  # peak hour flag
    "lag_delay_1",  # delay at previous stop
    "lag_delay_2",  # delay two stops back
    "cum_delay",  # cumulative delay so far in trip
]

if "direction_id" in arr.columns:
    FEATURES.insert(1, "direction_id")

print(f"  Features ({len(FEATURES)}): {FEATURES}")

TARGET = "delay_sec"

model_df = arr[FEATURES + [TARGET]].dropna(subset=[TARGET, "lag_delay_1"])

# cap outliers at 1st and 99th percentile
p99 = model_df[TARGET].quantile(0.99)
p01 = model_df[TARGET].quantile(0.01)
model_df = model_df[
    (model_df[TARGET] >= p01) & (model_df[TARGET] <= p99)
    ].copy()

print(f"  Model rows after cleaning : {len(model_df):,}")

SAMPLE = 30_000
model_df = model_df.sample(SAMPLE, random_state=42)

np.random.seed(42)
idx = np.random.permutation(len(model_df))
split = int(0.8 * len(model_df))
train_idx = idx[:split]
test_idx = idx[split:]

X_all = model_df[FEATURES].values.astype(np.float64)
y_all = model_df[TARGET].values.astype(np.float64)

# fill NaNs with column median
col_medians = np.nanmedian(X_all, axis=0)
for col_i in range(X_all.shape[1]):
    nan_mask = np.isnan(X_all[:, col_i])
    X_all[nan_mask, col_i] = col_medians[col_i]

X_train, y_train = X_all[train_idx], y_all[train_idx]
X_test, y_test = X_all[test_idx], y_all[test_idx]

print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")

# model training
print("\nTraining Gradient Boosting...")

model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=4,
    min_samples_leaf=20,
)

model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

# save model
print("\nSaving model...")
save_model(model, FEATURES, ARTIFACT_PATH)

# evaluate the model
print("\nEvaluating...")

y_pred = model.predict(X_test)

mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\n  MAE  : {mae:.1f}s  ({mae / 60:.2f} min)")
print(f"  RMSE : {rmse:.1f}s  ({rmse / 60:.2f} min)")
print(f"  R²   : {r2:.4f}")

# feature importance
print("\nCalculating feature importances...")

base_mae = np.mean(np.abs(y_test - y_pred))
importances = []

for i, feat in enumerate(FEATURES):
    X_shuffled = X_test.copy()
    np.random.shuffle(X_shuffled[:, i])
    shuffled_pred = model.predict(X_shuffled)
    shuffled_mae = np.mean(np.abs(y_test - shuffled_pred))
    importances.append(shuffled_mae - base_mae)

importances = np.array(importances)
imp_series = pd.Series(importances, index=FEATURES).sort_values()

# plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "MBTA Delay Model\n"
    f"(50 trees, lr=0.1, max_depth=4  |  n={SAMPLE:,} sample)",
    fontsize=12
)

colors = ["tomato" if v > 0 else "lightgray" for v in imp_series.values]
axes[0].barh(imp_series.index, imp_series.values, color=colors)
axes[0].set_xlabel("Increase in MAE when feature is shuffled (seconds)")
axes[0].set_title("Feature Importance (permutation)")
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].grid(axis="x", alpha=0.3)

axes[1].scatter(y_test, y_pred, alpha=0.2, s=8, color="steelblue")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[1].plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
axes[1].set_xlabel("Actual Delay (sec)")
axes[1].set_ylabel("Predicted Delay (sec)")
axes[1].set_title(f"Actual vs Predicted  (R²={r2:.3f})")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\Users\ryuli\Downloads\mbta_results.png", dpi=150, bbox_inches="tight")
print("  Plot saved → mbta_results.png")

# EXAMPLE: 
print("\nLoading model from pickle and predicting...")
loaded_model, loaded_features, loaded_target = load_model(ARTIFACT_PATH)

print(f"  Loaded target  : {loaded_target}")
print(f"  Loaded features: {loaded_features}")

example_values = [
    0,  # route_enc     (Red=0)
    100,  # stop_sequence
    0.5,  # stop_seq_norm
    10,  # stop_enc
    8,  # hour
    1,  # dow           (Tuesday=1)
    0,  # is_weekend
    1,  # is_peak
    180,  # lag_delay_1   (3 min delay at previous stop)
    90,  # lag_delay_2
    270,  # cum_delay
]

if "direction_id" in loaded_features:
    example_values.insert(1, 1)

example = np.array([example_values])
pred = loaded_model.predict(example)[0]
print(f"  Predicted delay : {pred:.0f}s  ({pred / 60:.1f} min)")
