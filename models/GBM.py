import zipfile
import numpy as np


class DecisionTreeNode:

    def __init__(self):
        self.feature_index = None  # which feature to split on
        self.threshold = None  # value to split at
        self.left = None  # left child node  (value <= threshold)
        self.right = None  # right child node (value >  threshold)
        self.value = None  # leaf prediction (set if this is a leaf)

    def is_leaf(self):
        return self.value is not None


# regression tree for predicting continuous values
class DecisionTree:

    def __init__(self, max_depth=4, min_samples_leaf=20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    # training

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)

    def _build(self, X, y, depth):
        node = DecisionTreeNode()

        if (depth >= self.max_depth
                or len(y) < 2 * self.min_samples_leaf
                or np.std(y) < 1e-6):
            node.value = np.mean(y)
            return node

        best_feature, best_threshold = self._best_split(X, y)

        # if no valid split found → leaf
        if best_feature is None:
            node.value = np.mean(y)
            return node

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # if either side is too small → leaf
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.value = np.mean(y)
            return node

        # split and recurse
        node.feature_index = best_feature
        node.threshold = best_threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)

        return node

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_loss = np.var(y) * len(y)

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            col = X[:, feature_idx]


            if np.nanmax(col) == np.nanmin(col):
                continue


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

    # predictions
    def predict(self, X):
        return np.array([self._traverse(row, self.root) for row in X])

    def _traverse(self, row, node):
        if node.is_leaf():
            return node.value
        if row[node.feature_index] <= node.threshold:
            return self._traverse(row, node.left)
        else:
            return self._traverse(row, node.right)

# gradient boosting: find disparities in errors -> learning -> update predictions
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

            # fit a tree to the residuals
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X, residuals)

            # update prediction
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
