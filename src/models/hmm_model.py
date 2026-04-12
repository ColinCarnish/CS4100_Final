import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

DATA_DIR = 'Downloads'

# Features the HMM observes
HMM_FEATURES = ['ARR', 'DEP', 'PRA', 'PRD', 'PRA_2', 'PRD_2']
 
# Additional features for the classifier
EXTRA_FEATURES = ['hour', 'day_of_week_num', 'month_num', 'prev_delayed', 'prev_delayed_2']
 
 
def load_split(filepath):
    df = pd.read_csv(filepath, dtype={'trip_id': str})
    df = df[df['event_type'] == 'ARR'].copy()
    df = df[(df['delay_min'] >= -10) & (df['delay_min'] <= 60)].copy()
    all_cols = HMM_FEATURES + EXTRA_FEATURES + ['delayed']
    df[all_cols] = df[all_cols].fillna(0)
    return df
 
 
def df_to_sequences(df, scaler=None, fit_scaler=False, chunk_size=50, min_length=10):
    """Build observation sequences for HMM + parallel raw feature rows for classifier."""
    if fit_scaler:
        scaler = StandardScaler()
        df[HMM_FEATURES] = scaler.fit_transform(df[HMM_FEATURES])
    else:
        df[HMM_FEATURES] = scaler.transform(df[HMM_FEATURES])
 
    sequences, labels, extra_feats, metadata = [], [], [], []
 
    for (route, date), group in df.groupby(['route_id', 'service_date']):
        group = group.sort_values('event_time_sec')
        obs_matrix = group[HMM_FEATURES].values
        delayed = group['delayed'].values
        extras = group[EXTRA_FEATURES].values
 
        for start in range(0, len(obs_matrix), chunk_size):
            chunk_obs = obs_matrix[start : start + chunk_size]
            chunk_lab = delayed[start : start + chunk_size]
            chunk_ext = extras[start : start + chunk_size]
            if len(chunk_obs) < min_length:
                continue
            sequences.append(chunk_obs)
            labels.append(chunk_lab)
            extra_feats.append(chunk_ext)
            metadata.append({'route_id': route, 'service_date': date})
 
    return sequences, labels, extra_feats, metadata, scaler
 
 
STATE_NAMES = ['Normal', 'Minor Delay', 'Major Delay']
 
class MultivariateGaussianHMM:
    def __init__(self, n_states=3, n_features=6, smooth=0.1, seed=42):
        self.n_states = n_states
        self.n_features = n_features
        self.smooth = smooth
        rng = np.random.default_rng(seed)
 
        self.pi = np.array([0.7, 0.2, 0.1])[:n_states]
        self.pi /= self.pi.sum()
 
        self.A = np.array([
            [0.85, 0.10, 0.05],
            [0.15, 0.70, 0.15],
            [0.05, 0.25, 0.70],
        ])[:n_states, :n_states]
        self.A += rng.uniform(0, 0.02, self.A.shape)
        self.A /= self.A.sum(axis=1, keepdims=True)
 
        self.means = np.zeros((n_states, n_features))
        for k in range(n_states):
            self.means[k, 0] = -0.5 + k * 0.5
 
        self.variances = np.ones((n_states, n_features))
 
    def _emission_prob(self, x):
        D = self.n_features
        probs = np.zeros(self.n_states)
        for k in range(self.n_states):
            diff = x - self.means[k]
            log_p = (-0.5 * np.sum(diff**2 / self.variances[k])
                     - 0.5 * np.sum(np.log(self.variances[k]))
                     - 0.5 * D * np.log(2 * np.pi))
            probs[k] = np.exp(log_p)
        return np.maximum(probs, 1e-300)
 
    def _forward(self, obs):
        T, N = len(obs), self.n_states
        alpha = np.zeros((T, N))
        c = np.zeros(T)
        alpha[0] = self.pi * self._emission_prob(obs[0])
        c[0] = alpha[0].sum()
        if c[0] > 0: alpha[0] /= c[0]
        for t in range(1, T):
            emit = self._emission_prob(obs[t])
            for j in range(N):
                alpha[t, j] = alpha[t - 1] @ self.A[:, j] * emit[j]
            c[t] = alpha[t].sum()
            if c[t] > 0: alpha[t] /= c[t]
        return alpha, c
 
    def _backward(self, obs, c):
        T, N = len(obs), self.n_states
        beta = np.zeros((T, N))
        beta[T - 1] = 1.0 / (c[T - 1] + 1e-300)
        for t in range(T - 2, -1, -1):
            emit = self._emission_prob(obs[t + 1])
            for i in range(N):
                beta[t, i] = (self.A[i, :] * emit) @ beta[t + 1]
            if c[t] > 0: beta[t] /= c[t]
        return beta
 
    def viterbi(self, obs):
        T, N = len(obs), self.n_states
        log_A = np.log(self.A + 1e-300)
        V = np.zeros((T, N))
        bp = np.zeros((T, N), dtype=int)
        V[0] = np.log(self.pi + 1e-300) + np.log(self._emission_prob(obs[0]) + 1e-300)
        for t in range(1, T):
            log_emit = np.log(self._emission_prob(obs[t]) + 1e-300)
            for j in range(N):
                scores = V[t - 1] + log_A[:, j]
                bp[t, j] = np.argmax(scores)
                V[t, j] = scores[bp[t, j]] + log_emit[j]
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        return path, V[-1, path[-1]]
 
    def get_posteriors(self, obs):
        """Get full posterior matrix (T, n_states) for use as classifier features."""
        alpha, _ = self._forward(obs)
        posteriors = alpha / (alpha.sum(axis=1, keepdims=True) + 1e-300)
        return posteriors
 
    def fit(self, sequences, max_iter=50, tol=1e-4, verbose=True):
        N, D = self.n_states, self.n_features
        history = []
 
        for it in range(max_iter):
            pi_acc = np.zeros(N)
            A_num, A_den = np.zeros((N, N)), np.zeros(N)
            mean_num = np.zeros((N, D))
            total_ll = 0.0
            gammas_cache = []
 
            for obs in sequences:
                if len(obs) < 2:
                    gammas_cache.append(None)
                    continue
                alpha, c = self._forward(obs)
                beta = self._backward(obs, c)
                total_ll += np.sum(np.log(c + 1e-300))
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
                gammas_cache.append(gamma)
 
                for t in range(len(obs) - 1):
                    emit = self._emission_prob(obs[t + 1])
                    xi = np.outer(alpha[t], emit * beta[t + 1]) * self.A
                    xi /= xi.sum() + 1e-300
                    A_num += xi
                    A_den += gamma[t]
 
                pi_acc += gamma[0]
                for t in range(len(obs)):
                    for k in range(N):
                        mean_num[k] += gamma[t, k] * obs[t]
 
            mean_den = np.zeros(N)
            for idx, obs in enumerate(sequences):
                g = gammas_cache[idx]
                if g is None: continue
                mean_den += g.sum(axis=0)
 
            self.pi = pi_acc / (pi_acc.sum() + 1e-300)
            A_learned = A_num / (A_den[:, None] + 1e-300)
            uniform = np.full((N, N), 1.0 / N)
            self.A = (1 - self.smooth) * A_learned + self.smooth * uniform
 
            new_means = np.zeros((N, D))
            for k in range(N):
                new_means[k] = mean_num[k] / (mean_den[k] + 1e-300)
 
            var_num = np.zeros((N, D))
            for idx, obs in enumerate(sequences):
                g = gammas_cache[idx]
                if g is None: continue
                for t in range(len(obs)):
                    for k in range(N):
                        var_num[k] += g[t, k] * (obs[t] - new_means[k])**2
 
            new_vars = np.zeros((N, D))
            for k in range(N):
                new_vars[k] = var_num[k] / (mean_den[k] + 1e-300)
            self.means = new_means
            self.variances = np.maximum(new_vars, 0.01)
 
            history.append(total_ll)
            if verbose and it % 5 == 0:
                print(f"  Iter {it:3d}: LL = {total_ll:.2f}")
            if it > 0 and abs(history[-1] - history[-2]) < tol:
                if verbose: print(f"  Converged at iteration {it}")
                break
 
        return history
 
    def predict_state(self, obs):
        path, _ = self.viterbi(obs)
        posteriors = self.get_posteriors(obs)
        state_probs = posteriors[-1]
        next_probs = state_probs @ self.A
        return {
            'current_state': STATE_NAMES[path[-1]],
            'state_probs': dict(zip(STATE_NAMES, state_probs.round(4))),
            'delay_probability': round(float(1 - next_probs[0]), 4),
        }
 
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'pi': self.pi, 'A': self.A,
                'means': self.means, 'variances': self.variances,
                'n_states': self.n_states, 'n_features': self.n_features,
                'smooth': self.smooth,
            }, f)
        print(f"HMM saved to {filepath}")
 
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.pi, self.A = d['pi'], d['A']
        self.means, self.variances = d['means'], d['variances']
        self.n_states, self.n_features = d['n_states'], d['n_features']
        self.smooth = d['smooth']
 

 
def build_classifier_features(hmm, sequences, extra_feats):
    """
    For each observation, create a feature vector:
        [HMM_posterior_state_0, HMM_posterior_state_1, HMM_posterior_state_2,
         extra_feat_0, extra_feat_1, ..., extra_feat_N]
 
    The HMM posteriors capture temporal/sequential information.
    The extra features add context the HMM doesn't directly model.
    """
    all_features = []
    for obs_seq, ext_seq in zip(sequences, extra_feats):
        posteriors = hmm.get_posteriors(obs_seq) 
        
        combined = np.hstack([posteriors, ext_seq]) 
        all_features.append(combined)
 
    X = np.concatenate(all_features, axis=0)
    return X
 
 
def train_classifier(X_train, y_train):
    """Train a logistic regression on HMM posteriors + extra features."""
    clf = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs',
    )
    clf.fit(X_train, y_train)
    return clf
 

 
def evaluate_hmm_native(hmm, sequences):
    total_ll, total_obs = 0.0, 0
    state_counts = np.zeros(hmm.n_states)
    for obs in sequences:
        if len(obs) < 2: continue
        _, c = hmm._forward(obs)
        total_ll += np.sum(np.log(c + 1e-300))
        total_obs += len(obs)
        path, _ = hmm.viterbi(obs)
        for s in range(hmm.n_states):
            state_counts[s] += (path == s).sum()
    print(f"Log-likelihood: {total_ll:.2f} ({total_ll / total_obs:.4f} per obs)")
    print("State distribution:")
    for name, count in zip(STATE_NAMES, state_counts):
        print(f"  {name}: {count:.0f} ({count / state_counts.sum() * 100:.1f}%)")
 
 
def evaluate_classifier(clf, X, y_true, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Evaluate the full HMM+classifier pipeline."""
    probs = clf.predict_proba(X)[:, 1]
 
    print(f"\nBinary Classification Metrics (HMM + Classifier):")
    print(f"Total observations: {len(y_true)}")
    print(f"Actual delay rate: {y_true.mean():.1%}\n")
 
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        prec_1 = precision_score(y_true, preds, zero_division=0)
        rec_1 = recall_score(y_true, preds, zero_division=0)
        f1_1 = f1_score(y_true, preds, zero_division=0)
        prec_0 = precision_score(y_true, preds, pos_label=0, zero_division=0)
        rec_0 = recall_score(y_true, preds, pos_label=0, zero_division=0)
 
        print(f"  Threshold: {thresh}")
        print(f"    Accuracy:                {acc:.4f}")
        print(f"    Precision (delayed):     {prec_1:.4f}")
        print(f"    Recall    (delayed):     {rec_1:.4f}")
        print(f"    F1        (delayed):     {f1_1:.4f}")
        print(f"    Precision (not delayed): {prec_0:.4f}")
        print(f"    Recall    (not delayed): {rec_0:.4f}\n")
 

 
def main():
    # Load data
    print("Loading data...")
    train_df = load_split(f'/Users/cccar/{DATA_DIR}/train_df.csv')
    val_df = load_split(f'/Users/cccar/{DATA_DIR}/val_df.csv')
    test_df = load_split(f'/Users/cccar/{DATA_DIR}/test_df.csv')
 
    print(f"  Train: {len(train_df)} ARR events")
    print(f"  Val:   {len(val_df)} ARR events")
    print(f"  Test:  {len(test_df)} ARR events")
    print(f"  Delay rates — Train: {train_df['delayed'].mean():.1%}, "
          f"Val: {val_df['delayed'].mean():.1%}, Test: {test_df['delayed'].mean():.1%}")
 
    # Build sequences
    train_seqs, train_labs, train_ext, _, scaler = df_to_sequences(train_df, fit_scaler=True)
    val_seqs, val_labs, val_ext, _, _ = df_to_sequences(val_df, scaler=scaler)
    test_seqs, test_labs, test_ext, test_meta, _ = df_to_sequences(test_df, scaler=scaler)
 
    print(f"  Sequences — Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")
 
    # Stage 1: Train HMM
    n_feat = len(HMM_FEATURES)
    hmm = MultivariateGaussianHMM(n_states=3, n_features=n_feat, smooth=0.1)
    print(f"\nStage 1: Training HMM ({n_feat} features)...")
    history = hmm.fit(train_seqs, max_iter=50)
 
    print(f"\nLearned transition matrix:\n{np.array2string(hmm.A, precision=3)}")
    print(f"\nLearned emission means:")
    for i, name in enumerate(STATE_NAMES):
        vals = [f"{HMM_FEATURES[j]}={hmm.means[i,j]:.2f}" for j in range(n_feat)]
        print(f"  {name}: {', '.join(vals)}")
 
    # Stage 2: Build classifier features and train
    print(f"\nStage 2: Training classifier on HMM posteriors + {len(EXTRA_FEATURES)} extra features...")
 
    X_train = build_classifier_features(hmm, train_seqs, train_ext)
    y_train = np.concatenate(train_labs)
    X_val = build_classifier_features(hmm, val_seqs, val_ext)
    y_val = np.concatenate(val_labs)
    X_test = build_classifier_features(hmm, test_seqs, test_ext)
    y_test = np.concatenate(test_labs)
 
    print(f"  Classifier feature shape: {X_train.shape} "
          f"({hmm.n_states} HMM posteriors + {len(EXTRA_FEATURES)} extra features)")
 
    clf = train_classifier(X_train, y_train)
 
    feature_names = STATE_NAMES + EXTRA_FEATURES
    print(f"\n  Classifier coefficients:")
    for name, coef in zip(feature_names, clf.coef_[0]):
        print(f"    {name:>20s}: {coef:+.3f}")
        
    print("\n" + "=" * 60)
    print("VALIDATION SET")
    print("=" * 60)
    evaluate_hmm_native(hmm, val_seqs)
    evaluate_classifier(clf, X_val, y_val)
 
    print("=" * 60)
    print("TEST SET")
    print("=" * 60)
    evaluate_hmm_native(hmm, test_seqs)
    evaluate_classifier(clf, X_test, y_test)
    
    if test_seqs:
        hmm_result = hmm.predict_state(test_seqs[0][:20])
        print(f"Example HMM prediction ({test_meta[0]['route_id']}, "
              f"{test_meta[0]['service_date']}):")
        print(f"  State: {hmm_result['current_state']}")
        print(f"  Delay probability: {hmm_result['delay_probability']}")
 
    # Save
    hmm.save(f'{DATA_DIR}/models/hmm_trained.pkl')
    with open(f'{DATA_DIR}/models/hmm_classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print(f"Classifier saved to {DATA_DIR}/models/hmm_classifier.pkl")
    with open(f'{DATA_DIR}/models/hmm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {DATA_DIR}/models/hmm_scaler.pkl")
 
    return hmm, clf, history
 
 
class ReadyHMM:
    """
    Loads pre-trained HMM + classifier and predicts delay probability
    for a future trip.
    """
 
    def __init__(
        self,
        hmm_path='src/models/model_storage/hmm_trained.pkl',
        clf_path='src/models/model_storage/hmm_classifier.pkl',
        scaler_path='src/models/model_storage/hmm_scaler.pkl',
        data_path='src/models/model_storage/train_df.csv',
    ):
        # Load models
        self.hmm = MultivariateGaussianHMM(n_states=3, n_features=len(HMM_FEATURES))
        self.hmm.load(hmm_path)
        with open(clf_path, 'rb') as f:
            self.clf = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
 
        # Load training data for historical delay lookup and trip routing
        self.df = pd.read_csv(data_path, dtype={'trip_id': str})
        self.df['stop_sequence'] = pd.to_numeric(self.df['stop_sequence'], errors='coerce')
        self.df['arrival_time_sec_x'] = pd.to_numeric(
            self.df.get('arrival_time_sec_x', self.df.get('arrival_time_sec', pd.Series())),
            errors='coerce'
        )
        self.df = self.df.dropna(subset=['trip_id', 'stop_name', 'stop_sequence'])
 
        # Build historical delay lookup
        arr_df = self.df[self.df['event_type'] == 'ARR'] if 'event_type' in self.df.columns else self.df
        self.delay_lookup = arr_df.groupby(
            ['route_id', 'stop_name', 'hour', 'day_of_week_num']
        )['delay_sec'].mean().to_dict()
 
    def time_to_seconds(self, dt):
        return dt.hour * 3600 + dt.minute * 60 + dt.second
 
    def get_valid_trips(self, route_df, origin_stop, destination_stop):
        trips = []
        for trip_id, trip in route_df.groupby('trip_id'):
            trip = trip.sort_values('stop_sequence')
            stops = trip['stop_name'].values
            if origin_stop not in stops or destination_stop not in stops:
                continue
            origin_seq = trip[trip['stop_name'] == origin_stop]['stop_sequence'].values[0]
            dest_seq = trip[trip['stop_name'] == destination_stop]['stop_sequence'].values[0]
            if origin_seq < dest_seq:
                trips.append(trip_id)
        return trips
 
    def get_next_trip(self, route_df, trips, origin_stop, current_time_sec):
        candidates = []
        for trip_id in trips:
            trip = route_df[route_df['trip_id'] == trip_id]
            origin_row = trip[trip['stop_name'] == origin_stop]
            if origin_row.empty:
                continue
            # Try both possible column names for arrival time
            time_col = 'arrival_time_sec_x' if 'arrival_time_sec_x' in origin_row.columns else 'arrival_time_sec'
            origin_time = float(origin_row[time_col].values[0])
            if origin_time >= current_time_sec:
                candidates.append((trip_id, origin_time))
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1])[0]
 
    def build_trip_sequence(self, route_df, trip_id, origin_stop, destination_stop):
        trip = route_df[route_df['trip_id'] == trip_id].copy().sort_values('stop_sequence')
        origin_seq = trip[trip['stop_name'] == origin_stop]['stop_sequence'].values[0]
        dest_seq = trip[trip['stop_name'] == destination_stop]['stop_sequence'].values[0]
        return trip[(trip['stop_sequence'] >= origin_seq) &
                    (trip['stop_sequence'] <= dest_seq)].copy()
 
    def predict(self, obs):
        route, arr_stop, dest_stop, cur_datetime = obs
        current_time_sec = self.time_to_seconds(cur_datetime)
        hour = cur_datetime.hour
        dow = cur_datetime.weekday()
        month = cur_datetime.month
 
        # Find the next valid trip
        route_df = self.df[self.df['route_id'] == route]
        valid_trips = self.get_valid_trips(route_df, arr_stop, dest_stop)
        next_trip = self.get_next_trip(route_df, valid_trips, arr_stop, current_time_sec)
 
        if next_trip is None:
            raise ValueError("No future trip found for this route and stop selection.")
 
        trip_seq_df = self.build_trip_sequence(route_df, next_trip, arr_stop, dest_stop)
 
        # Estimate delay features from historical averages
        df = trip_seq_df.copy()
        estimated_delays = []
        for _, row in df.iterrows():
            key = (route, row.get('stop_name', ''), hour, dow)
            estimated_delays.append(self.delay_lookup.get(key, 60.0))
 
        df['ARR'] = estimated_delays
        df['DEP'] = estimated_delays
        df['PRA'] = pd.Series(estimated_delays).shift(1).fillna(0).values
        df['PRD'] = pd.Series(estimated_delays).shift(1).fillna(0).values
        df['PRA_2'] = pd.Series(estimated_delays).shift(2).fillna(0).values
        df['PRD_2'] = pd.Series(estimated_delays).shift(2).fillna(0).values
        df['hour'] = hour
        df['day_of_week_num'] = dow
        df['month_num'] = month
        df['prev_delayed'] = (df['PRA'] > 60).astype(int)
        df['prev_delayed_2'] = (df['PRA_2'] > 60).astype(int)
 
        # Run HMM + classifier
        obs_scaled = self.scaler.transform(df[HMM_FEATURES].fillna(0).values)
        posteriors = self.hmm.get_posteriors(obs_scaled)
        extras = df[EXTRA_FEATURES].fillna(0).values
        clf_features = np.hstack([posteriors, extras])
        probs = self.clf.predict_proba(clf_features)[:, 1]
 
        last_state = STATE_NAMES[np.argmax(posteriors[-1])]
        arrival_prob = float(probs[0])
        destination_prob = float(probs[-1])
 
        if destination_prob > 0.7 and arrival_prob > 0.7:
            likelihood = "Very Likely"
        elif destination_prob > 0.5 and arrival_prob > 0.5:
            likelihood = "Likely"
        else:
            likelihood = "Not Likely"
 
        return trip_seq_df, {
            "Arrival at Starting Point Delay": (
                f'The HMM predicts a {int(arrival_prob * 100)}% chance of delay '
                f'at {arr_stop} (system state: {last_state})'
            ),
            "Arrival at Destination Delay": (
                f'The HMM predicts a {int(destination_prob * 100)}% chance of delay '
                f'at {dest_stop}'
            ),
            "Delay is": likelihood,
        }
 
 
if __name__ == '__main__':
    main()
