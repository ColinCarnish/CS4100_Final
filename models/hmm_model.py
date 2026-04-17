import numpy as np
import pandas as pd

# AI DISCLOSURE: Claude Opus 4.6 was used to help implement and debug this code

# Discretizer — converts continuous delay values to discrete HMM symbols

class DelayDiscretizer:
    """Quantile-based binning of delay_min into discrete observation symbols."""

    def __init__(self, n_bins=5):
        self.n_bins = n_bins
        self.bin_edges = None

    def fit(self, values):
        """Learn bin edges from data using evenly spaced quantiles."""
        quantiles = np.linspace(0, 100, self.n_bins + 1)
        self.bin_edges = np.unique(np.percentile(values, quantiles))
        self.n_bins = len(self.bin_edges) - 1
        return self

    def transform(self, values):
        """Map continuous values to symbol indices 0..n_bins-1."""
        return np.clip(np.digitize(values, self.bin_edges[1:-1]), 0, self.n_bins - 1)

    def fit_transform(self, values):
        return self.fit(values).transform(values)


# Sequence Builder

def build_sequences(df, discretizer, chunk_size=75, min_length=10):
    """
    Group events by (route, date, hour) and split into fixed-size chunks.

    Args:
        df:           merged DataFrame with delay_min, event_time_sec columns
        discretizer:  fitted DelayDiscretizer
        chunk_size:   max observations per sequence (default 75)
        min_length:   discard chunks shorter than this

    Returns:
        sequences: list of np arrays (symbol indices)
        metadata:  list of dicts with route_id, service_date
    """
    sequences, metadata = [], []

    for (route, date), group in df.groupby(['route_id', 'service_date']):
        group = group.sort_values('event_time_sec')
        symbols = discretizer.transform(group['delay_min'].values)

        # Split day into fixed-size chunks
        for start in range(0, len(symbols), chunk_size):
            chunk = symbols[start : start + chunk_size]
            if len(chunk) < min_length:
                continue
            sequences.append(chunk)
            metadata.append({'route_id': route, 'service_date': date})

    return sequences, metadata


def chronological_split(sequences, metadata, test_fraction=0.2):
    """Split sequences by date (not random) to avoid time-series leakage."""
    dates = sorted(set(m['service_date'] for m in metadata))
    cutoff = dates[int(len(dates) * (1 - test_fraction))]

    train_s, test_s, train_m, test_m = [], [], [], []
    for seq, meta in zip(sequences, metadata):
        if meta['service_date'] <= cutoff:
            train_s.append(seq)
            train_m.append(meta)
        else:
            test_s.append(seq)
            test_m.append(meta)

    return train_s, test_s, train_m, test_m


# Hidden Markov Model

class HMM:
    """
    Discrete HMM with 3 hidden states representing transit conditions:
        0 = Normal, 1 = Minor Delay, 2 = Major Delay

    Core algorithms implemented from scratch:
        - Forward-backward
        - Baum-Welch
        - Viterbi
    """

    def __init__(self, n_states=3, n_obs=5, seed=42):
        self.n_states = n_states
        self.n_obs = n_obs
        self.state_names = ['Normal', 'Minor Delay', 'Major Delay'][:n_states]

        # Initialize with informed priors
        rng = np.random.default_rng(seed)

        # pi: initial distribution
        self.pi = np.array([0.7, 0.2, 0.1])[:n_states]
        self.pi /= self.pi.sum()

        # A: transition matrix
        self.A = np.array([
            [0.85, 0.10, 0.05],
            [0.15, 0.70, 0.15],
            [0.05, 0.25, 0.70],
        ])[:n_states, :n_states]
        self.A += rng.uniform(0, 0.02, self.A.shape)
        self.A /= self.A.sum(axis=1, keepdims=True)

        # B: emission matrix
        self.B = np.zeros((n_states, n_obs))
        for i in range(n_states):
            center = (i / max(n_states - 1, 1)) * (n_obs - 1)
            for j in range(n_obs):
                self.B[i, j] = np.exp(-0.5 * ((j - center) / (n_obs / 3)) ** 2)
        self.B /= self.B.sum(axis=1, keepdims=True)


    # Forward algorithm

    def _forward(self, obs):
        """Returns alpha (T x N) and scaling factors (T,)."""
        T, N = len(obs), self.n_states
        alpha = np.zeros((T, N))
        c = np.zeros(T)  # scaling factors

        alpha[0] = self.pi * self.B[:, obs[0]]
        c[0] = alpha[0].sum()
        if c[0] > 0:
            alpha[0] /= c[0]

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = alpha[t - 1] @ self.A[:, j] * self.B[j, obs[t]]
            c[t] = alpha[t].sum()
            if c[t] > 0:
                alpha[t] /= c[t]

        return alpha, c


    # Backward algorithm

    def _backward(self, obs, c):
        """Returns beta (T x N), using scaling factors from forward pass."""
        T, N = len(obs), self.n_states
        beta = np.zeros((T, N))

        beta[T - 1] = 1.0 / (c[T - 1] + 1e-300)

        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t, i] = (self.A[i, :] * self.B[:, obs[t + 1]]) @ beta[t + 1]
            if c[t] > 0:
                beta[t] /= c[t]

        return beta

    # Viterbi

    def viterbi(self, obs):
        """Returns most likely state path (T,) and its log-probability."""
        T, N = len(obs), self.n_states
        log_A = np.log(self.A + 1e-300)
        log_B = np.log(self.B + 1e-300)

        V = np.zeros((T, N))
        bp = np.zeros((T, N), dtype=int)

        V[0] = np.log(self.pi + 1e-300) + log_B[:, obs[0]]

        for t in range(1, T):
            for j in range(N):
                scores = V[t - 1] + log_A[:, j]
                bp[t, j] = np.argmax(scores)
                V[t, j] = scores[bp[t, j]] + log_B[j, obs[t]]

        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]

        return path, V[-1, path[-1]]

    # Baum-Welch training

    def fit(self, sequences, max_iter=100, tol=1e-4, smooth=0.01, verbose=True):
        """
        Learn pi, A, B from data via Expectation-Maximization.
        smooth: additive smoothing for transition matrix
        Returns list of log-likelihoods per iteration
        """
        N, M = self.n_states, self.n_obs
        history = []

        for it in range(max_iter):
            # Accumulators
            pi_acc = np.zeros(N)
            A_num, A_den = np.zeros((N, N)), np.zeros(N)
            B_num, B_den = np.zeros((N, M)), np.zeros(N)
            total_ll = 0.0

            # E-step: forward-backward on each sequence
            for obs in sequences:
                if len(obs) < 2:
                    continue

                alpha, c = self._forward(obs)
                beta = self._backward(obs, c)
                total_ll += np.sum(np.log(c + 1e-300))

                # Gamma: P(state_t = i | all observations)
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

                # Xi: P(state_t=i, state_{t+1}=j | all observations)
                for t in range(len(obs) - 1):
                    xi = np.outer(alpha[t], self.B[:, obs[t + 1]] * beta[t + 1]) * self.A
                    xi /= xi.sum() + 1e-300
                    A_num += xi
                    A_den += gamma[t]

                # Accumulate statistics
                pi_acc += gamma[0]
                for t in range(len(obs)):
                    B_num[:, obs[t]] += gamma[t]
                    B_den += gamma[t]

            self.pi = pi_acc / (pi_acc.sum() + 1e-300)
            A_learned = A_num / (A_den[:, None] + 1e-300)
            A_prior = np.full((N, N), 1.0 / N)
            self.A = (1 - smooth) * A_learned + smooth * A_prior
            self.B = B_num / (B_den[:, None] + 1e-300)

            history.append(total_ll)
            if verbose and it % 10 == 0:
                print(f"  Iter {it:3d}: LL = {total_ll:.2f}")
            if it > 0 and abs(history[-1] - history[-2]) < tol:
                if verbose:
                    print(f"  Converged at iteration {it}")
                break

        return history


    # Prediction

    def predict(self, obs):
        """
        Given recent observations, return current state and delay forecast.

        Returns dict with: current_state, state_probs, delay_probability,
        next_state_forecast.
        """
        path, _ = self.viterbi(obs)
        alpha, _ = self._forward(obs)
        state_probs = alpha[-1] / (alpha[-1].sum() + 1e-300)
        next_probs = state_probs @ self.A

        return {
            'current_state': self.state_names[path[-1]],
            'state_probs': dict(zip(self.state_names, state_probs.round(4))),
            'delay_probability': round(float(1 - next_probs[0]), 4),
            'next_state_forecast': dict(zip(self.state_names, next_probs.round(4))),
        }

    # Save Results

    def save(self, path):
        np.savez(path, pi=self.pi, A=self.A, B=self.B,
                 n_states=self.n_states, n_obs=self.n_obs)

    def load(self, path):
        d = np.load(path)
        self.pi, self.A, self.B = d['pi'], d['A'], d['B']
        self.n_states, self.n_obs = int(d['n_states']), int(d['n_obs'])



# Evaluation helpers

def evaluate(model, sequences):
    """Print log-likelihood and decoded state distribution on test sequences."""
    total_ll, total_obs = 0.0, 0
    state_counts = np.zeros(model.n_states)

    for obs in sequences:
        if len(obs) < 2:
            continue
        _, c = model._forward(obs)
        total_ll += np.sum(np.log(c + 1e-300))
        total_obs += len(obs)

        path, _ = model.viterbi(obs)
        for s in range(model.n_states):
            state_counts[s] += (path == s).sum()

    print(f"Test log-likelihood: {total_ll:.2f} ({total_ll / total_obs:.4f} per obs)")
    print("State distribution:")
    for name, count in zip(model.state_names, state_counts):
        print(f"  {name}: {count:.0f} ({count / state_counts.sum() * 100:.1f}%)")



# Main pipeline

def main():
    # Load preprocessed data
    df = pd.read_csv('/Users/cccar/Downloads/merged_events.csv')
    df = df[df['event_type'] == 'ARR'].copy()
    print(f"Loaded {len(df)} events | {df['service_date'].nunique()} days | "
          f"Routes: {sorted(df['route_id'].unique())}")

    # Clip outliers: cap delay_min to [-10, 60] minutes
    clip_low, clip_high = -10, 60
    n_before = len(df)
    df = df[(df['delay_min'] >= clip_low) & (df['delay_min'] <= clip_high)].copy()
    n_dropped = n_before - len(df)
    print(f"\nClipped to [{clip_low}, {clip_high}] min — dropped {n_dropped} "
          f"({n_dropped / n_before * 100:.1f}%) outlier events")
    print(f"  min={df['delay_min'].min():.1f}  25%={df['delay_min'].quantile(.25):.1f}  "
          f"median={df['delay_min'].median():.1f}  75%={df['delay_min'].quantile(.75):.1f}  "
          f"max={df['delay_min'].max():.1f}")

    # Discretize delays
    disc = DelayDiscretizer(n_bins=5)
    disc.fit(df['delay_min'].values)
    print(f"Bin edges: {[round(e, 2) for e in disc.bin_edges]}")

    # Build and split sequences
    seqs, meta = build_sequences(df, disc, chunk_size=50)
    train_s, test_s, train_m, test_m = chronological_split(seqs, meta)
    print(f"Train: {len(train_s)} seqs | Test: {len(test_s)} seqs")

    # Train
    model = HMM(n_states=3, n_obs=disc.n_bins)
    print("\nTraining...")
    history = model.fit(train_s, max_iter=50, smooth=0.1)

    # Print learned parameters
    print(f"\nLearned transition matrix:\n{np.array2string(model.A, precision=3)}")
    print(f"\nLearned emission matrix:\n{np.array2string(model.B, precision=3)}")

    # Evaluate
    print()
    evaluate(model, test_s)

    # Example prediction
    if test_s:
        result = model.predict(test_s[0][:20])
        print(f"\nExample prediction ({test_m[0]['route_id']}, {test_m[0]['service_date']}):")
        print(f"  State: {result['current_state']}")
        print(f"  Delay probability: {result['delay_probability']}")

    model.save('/Users/cccar/Downloads/hmm_trained.npz')
    return model, disc, history


if __name__ == '__main__':
    main()
