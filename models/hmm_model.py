import numpy as np
import pandas as pd
import pickle
from hmmlearn.hmm import GaussianHMM


def build_sequences(df, chunk_size=50, min_length=10):
    """
    Group events by (route, date), split into fixed-size chunks,
    and return as continuous delay_min arrays
    """
    sequences, metadata = [], []

    for (route, date), group in df.groupby(['route_id', 'service_date']):
        group = group.sort_values('event_time_sec')
        delays = group['delay_min'].values

        for start in range(0, len(delays), chunk_size):
            chunk = delays[start : start + chunk_size]
            if len(chunk) < min_length:
                continue

            sequences.append(chunk.reshape(-1, 1))
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


# Model building

class HMM(GaussianHMM):
    """
    GaussianHMM with interpolation smoothing on the transition matrix.
    """

    def __init__(self, *args, smooth=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.smooth = smooth

    def _do_mstep(self, stats):
        """Override M-step to apply transition smoothing after update."""
        super()._do_mstep(stats)
        uniform = np.full_like(self.transmat_, 1.0 / self.n_components)
        self.transmat_ = (1 - self.smooth) * self.transmat_ + self.smooth * uniform
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)


def build_model(n_states=3, smooth=0.1, random_seed=42):
    """
    Create a SmoothedGaussianHMM with informed initial parameters.

    Hidden states:
        0 = Normal (low delay)
        1 = Minor Delay (moderate delay)
        2 = Major Delay (significant delay)
    """
    model = HMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=100,
        tol=1e-4,
        random_state=random_seed,
        smooth=smooth,
        init_params='',
        verbose=True,
    )

    # Initial state distribution
    model.startprob_ = np.array([0.7, 0.2, 0.1])

    # Transition matrix
    model.transmat_ = np.array([
        [0.85, 0.10, 0.05],
        [0.15, 0.70, 0.15],
        [0.05, 0.25, 0.70],
    ])

    # Emission means
    model.means_ = np.array([[0.0], [5.0], [15.0]])

    # Emission covariances
    model.covars_ = np.array([[[4.0]], [[9.0]], [[25.0]]])

    return model


# Prediction wrapper

STATE_NAMES = ['Normal', 'Minor Delay', 'Major Delay']

def predict_state(model, recent_delays):
    """
    Given recent delay_min values, predict current system state.
    """
    obs = np.array(recent_delays).reshape(-1, 1)

    # Decode most likely state sequence
    state_path = model.predict(obs)

    # Get state probabilities at the last time step
    posteriors = model.predict_proba(obs)
    state_probs = posteriors[-1]

    # Forecast next state using transition matrix
    next_probs = state_probs @ model.transmat_

    return {
        'current_state': STATE_NAMES[state_path[-1]],
        'state_probs': dict(zip(STATE_NAMES, state_probs.round(4))),
        'delay_probability': round(float(1 - next_probs[0]), 4),
        'next_state_forecast': dict(zip(STATE_NAMES, next_probs.round(4))),
    }



# Evaluation

def evaluate(model, sequences):
    """Print log-likelihood and decoded state distribution on test sequences."""
    total_ll = 0.0
    total_obs = 0
    state_counts = np.zeros(model.n_components)

    for obs in sequences:
        total_ll += model.score(obs)
        total_obs += len(obs)

        path = model.predict(obs)
        for s in range(model.n_components):
            state_counts[s] += (path == s).sum()

    print(f"Test log-likelihood: {total_ll:.2f} ({total_ll / total_obs:.4f} per obs)")
    print("State distribution:")
    for name, count in zip(STATE_NAMES, state_counts):
        print(f"  {name}: {count:.0f} ({count / state_counts.sum() * 100:.1f}%)")


# Main pipeline

def main():
    # Load preprocessed data
    df = pd.read_csv('/Users/cccar/Downloads/merged_events.csv', dtype={'trip_id': str})
    df = df[df['event_type'] == 'ARR'].copy()

    # Clip outliers
    clip_low, clip_high = -10, 60
    n_before = len(df)
    df = df[(df['delay_min'] >= clip_low) & (df['delay_min'] <= clip_high)].copy()

    # Build and split sequences
    seqs, meta = build_sequences(df, chunk_size=50)
    train_s, test_s, train_m, test_m = chronological_split(seqs, meta)

    # Concatenate training sequences
    X_train = np.concatenate(train_s)
    lengths_train = [len(s) for s in train_s]

    # Train
    model = build_model(n_states=3)
    model.fit(X_train, lengths_train)

    # Print learned parameters
    print(f"\nLearned transition matrix:\n{np.array2string(model.transmat_, precision=3)}")
    print(f"\nLearned emission means (delay_min per state):")
    for name, mean, cov in zip(STATE_NAMES, model.means_, model.covars_):
        std = np.sqrt(cov[0, 0])
        print(f"  {name}: mean={mean[0]:.2f} min, std={std:.2f} min")

    # Evaluate
    evaluate(model, test_s)

    # Example prediction
    if test_s:
        result = predict_state(model, test_s[0][:20])
        print(f"\nExample prediction ({test_m[0]['route_id']}, {test_m[0]['service_date']}):")
        print(f"  State: {result['current_state']}")
        print(f"  Delay probability: {result['delay_probability']}")

    # Save model
    with open('/Users/cccar/Downloads/hmm_trained.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved to /Users/cccar/Downloads/hmm_trained.pkl")

    return model


if __name__ == '__main__':
    main()
