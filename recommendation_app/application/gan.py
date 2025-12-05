import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "key",
    "mode",
    "duration_ms_x",
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _relu(x):
    return np.maximum(0.0, x)


def _tanh(x):
    return np.tanh(x)


class LightweightGANRecommender:
    def __init__(self, feature_dim: int, latent_dim: int = 16, hidden: int = 32, lr: float = 1e-3):
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.lr = lr

        # Generator parameters
        self.Wg1 = np.random.randn(latent_dim, hidden) * 0.1
        self.bg1 = np.zeros((1, hidden))
        self.Wg2 = np.random.randn(hidden, feature_dim) * 0.1
        self.bg2 = np.zeros((1, feature_dim))

        # Discriminator parameters
        self.Wd1 = np.random.randn(feature_dim, hidden) * 0.1
        self.bd1 = np.zeros((1, hidden))
        self.Wd2 = np.random.randn(hidden, 1) * 0.1
        self.bd2 = np.zeros((1, 1))

    def _generator(self, z):
        h1 = _relu(np.dot(z, self.Wg1) + self.bg1)
        out = _tanh(np.dot(h1, self.Wg2) + self.bg2)
        return out, h1

    def _discriminator(self, x):
        h1 = _relu(np.dot(x, self.Wd1) + self.bd1)
        logits = np.dot(h1, self.Wd2) + self.bd2
        prob = _sigmoid(logits)
        return prob, logits, h1

    def _bce_grad_logits(self, pred, target):
        return (pred - target) / pred.shape[0]

    def train(self, real_data: np.ndarray, epochs: int = 200, batch_size: int = 64):
        data = real_data.copy().astype(np.float32)

        # Normalize roughly to [-1, 1]
        data_min = data.min(axis=0)
        data_max = data.max(axis=0) + 1e-6
        data = 2 * ((data - data_min) / (data_max - data_min)) - 1
        n = data.shape[0]

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            data = data[idx]

            for start in range(0, n, batch_size):
                real_batch = data[start:start + batch_size]
                if real_batch.shape[0] == 0:
                    continue

                # --- Train Discriminator ---
                z = np.random.randn(real_batch.shape[0], self.latent_dim).astype(np.float32)
                fake_batch, _ = self._generator(z)

                d_real, _, h_real = self._discriminator(real_batch)
                d_fake, _, h_fake = self._discriminator(fake_batch)

                y_real = np.ones_like(d_real)
                y_fake = np.zeros_like(d_fake)

                grad_logits_real = self._bce_grad_logits(d_real, y_real)
                grad_logits_fake = self._bce_grad_logits(d_fake, y_fake)

                grad_Wd2 = np.dot(h_real.T, grad_logits_real) + np.dot(h_fake.T, grad_logits_fake)
                grad_bd2 = grad_logits_real.sum(axis=0, keepdims=True) + grad_logits_fake.sum(axis=0, keepdims=True)

                grad_h_real = np.dot(grad_logits_real, self.Wd2.T)
                grad_h_fake = np.dot(grad_logits_fake, self.Wd2.T)
                grad_h_real[h_real <= 0] = 0
                grad_h_fake[h_fake <= 0] = 0

                grad_Wd1 = np.dot(real_batch.T, grad_h_real) + np.dot(fake_batch.T, grad_h_fake)
                grad_bd1 = grad_h_real.sum(axis=0, keepdims=True) + grad_h_fake.sum(axis=0, keepdims=True)

                self.Wd1 -= self.lr * grad_Wd1
                self.bd1 -= self.lr * grad_bd1
                self.Wd2 -= self.lr * grad_Wd2
                self.bd2 -= self.lr * grad_bd2

                # --- Train Generator ---
                z = np.random.randn(real_batch.shape[0], self.latent_dim).astype(np.float32)
                fake_batch, gh1 = self._generator(z)
                d_fake_for_g, _, h_fake_g = self._discriminator(fake_batch)

                y_g = np.ones_like(d_fake_for_g)
                grad_logits_g = self._bce_grad_logits(d_fake_for_g, y_g)

                grad_h_fake_g = np.dot(grad_logits_g, self.Wd2.T)
                grad_h_fake_g[h_fake_g <= 0] = 0
                grad_x_fake = np.dot(grad_h_fake_g, self.Wd1.T)
                grad_x_fake *= (1 - np.square(fake_batch))  # tanh backprop

                grad_Wg2 = np.dot(gh1.T, grad_x_fake)
                grad_bg2 = grad_x_fake.sum(axis=0, keepdims=True)

                grad_gh1 = np.dot(grad_x_fake, self.Wg2.T)
                grad_gh1[gh1 <= 0] = 0
                grad_Wg1 = np.dot(z.T, grad_gh1)
                grad_bg1 = grad_gh1.sum(axis=0, keepdims=True)

                self.Wg1 -= self.lr * grad_Wg1
                self.bg1 -= self.lr * grad_bg1
                self.Wg2 -= self.lr * grad_Wg2
                self.bg2 -= self.lr * grad_bg2

            self.lr *= 0.999  # light decay

    def sample(self, num: int = 128):
        z = np.random.randn(num, self.latent_dim).astype(np.float32)
        fake_batch, _ = self._generator(z)
        return fake_batch


def rank_with_gan(
    base_vector: pd.Series,
    nonplaylist_features: pd.DataFrame,
    song_df: pd.DataFrame,
    gan: LightweightGANRecommender,
    synth_samples: int = 256,
    top_k: int = 10,
):
    # Align columns
    feature_cols = [c for c in FEATURE_COLUMNS if c in nonplaylist_features.columns]
    if not feature_cols:
        return song_df.head(0)

    features = nonplaylist_features[feature_cols].fillna(0.0).astype(np.float32)
    if features.empty:
        return song_df.head(0)

    # Quick train
    gan.train(features.values, epochs=120, batch_size=64)
    synth = gan.sample(synth_samples)
    synth_mean = synth.mean(axis=0)

    # Combine base and synthetic signal
    base = base_vector.copy()
    base = base[[c for c in feature_cols if c in base.index]].fillna(0.0).values
    if base.sum() == 0:
        base = np.ones_like(base)

    combo = 0.6 * base + 0.4 * synth_mean[: base.shape[0]]
    sims = cosine_similarity(features.values, combo.reshape(1, -1))[:, 0]
    scores = pd.Series(sims, index=nonplaylist_features["id"])

    candidates = song_df[song_df["id"].isin(scores.index)].copy()
    candidates["sim"] = candidates["id"].map(scores)
    candidates = candidates.sort_values("sim", ascending=False)
    return candidates.head(top_k)
