"""Random-trajectory collection + KMeans clustering for the z-flow cluster view.

Collects states visited under a uniform-random policy, clusters them by their
RSSM stochastic latent (`stoch`), and projects the same latents to 2D via PCA
for scatter-plot layout. Used by serve.py to power the /clusters/* endpoints.
"""

import dataclasses

import numpy as np
import torch
from serve_utils import make_trans
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


@torch.no_grad()
def collect_random_states(agent, env, num_episodes: int, max_steps_per_episode: int, device) -> dict:
    """Roll out `num_episodes` uniform-random-policy episodes, collecting every
    visited (image, stoch) pair.

    `agent.act(trans, state, random=True)` still runs the encoder + RSSM
    posterior on `obs` before sampling the random action (only the action
    itself is randomized), so pairing `obs["image"]` with the `stoch` returned
    by that same call is the correct, non-off-by-one pairing.
    """
    images = []
    stochs = []
    trajectory_ids = []
    timesteps = []
    for episode in range(num_episodes):
        obs = env.reset()
        state = agent.get_initial_state(1)
        action = state["prev_action"].clone()
        done = False
        step = 0
        while not done and step < max_steps_per_episode:
            trans = make_trans(obs, action, device)
            action, state, _ = agent.act(trans, state, random=True)
            images.append(np.asarray(obs["image"]))
            stochs.append(state["stoch"][0].detach().cpu().numpy())
            trajectory_ids.append(episode)
            timesteps.append(step)
            obs, _, done, _ = env.step(action[0].cpu().numpy())
            step += 1

    return {
        "images": np.stack(images),
        "stochs": np.stack(stochs).astype(np.float32),
        "trajectory_ids": np.array(trajectory_ids, dtype=int),
        "timesteps": np.array(timesteps, dtype=int),
    }


def run_clustering(stochs: np.ndarray, k: int, seed: int) -> dict:
    """KMeans on the flattened one-hot stoch; PCA(2) on the same vectors for layout."""
    n = stochs.shape[0]
    flat = stochs.reshape(n, -1)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(flat)
    coords = PCA(n_components=2, random_state=seed).fit_transform(flat)
    return {
        "cluster_ids": kmeans.labels_.astype(int),
        "coords": coords.astype(np.float32),
    }


def compute_row_stats(stochs: np.ndarray) -> dict:
    """Per-row (stoch group) category-usage stats across the collected states.

    A low entropy_ratio means a row is concentrated on a handful of
    categories regardless of which state was visited — a candidate signal
    that the row encodes something structural/static rather than freely
    varying content.
    """
    n, s, k = stochs.shape
    counts = stochs.sum(axis=0)  # (S, K)
    probs = counts / max(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probs > 0, probs * np.log2(probs), 0.0)
    entropy = -terms.sum(axis=1)  # (S,)
    max_entropy = np.log2(k) if k > 1 else 1.0
    dominant_category = counts.argmax(axis=1)
    dominant_frac = counts.max(axis=1) / max(n, 1)
    return {
        "rows": int(s),
        "categories": int(k),
        "entropy": entropy.tolist(),
        "entropy_ratio": (entropy / max_entropy).tolist(),
        "dominant_category": dominant_category.astype(int).tolist(),
        "dominant_frac": dominant_frac.tolist(),
    }


@dataclasses.dataclass
class ClusterStore:
    coords: np.ndarray  # (N, 2) float32
    cluster_ids: np.ndarray  # (N,) int
    images: np.ndarray  # (N, H, W, C) uint8
    stochs: np.ndarray  # (N, S, K) float32
    trajectory_ids: np.ndarray  # (N,) int
    timesteps: np.ndarray  # (N,) int
    k: int

    def __post_init__(self):
        self.n_states = len(self.cluster_ids)
        self.n_trajectories = int(self.trajectory_ids.max()) + 1 if self.n_states else 0
        self.cluster_members: dict[int, list[int]] = {c: [] for c in range(self.k)}
        for idx, cluster_id in enumerate(self.cluster_ids):
            self.cluster_members[int(cluster_id)].append(idx)

        # Collection order is sequential per episode, so idx/idx+1 are
        # consecutive timesteps of the same trajectory iff their
        # trajectory_ids match — precompute the adjacency once here so the
        # API can serve prev/next lookups in O(1).
        n = self.n_states
        self.prev_ids: list[int | None] = [None] * n
        self.next_ids: list[int | None] = [None] * n
        for idx in range(n - 1):
            if self.trajectory_ids[idx] == self.trajectory_ids[idx + 1]:
                self.next_ids[idx] = idx + 1
                self.prev_ids[idx + 1] = idx

    @staticmethod
    def state_id(idx: int | None) -> str | None:
        return None if idx is None else f"cz_{idx}"

    def parse_state_id(self, state_id: str) -> int | None:
        if not state_id.startswith("cz_"):
            return None
        try:
            idx = int(state_id[len("cz_") :])
        except ValueError:
            return None
        if idx < 0 or idx >= self.n_states:
            return None
        return idx
