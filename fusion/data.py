"""fusion/data.py -- pooled dataset F (docs/challenge_text.md §2), struct-of-arrays rows.

C/C++ -> Python -> R: `struct Row{int64 t; double z,y;} rows[2mn]` filled by nested loops ->
numpy int64/float64 arrays built by concatenation -> a `data.frame(T=,Z=,Y=)` built by rbind.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fusion import dgp
from fusion.config import Config


def build_dataset(rng: np.random.Generator, cfg: Config):
    """Step 9 (empirical risk's data): F = pooled rows (T,Z,Y), t=1..m, n rows per (t,z). 2mn rows.

    z=1 <-> P_{t,Y} (challenge_text.md §2's F); z=0 <-> S_t.
    """
    T, Z, Y = [], [], []
    for t in range(1, cfg.m + 1):
        ys = dgp.draw_survey(cfg, rng, t, cfg.n)
        yp = dgp.draw_pop(cfg, rng, t, cfg.n)
        T += [t] * (2 * cfg.n)
        Z += [0.0] * cfg.n + [1.0] * cfg.n
        Y += list(ys) + list(yp)
    return np.array(T, dtype=np.int64), np.array(Z, dtype=np.float64), np.array(Y, dtype=np.float64)


def stratified_split(rng: np.random.Generator, T: np.ndarray, Z: np.ndarray, val_frac: float):
    """Hold out val_frac inside every (t,z) cell (CLAUDE.md §2.4). Returns (train_mask, val_mask).

    C/C++ -> Python -> R: per-cell `for` loop drawing a random subset -> `rng.choice` per cell ->
    `sample()` inside a `split(df, list(t,z))`.
    """
    val = np.zeros(len(T), dtype=bool)
    for t in np.unique(T):
        for z in (0.0, 1.0):
            idx = np.where((T == t) & (Z == z))[0]
            n_val = int(round(val_frac * len(idx)))
            if n_val > 0:
                val[rng.choice(idx, n_val, replace=False)] = True
    return ~val, val


@dataclass
class Standardizer:
    """y standardised with TRAINING-set mean/std only; the same constants are reused for t>m
    (CLAUDE.md §2.4) -- theta is evaluated pointwise on standardised y, no Jacobian anywhere."""

    mean: float
    std: float

    @classmethod
    def fit(cls, y_train: np.ndarray) -> "Standardizer":
        return cls(float(np.mean(y_train)), float(np.std(y_train)))

    def transform(self, y) -> np.ndarray:
        return (np.asarray(y, dtype=np.float64) - self.mean) / self.std

    def to_json(self) -> dict:
        return {"mean": self.mean, "std": self.std}


def iterate_batches(rng: np.random.Generator, idx: np.ndarray, batch_size: int):
    """One shuffled epoch of index batches.

    C/C++ -> Python -> R: Fisher-Yates shuffle + fixed-stride `for` -> `rng.shuffle` + slicing ->
    `sample(idx)` + `split(..., ceiling(seq_along(idx)/batch_size))`.
    """
    order = idx.copy()
    rng.shuffle(order)
    for s in range(0, len(order), batch_size):
        yield order[s: s + batch_size]
