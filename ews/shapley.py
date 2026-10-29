"""Dependency-free permutation Shapley values (interventional, background-averaged).

For each sampled feature ordering, features are switched from background values
to the explained observation's values one at a time; the score change is
credited to the switched feature. Every permutation is exactly additive, so
contributions always sum to f(x) - mean f(background).
"""
from __future__ import annotations
import numpy as np


def shapley(predict, x, background, n_perm=48, seed=0):
    x = np.asarray(x, dtype=float); B = np.asarray(background, dtype=float)
    d = x.shape[0]; rng = np.random.default_rng(seed)
    phi = np.zeros(d)
    perms = []
    for _ in range(n_perm // 2):
        p = rng.permutation(d); perms += [p, p[::-1]]  # antithetic pairs reduce variance
    n = B.shape[0]
    for p in perms:
        states = np.repeat(B[None, :, :], d + 1, axis=0)  # state k: first k features of p switched
        for k, j in enumerate(p):
            states[k + 1:, :, j] = x[j]
        vals = predict(states.reshape(-1, d)).reshape(d + 1, n).mean(axis=1)
        phi[p] += np.diff(vals)
    phi /= len(perms)
    base = float(predict(B).mean())
    return phi, base


def explain_rows(predict, X, background, n_perm=48, seed=0):
    out = []
    for i in range(len(X)):
        phi, base = shapley(predict, X[i], background, n_perm, seed + i)
        out.append((phi, base))
    return out
