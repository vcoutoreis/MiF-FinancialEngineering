"""
Monte Carlo valuation of an arithmetic-average Asian call option.

Key design points:
- Paths are built CUMULATIVELY: each sample size N in parts (a)-(j) reuses
  the deviates from the smaller previous N (as required by the problem).
- Under the martingale measure, the drift is r (not mu).
- Antithetic variates: for each "primary" path using deviates Z, a second
  path is built using -Z, then the two payoffs are averaged per pair.
"""

# %%

import numpy as np
import pandas as pd

# ----- Parameters -----
S0    = 100.0
mu    = 0.15      # physical drift (NOT used in simulation — martingale valuation uses r)
sigma = 0.20
r     = 0.08
T     = 1.0
K     = 100.0
n_steps = 250
dt    = T / n_steps

# ----- Master deviate matrix -----
# We generate enough paths to cover the largest N needed (N=1000),
# plus the antithetic pairs which use the SAME deviates with opposite sign.
N_MAX = 1000
rng = np.random.default_rng(seed=42)
Z = rng.standard_normal(size=(N_MAX, n_steps))   # one row per path, 250 deviates each


def simulate_paths(Z_block):
    """
    Given a (N, 250) matrix of standard-normal deviates,
    produce (N, 250) matrix of stock prices S_1, S_2, ..., S_250.
    Uses exact GBM solution under the risk-neutral measure.
    """
    # log-increments (risk-neutral drift)
    log_incr = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_block
    # cumulative sum gives log(S_t / S_0)
    log_S = np.cumsum(log_incr, axis=1)
    S = S0 * np.exp(log_S)
    return S


def discounted_payoffs(Z_block):
    """Return array of discounted Asian-call payoffs for each path."""
    S = simulate_paths(Z_block)
    avg = S.mean(axis=1)                     # arithmetic average over 250 days
    payoff = np.maximum(avg - K, 0.0)
    return np.exp(-r * T) * payoff


def plain_mc(N):
    """Plain Monte Carlo on the first N paths."""
    V = discounted_payoffs(Z[:N])
    V_hat = V.mean()
    se = np.sqrt(np.sum((V - V_hat)**2) / (N * (N - 1)))
    return V_hat, se, (V_hat - 2*se, V_hat + 2*se)


def antithetic_mc(N):
    """
    Antithetic-variate Monte Carlo on the first N paths.
    Uses the SAME N rows of Z as the primary paths, then -Z for the antithetic pair.
    """
    V1 = discounted_payoffs( Z[:N])
    V2 = discounted_payoffs(-Z[:N])
    # per-pair average, then average over pairs
    pair_means = 0.5 * (V1 + V2)
    V_hat = pair_means.mean()
    se_star = np.sqrt(np.sum((pair_means - V_hat)**2) / (N * (N - 1)))
    return V_hat, se_star, (V_hat - 2*se_star, V_hat + 2*se_star)


# ----- Run all parts -----
rows = []

# (a)-(e): small samples
for part, N in [('(a)', 5), ('(b)', 10), ('(c)', 20)]:
    V_hat, se, ci = plain_mc(N)
    rows.append([part, 'Plain',       N, V_hat, se, ci[0], ci[1]])

for part, N in [('(d)', 10), ('(e)', 20)]:
    V_hat, se, ci = antithetic_mc(N)
    rows.append([part, 'Antithetic',  N, V_hat, se, ci[0], ci[1]])

# (f)-(j): larger samples
for part, N in [('(f)', 100), ('(g)', 500), ('(h)', 1000)]:
    V_hat, se, ci = plain_mc(N)
    rows.append([part, 'Plain',       N, V_hat, se, ci[0], ci[1]])

for part, N in [('(i)', 500), ('(j)', 1000)]:
    V_hat, se, ci = antithetic_mc(N)
    rows.append([part, 'Antithetic',  N, V_hat, se, ci[0], ci[1]])

df = pd.DataFrame(rows, columns=['Part', 'Method', 'N', 'V_hat', 's.e.', 'CI_low', 'CI_high'])

# Pretty print
pd.set_option('display.float_format', lambda x: f'{x:8.4f}')
print(df.to_string(index=False))

# Save for later reference
df.to_csv(r'C:\Users\vitor\Downloads\results.csv', index=False)