#!/usr/bin/env python3
"""Synthetic experiments for "Learning Lotteries with Minimal Stakeholder
Violations from Membership Queries" (Section 7 and Appendix J).

Usage:  python3 run_experiments.py [output.json]

Requires numpy and scipy (HiGHS solvers); tested with Python 3.13.16,
NumPy 2.5.3 and SciPy 1.18.1. All randomness is seeded, so the script
reproduces the reported numbers exactly on the same library versions. It
prints a summary (Table 1, Table 3 and the numbers in Appendix J) and writes
all raw results to output.json (default: results.json). Running time: about
2 minutes on 2 CPU cores, most of it for the mixed-integer check.

Conventions (as in Section 2 of the paper):
  * a lottery is a probability vector x over m alternatives;
  * agent i has a normalized pair (u_i, tau_i): min_j u_i(s_j) = 0,
    max_j u_i(s_j) = 1, and tau_i in (0, 1);
  * agent i rejects x iff <u_i, x> < tau_i; viol(x) counts rejecting agents.
Membership queries are not simulated: "recovering" an agent means reading its
normalized pair, which is what the recovery routine of Choo et al. returns.
"""
import json
import platform
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
import sys
import time

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

SEED = 2026
TOL = 1e-9  # agent i accepts x iff <u_i, x> >= tau_i - TOL (floating point)

# Experiments with m = 3 (Table 1)
N3 = 500
M3 = 3
N_POP = 20
ACCEPT_FRAC = 0.85
SAMPLE_SIZES = (30, 120, 480)
N_SAMPLES = 6
ETAS = (0.02, 0.05)
LP_SAMPLE_SIZE = 120
K_VALUES = (10, 25, 50, 100, 200, 400)
N_SUBSETS = 6
SETTINGS = ("separated", "near-threshold")
# margin <u_i, x0> - tau_i of agents who accept the planted lottery x0
ACCEPT_MARGIN = {"separated": (0.05, 0.25), "near-threshold": (0.0, 0.05)}
# amount tau_i - <u_i, x0> by which the other agents reject x0
REJECT_GAP = (0.05, 0.25)

# Few-alternatives experiment (Theorem 4.2)
NF, MF, ETA_F = 1000, 500, 0.15
S_ETA = int(np.ceil(np.log(2 * NF) / (2 * ETA_F ** 2)))  # = 169
S_VALUES = (50, S_ETA)
N_TRIALS = 480
NEAR_FRAC = 0.20


# ----------------------------------------------------------------------------
# Basic quantities
# ----------------------------------------------------------------------------
def normalize_rows(raw):
    lo = raw.min(axis=1, keepdims=True)
    hi = raw.max(axis=1, keepdims=True)
    return (raw - lo) / (hi - lo)


def viol(U, tau, x):
    """Number of agents (rows of U) who reject the lottery x."""
    return int(np.sum(U @ x < tau - TOL))


def clean(x):
    x = np.clip(x, 0.0, None)
    return x / x.sum()


# ----------------------------------------------------------------------------
# Exact minimization for m = 3: the vertex enumeration of Theorem 4.1
# ----------------------------------------------------------------------------
def exact_min_viol_m3(U, tau, w=None):
    """Return (x, weighted violation count) for a lottery minimizing the
    weighted number of rejecting agents.  Candidate lotteries are the unique
    solutions of x1+x2+x3 = 1 together with two tight constraints, each of the
    form x_j = 0 or <u_i, x> = tau_i (Theorem 4.1)."""
    d = len(tau)
    w = np.ones(d) if w is None else np.asarray(w, dtype=float)
    # coordinates y = (x1, x2), x3 = 1 - x1 - x2; each constraint is a.y = b
    A = np.column_stack([U[:, 0] - U[:, 2], U[:, 1] - U[:, 2]])
    b = tau - U[:, 2]
    A = np.vstack([A, [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]])
    b = np.concatenate([b, [0.0, 0.0, -1.0]])
    I, J = np.triu_indices(len(b), 1)
    det = A[I, 0] * A[J, 1] - A[I, 1] * A[J, 0]
    keep = np.abs(det) > 1e-12
    I, J, det = I[keep], J[keep], det[keep]
    y1 = (b[I] * A[J, 1] - b[J] * A[I, 1]) / det
    y2 = (A[I, 0] * b[J] - A[J, 0] * b[I]) / det
    X = np.column_stack([y1, y2, 1.0 - y1 - y2])
    X = X[(X >= -TOL).all(axis=1)]
    best_val, best_x = np.inf, None
    for start in range(0, len(X), 20000):
        C = X[start:start + 20000]
        vals = (C @ U.T < tau - TOL) @ w
        t = int(np.argmin(vals))
        if vals[t] < best_val:
            best_val, best_x = vals[t], C[t]
    return clean(best_x), float(best_val)


def exact_on_agents(U, tau, idx):
    """Exact minimization over the agents in idx (with multiplicity)."""
    agents, counts = np.unique(idx, return_counts=True)
    x, _ = exact_min_viol_m3(U[agents], tau[agents], counts)
    return x


# ----------------------------------------------------------------------------
# The linear program (4) of Section 5.1
# ----------------------------------------------------------------------------
def lp_lottery(U, tau, w, eta):
    d, m = U.shape
    w = np.asarray(w, dtype=float)
    c = np.concatenate([np.zeros(m), w / w.sum()])
    # eta * z_t >= tau_t + eta - <u_t, x>
    A_ub = np.hstack([-U, -eta * np.eye(d)])
    b_ub = -(tau + eta)
    A_eq = np.concatenate([np.ones(m), np.zeros(d)])[None, :]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * (m + d), method="highs")
    assert res.status == 0, res.message
    return clean(res.x[:m])


# ----------------------------------------------------------------------------
# Mixed-integer programming check of the exact optimum
# ----------------------------------------------------------------------------
def mip_opt(U, tau, w=None):
    """min sum_i w_i y_i s.t. <u_i, x> + y_i >= tau_i, x in simplex, y binary.
    Since <u_i, x> >= 0 and tau_i <= 1, y_i = 1 makes constraint i vacuous."""
    n, m = U.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    c = np.concatenate([np.zeros(m), w])
    cons = [LinearConstraint(np.hstack([U, np.eye(n)]), tau, np.inf),
            LinearConstraint(np.concatenate([np.ones(m), np.zeros(n)])[None, :], 1, 1)]
    integrality = np.concatenate([np.zeros(m), np.ones(n)])
    bounds = Bounds(np.zeros(m + n), np.concatenate([np.full(m, np.inf), np.ones(n)]))
    res = milp(c, constraints=cons, integrality=integrality, bounds=bounds,
               options={"mip_rel_gap": 0.0})
    assert res.status == 0, res.message
    return int(round(res.fun)), clean(res.x[:m])


# ----------------------------------------------------------------------------
# Populations with m = 3
# ----------------------------------------------------------------------------
def make_population(rng, setting, n=N3, m=M3):
    """A planted lottery x0, drawn uniformly from the simplex, is accepted by
    85% of the agents.  Each agent's utilities are m independent uniform
    values on [0,1], rescaled so that the smallest is 0 and the largest 1.
    Its threshold is <u_i, x0> minus a margin (accepting agents) or plus a
    gap (rejecting agents); an agent whose threshold falls outside (0, 1) is
    drawn again."""
    x0 = rng.dirichlet(np.ones(m))
    n_acc = int(round(ACCEPT_FRAC * n))
    U = np.empty((n, m))
    tau = np.empty(n)
    redraws = 0
    for i in range(n):
        while True:
            u = normalize_rows(rng.random((1, m)))[0]
            if i < n_acc:
                lo, hi = ACCEPT_MARGIN[setting]
                t = u @ x0 - rng.uniform(lo, hi)
            else:
                t = u @ x0 + rng.uniform(*REJECT_GAP)
            if 0.0 < t < 1.0:
                break
            redraws += 1
        U[i], tau[i] = u, t
    perm = rng.permutation(n)
    return U[perm], tau[perm], x0, redraws


def near_count(U, tau, x, eta):
    """|B_eta(x)|: agents who accept x with margin less than eta."""
    marg = U @ x - tau
    return int(np.sum((marg >= -TOL) & (marg < eta)))


def experiment_m3(setting, s_id, mip_check=True):
    out = {"opt": [], "viol_x0": [], "B002_x0": [], "B005_x0": [], "redraws": 0,
           "mip_agree": [], "mip_agree_sample": [], "exact": {k: [] for k in SAMPLE_SIZES},
           "lp_all": {e: [] for e in ETAS},
           "lp_sample": {e: [] for e in ETAS},
           "Kgap": {K: [] for K in K_VALUES},
           "time_opt": 0.0, "time_mip": 0.0, "time_rest": 0.0}
    for p in range(N_POP):
        rng = np.random.default_rng([SEED, s_id, p])
        U, tau, x0, red = make_population(rng, setting)
        out["redraws"] += red
        t0 = time.perf_counter()
        x_opt, opt = exact_min_viol_m3(U, tau)
        opt = int(opt)
        assert viol(U, tau, x_opt) == opt
        out["time_opt"] += time.perf_counter() - t0
        out["opt"].append(opt)
        out["viol_x0"].append(viol(U, tau, x0))
        out["B002_x0"].append(near_count(U, tau, x0, 0.02))
        out["B005_x0"].append(near_count(U, tau, x0, 0.05))
        if mip_check:
            t0 = time.perf_counter()
            mval, mx = mip_opt(U, tau)
            out["time_mip"] += time.perf_counter() - t0
            out["mip_agree"].append(bool(mval == opt and viol(U, tau, mx) == opt))
        t0 = time.perf_counter()
        # exact minimization on k agents sampled uniformly with replacement
        for k in SAMPLE_SIZES:
            for _ in range(N_SAMPLES):
                idx = rng.integers(0, N3, size=k)
                x_hat = exact_on_agents(U, tau, idx)
                gap = viol(U, tau, x_hat) - opt
                assert gap >= 0
                out["exact"][k].append(gap)
                if k == LP_SAMPLE_SIZE:  # the LP on the same samples
                    agents, counts = np.unique(idx, return_counts=True)
                    if mip_check:  # MIP check of exact minimization on the sample
                        Us, ts = U[agents], tau[agents]
                        _, sv = exact_min_viol_m3(Us, ts, counts)
                        mval, mx = mip_opt(Us, ts, counts)
                        out["mip_agree_sample"].append(bool(
                            mval == int(sv) and int((Us @ mx < ts - TOL) @ counts) == int(sv)))
                    for eta in ETAS:
                        x_lp = lp_lottery(U[agents], tau[agents], counts, eta)
                        out["lp_sample"][eta].append(viol(U, tau, x_lp) - opt)
        # the LP on all agents
        for eta in ETAS:
            x_lp = lp_lottery(U, tau, np.ones(N3), eta)
            out["lp_all"][eta].append(viol(U, tau, x_lp) - opt)
        # K distinct agents chosen uniformly at random, exact minimization
        for K in K_VALUES:
            for _ in range(N_SUBSETS):
                idx = rng.choice(N3, size=K, replace=False)
                x_hat = exact_on_agents(U, tau, idx)
                out["Kgap"][K].append(viol(U, tau, x_hat) - opt)
        out["time_rest"] += time.perf_counter() - t0
    return out


def lower_bound_profile():
    """The profile of Theorem 6.1 for the algorithm that recovers the first K
    agents and minimizes the number of violations among them."""
    gaps = {}
    for K in K_VALUES:
        n = N3
        # answers seen by the algorithm: accept iff x3 = 0, i.e. u = (1,1,0), tau = 1
        U = np.tile([1.0, 1.0, 0.0], (n, 1))
        tau = np.ones(n)
        x_hat = exact_on_agents(U, tau, np.arange(K))
        z = np.array([1.0, 0, 0]) if not np.allclose(x_hat, [1, 0, 0]) else np.array([0, 1.0, 0])
        U[K:] = z  # unqueried agents accept only the pure lottery z
        _, opt = exact_min_viol_m3(U, tau)
        gaps[K] = viol(U, tau, x_hat) - int(opt)
    return gaps


# ----------------------------------------------------------------------------
# Few alternatives (Theorem 4.2)
# ----------------------------------------------------------------------------
def experiment_few(setting, s_id):
    """Every agent accepts the starting lottery x.  Margins <u_i,x> - tau_i are
    uniform on [eta, eta + 0.05]; in the harder ("near-threshold") setting, 20%
    of the agents instead have margins uniform on [0, eta).  We draw s alternatives
    independently from x, as in the proof of Theorem 4.2, and count agents who
    reject the empirical distribution (new rejections)."""
    res = {s: {"new": [], "max_drop": []} for s in S_VALUES}
    res["B_eta"] = []
    t0 = time.perf_counter()
    for trial in range(N_TRIALS):
        rng = np.random.default_rng([SEED, 100 + s_id, trial])
        U = normalize_rows(rng.random((NF, MF)))
        x = rng.dirichlet(np.ones(MF))
        margin = rng.uniform(ETA_F, ETA_F + 0.05, size=NF)
        if setting == "near-threshold":
            n_near = int(round(NEAR_FRAC * NF))
            margin[:n_near] = rng.uniform(0.0, ETA_F, size=n_near)
        Ux = U @ x
        tau = Ux - margin
        assert np.all((tau > 0) & (tau < 1))
        res["B_eta"].append(near_count(U, tau, x, ETA_F))
        for s in S_VALUES:
            J = rng.choice(MF, size=s, p=x)
            x_tilde = np.bincount(J, minlength=MF) / s
            Ut = U @ x_tilde
            res[s]["new"].append(int(np.sum(Ut < tau - TOL)))
            res[s]["max_drop"].append(float(np.max(Ux - Ut)))
    res["time"] = time.perf_counter() - t0
    return res


# ----------------------------------------------------------------------------
def rnd(q, nd=2):
    """Round a Fraction half-up to nd decimals (as reported in the paper)."""
    d = Decimal(q.numerator) / Decimal(q.denominator)
    return float(d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def mean_q(v):
    return Fraction(int(sum(v)), len(v))


def p95_q(v):
    """95th percentile with linear interpolation (numpy's default), exactly."""
    v = sorted(int(a) for a in v)
    h = Fraction(len(v) - 1) * Fraction(95, 100)
    lo = int(h)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (h - lo) * (v[hi] - v[lo])


def pct(v, n):
    """(mean, 95th percentile) of 100 * v / n, in percent."""
    return rnd(100 * mean_q(v) / n), rnd(100 * p95_q(v) / n)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    T0 = time.perf_counter()
    results = {"env": {"python": platform.python_version(), "numpy": np.__version__,
                       "scipy": scipy.__version__, "machine": platform.processor() or platform.machine()}}
    summary = {}
    for s_id, setting in enumerate(SETTINGS):
        r = experiment_m3(setting, s_id)
        results[setting] = r
        sm = {"OPT/n mean %": rnd(100 * mean_q(r["opt"]) / N3),
              "OPT range": (int(min(r["opt"])), int(max(r["opt"]))),
              "viol(x0) mean": float(np.mean(r["viol_x0"])),
              "|B_0.02(x0)| mean": rnd(mean_q(r["B002_x0"])),
              "|B_0.05(x0)| mean": float(np.mean(r["B005_x0"])),
              "redraws": r["redraws"],
              "MIP agrees (populations)": f"{sum(r['mip_agree'])}/{len(r['mip_agree'])}",
              "MIP agrees (samples k=120)": f"{sum(r['mip_agree_sample'])}/{len(r['mip_agree_sample'])}"}
        for k in SAMPLE_SIZES:
            sm[f"exact k={k} (mean%, p95%)"] = pct(r["exact"][k], N3)
            sm[f"exact k={k} mean agents"] = rnd(mean_q(r["exact"][k]))
        for e in ETAS:
            sm[f"LP eta={e} k={LP_SAMPLE_SIZE}"] = pct(r["lp_sample"][e], N3)
            sm[f"LP eta={e} all"] = pct(r["lp_all"][e], N3)
        for K in K_VALUES:
            g = r["Kgap"][K]
            sm[f"K={K}: mean gap, p95, % of n-K"] = (rnd(mean_q(g)), rnd(p95_q(g)),
                                                     rnd(100 * mean_q(g) / (N3 - K)))
        sm["time (s): opt / mip / rest"] = (round(r["time_opt"], 1), round(r["time_mip"], 1), round(r["time_rest"], 1))
        summary[setting] = sm
    t0 = time.perf_counter()
    results["lower_bound"] = lower_bound_profile()
    summary["lower bound gaps (should be n-K)"] = {K: (g, N3 - K) for K, g in results["lower_bound"].items()}
    summary["lower bound time (s)"] = round(time.perf_counter() - t0, 1)
    for s_id, setting in enumerate(SETTINGS):
        r = experiment_few(setting, s_id)
        results["few-" + setting] = r
        sm = {"s_eta": S_ETA, "|B_eta(x)| mean": rnd(mean_q(r["B_eta"]))}
        for s in S_VALUES:
            new = np.asarray(r[s]["new"])
            drop = np.asarray(r[s]["max_drop"])
            sm[f"s={s}: trials with no new rejection"] = f"{int(np.sum(new == 0))}/{N_TRIALS}"
            sm[f"s={s}: new rejection rate (mean%, p95%)"] = pct(new, NF)
            sm[f"s={s}: max new rejections"] = int(new.max())
            sm[f"s={s}: trials in which no agent loses more than eta"] = f"{int(np.sum(drop <= ETA_F))}/{N_TRIALS}"
            sm[f"s={s}: largest drop"] = round(float(drop.max()), 4)
        sm["time (s)"] = round(r["time"], 1)
        summary["few-" + setting] = sm
    summary["total time (s)"] = round(time.perf_counter() - T0, 1)
    results["summary"] = summary
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
