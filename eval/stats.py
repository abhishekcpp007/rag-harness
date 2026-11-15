"""Confidence intervals for a 42 question golden set, because point estimates were overclaiming.

The tables in the README report differences like 0.864 against 0.850 and then argue about which
configuration wins. With 35 answerable questions, one question changing rank moves chunk MRR by
about 0.024, which is larger than several of the differences being argued about. A repo whose
whole claim is "measured rather than assumed" cannot leave that unstated.

So every headline number gets a percentile bootstrap interval, and every comparison against the
default gets a *paired* one: the same resampled question ids go to both configurations, so the
question-to-question variation that both share cancels instead of swamping the difference. An
unpaired comparison of two runs over the same questions throws away the pairing and reports a
wider interval than the data deserves.

Two deliberate choices. The interval is the percentile kind rather than bias corrected, because
the statistic is a plain mean and the correction changes the third digit on numbers whose second
digit is uncertain. And the comparison family gets a Holm adjustment, since ten configurations
against one default is ten chances to find a winner by luck.
"""
from __future__ import annotations

import math

import numpy as np

RESAMPLES = 10_000
ALPHA = 0.05
SEED = 20261115


def _clean(values) -> np.ndarray:
    """Drop the questions a metric does not apply to, the way eval.metrics.mean does.

    citation_support is undefined for an answer that cited nothing, and dropping those is how the
    reported number is computed, so the interval has to be computed over the same population.
    """
    out = []
    for v in values:
        if v is None:
            continue
        f = 1.0 if v is True else 0.0 if v is False else float(v)
        if math.isnan(f):
            continue
        out.append(f)
    return np.asarray(out, dtype=np.float64)


def _draws(n: int, resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def mean_ci(values, *, resamples: int = RESAMPLES, alpha: float = ALPHA,
            seed: int = SEED) -> dict:
    """Percentile bootstrap interval for the mean of one metric."""
    xs = _clean(values)
    if not len(xs):
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    if len(xs) == 1:
        v = float(xs[0])
        return {"n": 1, "mean": v, "lo": v, "hi": v}
    means = xs[_draws(len(xs), resamples, seed)].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "n": len(xs),
        "mean": round(float(xs.mean()), 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "width": round(float(hi - lo), 4),
    }


def paired_delta(a_values, b_values, *, resamples: int = RESAMPLES, alpha: float = ALPHA,
                 seed: int = SEED) -> dict:
    """Interval on (a minus b) where both ran the same questions in the same order.

    Only questions where the metric is defined on both sides are used, so the two means being
    differenced are over one population rather than two. The p value is the usual bootstrap
    two-sided one: how often a resample lands on the far side of zero, doubled.

    `questions_differing` and `zero_mass` are reported because this statistic has an atom at
    zero and the interval alone reads as broken without them. Two configurations that retrieve
    identically on 33 of 35 questions produce a pile of resamples whose mean difference is
    exactly 0, those resamples count on both sides of the test, and the p value comes out large
    next to an interval whose edge is a hair from zero. The number of questions that actually
    move is the plainer version of the same fact.
    """
    pairs = []
    for av, bv in zip(a_values, b_values):
        a = _clean([av])
        b = _clean([bv])
        if len(a) and len(b):
            pairs.append((float(a[0]), float(b[0])))
    if len(pairs) < 2:
        return {"n": len(pairs), "delta": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan")}
    arr = np.asarray(pairs, dtype=np.float64)
    diffs = arr[:, 0] - arr[:, 1]
    idx = _draws(len(diffs), resamples, seed)
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    below = float(np.mean(boot <= 0.0))
    above = float(np.mean(boot >= 0.0))
    p = min(1.0, 2.0 * min(below, above))
    return {
        "n": len(diffs),
        "questions_differing": int(np.count_nonzero(diffs)),
        "delta": round(float(diffs.mean()), 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "p": round(p, 4),
        "zero_mass": round(float(np.mean(boot == 0.0)), 4),
        "crosses_zero": bool(lo <= 0.0 <= hi),
    }


def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm step down adjustment over one family of comparisons.

    Ten configurations tested against one default is ten chances at a false winner. Holm is the
    cheapest correction that keeps the family wide error rate at alpha without assuming the tests
    are independent, which these are not: they share the same 35 questions.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        adjusted = min(1.0, (m - i) * p)
        running = max(running, adjusted)
        out[name] = {"p": round(p, 4), "p_holm": round(running, 4),
                     "significant": bool(running < alpha)}
    return out
