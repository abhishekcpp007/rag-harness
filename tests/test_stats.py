"""The bootstrap. Fixed seed, so a changed number here means changed arithmetic.

What these pin is that the interval behaves: it brackets the mean, it narrows as the sample
grows, it collapses when there is no variance, and the paired version cancels the shared
question-to-question variation instead of adding it. The repo's conclusions now rest on this
file, which is reason enough to test it harder than the thing it measures.
"""
from __future__ import annotations

import numpy as np
import pytest

from eval.stats import holm, mean_ci, paired_delta

SMALL = 400


def test_the_interval_contains_the_sample_mean():
    ci = mean_ci([1.0, 0.5, 0.0, 1.0, 1.0, 0.33, 1.0, 0.25], resamples=SMALL)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert ci["n"] == 8


def test_no_variance_gives_no_interval():
    """All six identifier questions score exactly 1.0, so the bootstrap cannot say anything
    about them and the honest output is a point. The README has to say why rather than printing
    [1.00, 1.00] and letting it read as certainty."""
    ci = mean_ci([1.0] * 6, resamples=SMALL)
    assert (ci["mean"], ci["lo"], ci["hi"]) == (1.0, 1.0, 1.0)


def test_a_bigger_sample_narrows_the_interval():
    rng = np.random.default_rng(3)
    draw = rng.binomial(1, 0.7, size=400).astype(float)
    narrow = mean_ci(draw.tolist(), resamples=2000)
    wide = mean_ci(draw[:35].tolist(), resamples=2000)
    assert narrow["width"] < wide["width"] / 2


def test_undefined_questions_are_dropped_not_counted_as_zero():
    """citation_support is undefined when an answer cited nothing, and the reported mean drops
    those. Counting them as 0 would invent a failure."""
    assert mean_ci([1.0, None, 1.0], resamples=SMALL)["mean"] == 1.0
    assert mean_ci([1.0, float("nan"), 1.0], resamples=SMALL)["n"] == 2


def test_booleans_become_numbers():
    assert mean_ci([True, True, False, True], resamples=SMALL)["mean"] == 0.75


def test_an_empty_metric_is_not_an_exception():
    ci = mean_ci([None, None], resamples=SMALL)
    assert ci["n"] == 0
    assert np.isnan(ci["mean"])


def test_the_interval_is_reproducible():
    xs = [1.0, 0.5, 0.0, 1.0, 0.25]
    assert mean_ci(xs, resamples=SMALL) == mean_ci(xs, resamples=SMALL)


def test_pairing_finds_a_difference_that_the_marginals_hide():
    """Twenty questions whose difficulty varies wildly, and one configuration that is better on
    every single one by a hair. Unpaired, the spread across questions buries it. Paired, the
    spread cancels, because the same resampled questions go to both sides."""
    rng = np.random.default_rng(7)
    base = rng.uniform(0.0, 1.0, size=20)
    better = np.clip(base + 0.03, 0, 1)
    assert mean_ci(better.tolist(), resamples=4000)["lo"] < mean_ci(base.tolist(),
                                                                   resamples=4000)["hi"]
    d = paired_delta(better.tolist(), base.tolist(), resamples=4000)
    assert d["lo"] > 0.0
    assert d["p"] < 0.01


def test_identical_runs_give_a_zero_delta_and_no_claim():
    xs = [1.0, 0.5, 0.0, 1.0, 0.25, 1.0]
    d = paired_delta(xs, xs, resamples=SMALL)
    assert d["delta"] == 0.0
    assert d["questions_differing"] == 0
    assert d["zero_mass"] == 1.0
    assert d["crosses_zero"]


def test_one_question_moving_is_reported_as_one_question():
    """The whole reason this field exists. A difference carried by a single question out of 35
    is a fact about one question, and the interval alone does not say so."""
    a = [1.0] * 35
    b = [1.0] * 34 + [0.5]
    d = paired_delta(a, b, resamples=4000)
    assert d["questions_differing"] == 1
    assert d["zero_mass"] > 0.3


def test_the_paired_delta_refuses_to_pair_different_lengths_quietly():
    d = paired_delta([1.0], [1.0, 0.5], resamples=SMALL)
    assert d["n"] == 1


def test_holm_keeps_the_smallest_p_closest_to_itself():
    out = holm({"a": 0.001, "b": 0.02, "c": 0.4})
    assert out["a"]["p_holm"] == pytest.approx(0.003)
    assert out["a"]["significant"]
    assert not out["c"]["significant"]


def test_holm_never_lets_an_adjusted_p_fall_below_an_earlier_one():
    """Step down means the sequence is monotone. Without the running maximum a later test can
    come out more significant than a stricter earlier one, which is the bug this rule exists
    to avoid."""
    out = holm({"a": 0.04, "b": 0.041, "c": 0.042})
    ps = [out[k]["p_holm"] for k in ("a", "b", "c")]
    assert ps == sorted(ps)


def test_holm_with_one_test_changes_nothing():
    assert holm({"only": 0.03})["only"]["p_holm"] == 0.03
