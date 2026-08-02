"""Average Shannon entropy of a sequence of non-negative weight vectors.

    H^(t) = - sum_{i=1..K} p_i^(t) * log p_i^(t),    p_i^(t) = a_i^(t) / sum_j a_j^(t)
    Hbar  = (1/T) * sum_{t=1..T} H^(t)

`a^(t)` is any vector of `K` non-negative weights (counts, probabilities,
fitnesses); it is normalised internally, so callers pass raw quantities.

Not a quantity from the CTIM paper -- this is diagnostic instrumentation
(DEVIATIONS.md 4.7).  It exists to answer "did the model learn anything?": a row
of `theta`, `phi`, `psi` or `pi` that stays at its uniform prior carries no
information, and `Hbar` measures exactly that, in one number, across all rows.

Two readings of the same numbers are reported because both are useful:

  * **raw** `Hbar` in [0, log K] -- the printed formula, comparable only between
    runs with the same `K`;
  * **normalised** `Hbar / log K` in [0, 1] -- comparable across different `K`,
    which is what a `Z` sweep needs.  1.0 means uniform (no information),
    0.0 means a point mass.

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

import math

__all__ = ["entropy", "mean_entropy", "EntropyReport"]


def entropy(a, base=math.e, normalized=False):
    """`H^(t)` for one weight vector `a`.

    `0 * log 0` is taken as 0 (the standard convention and the limit).  A vector
    that sums to zero has no distribution at all; it is reported as maximal
    entropy, matching the uninformative prior it stands for.

    Raises ValueError on a negative weight -- that is a caller bug, not a
    degenerate distribution, and silently taking `abs` would hide it.
    """
    K = len(a)
    if K == 0:
        return 0.0
    total = 0.0
    for x in a:
        if x < 0.0:
            raise ValueError("entropy() needs non-negative weights, got %r" % (x,))
        total += x
    log_K = math.log(K, base) if K > 1 else 0.0
    if total <= 0.0:
        return 1.0 if normalized else log_K
    h = 0.0
    for x in a:
        if x > 0.0:
            p = x / total
            h -= p * math.log(p, base)
    if not normalized:
        return h
    return 1.0 if log_K <= 0.0 else h / log_K


class EntropyReport:
    """`Hbar` plus the per-step `H^(t)` it averages, and the shape it came from."""

    __slots__ = ("H", "mean", "T", "K", "base", "normalized")

    def __init__(self, H, T, K, base, normalized):
        self.H = H
        self.mean = (sum(H) / float(T)) if T else 0.0
        self.T = T
        self.K = K
        self.base = base
        self.normalized = normalized

    @property
    def max_possible(self):
        """`log K` for the raw form, 1.0 for the normalised one."""
        if self.normalized:
            return 1.0
        return math.log(self.K, self.base) if self.K > 1 else 0.0

    def __repr__(self):
        return ("EntropyReport(mean=%.5f, max=%.5f, T=%d, K=%d, normalized=%s)"
                % (self.mean, self.max_possible, self.T, self.K, self.normalized))


def mean_entropy(rows, base=math.e, normalized=False):
    """`Hbar` over `T = len(rows)` vectors of `K` weights each.

    Every row must have the same `K`; a ragged input is a caller bug and is
    rejected rather than averaged over inconsistent supports.
    """
    rows = list(rows)
    T = len(rows)
    if T == 0:
        return EntropyReport([], 0, 0, base, normalized)
    K = len(rows[0])
    for t, r in enumerate(rows):
        if len(r) != K:
            raise ValueError("row %d has K=%d, expected K=%d" % (t, len(r), K))
    H = [entropy(r, base=base, normalized=normalized) for r in rows]
    return EntropyReport(H, T, K, base, normalized)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    failures = []

    def check(name, cond, detail=""):
        print(("  PASS  " if cond else "  FAIL  ") + name
              + ((" -- " + detail) if detail else ""))
        if not cond:
            failures.append(name)

    print("[1] hand-computed values")
    check("uniform over K=4 gives log 4 (nats)",
          abs(entropy([1, 1, 1, 1]) - math.log(4)) < 1e-12,
          "%.6f vs %.6f" % (entropy([1, 1, 1, 1]), math.log(4)))
    check("uniform over K=4 gives 2 bits",
          abs(entropy([1, 1, 1, 1], base=2) - 2.0) < 1e-12)
    check("point mass gives 0", abs(entropy([0, 5, 0])) < 1e-12)
    check("normalisation is scale-free: [1,1] == [7,7]",
          abs(entropy([1, 1]) - entropy([7, 7])) < 1e-12)
    # p = (1/2, 1/4, 1/4) -> H = 1.5 bits
    check("H(1/2,1/4,1/4) = 1.5 bits",
          abs(entropy([2, 1, 1], base=2) - 1.5) < 1e-12,
          "%.6f" % entropy([2, 1, 1], base=2))

    print("[2] normalised form")
    check("uniform -> 1.0 for any K",
          all(abs(entropy([1] * K, normalized=True) - 1.0) < 1e-12
              for K in (2, 3, 8, 100)))
    check("point mass -> 0.0", abs(entropy([0, 1, 0, 0], normalized=True)) < 1e-12)
    check("normalised is base-independent",
          abs(entropy([5, 2, 1], normalized=True)
              - entropy([5, 2, 1], base=2, normalized=True)) < 1e-12)
    check("K=1 is degenerate -> 1.0 (no information possible)",
          entropy([3], normalized=True) == 1.0)

    print("[3] edge cases")
    check("empty vector -> 0.0", entropy([]) == 0.0)
    check("all-zero vector -> maximal", entropy([0, 0, 0], normalized=True) == 1.0)
    try:
        entropy([1, -1])
        check("negative weight raises ValueError", False)
    except ValueError:
        check("negative weight raises ValueError", True)

    print("[4] Hbar over T rows")
    rep = mean_entropy([[1, 1, 1, 1], [0, 4, 0, 0]], normalized=True)
    check("Hbar averages the per-row H", abs(rep.mean - 0.5) < 1e-12,
          "H=%s mean=%.6f" % ([round(x, 4) for x in rep.H], rep.mean))
    check("T and K are recorded", rep.T == 2 and rep.K == 4)
    check("max_possible is 1.0 when normalised", rep.max_possible == 1.0)
    raw = mean_entropy([[1, 1, 1, 1], [1, 1, 1, 1]])
    check("raw max_possible is log K", abs(raw.max_possible - math.log(4)) < 1e-12)
    check("raw Hbar of uniform rows == log K",
          abs(raw.mean - math.log(4)) < 1e-12)
    try:
        mean_entropy([[1, 1], [1, 1, 1]])
        check("ragged rows raise ValueError", False)
    except ValueError:
        check("ragged rows raise ValueError", True)
    check("empty input is handled", mean_entropy([]).mean == 0.0)

    print("[5] monotone under mixing (a sanity property)")
    # Moving mass toward uniform must not decrease entropy.
    prev = -1.0
    ok = True
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = entropy([1 - w * 0.5, w * 0.5], normalized=True) if w > 0 else 0.0
        # blend a point mass toward uniform
        a = [1.0 - w / 2.0, w / 2.0]
        h = entropy(a, normalized=True)
        if h < prev - 1e-12:
            ok = False
        prev = h
    check("entropy is non-decreasing as the vector approaches uniform", ok)

    print("")
    if failures:
        print("FAILED: %d check(s): %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("ALL CHECKS PASSED")
