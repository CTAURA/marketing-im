#!/usr/bin/env python3
"""Measure the impact of Algorithm 2 line 35 as ACTUALLY PRINTED in the paper.

The paper's Algorithm 2 (page 9 of viral_marketing.pdf) prints:

    35:  I[m,k] = max( I[m-1,k],  I[C,k-1] + dI_m )
    36:  if    I[C,k-1] + dI_m >= I[m-1,k]  then

Both lines read ``I[C,k-1]``.  SPEC.md Section 6 transcribes line 35 as
``I[m,k-1] + dI_m``, and DEVIATIONS.md 1.1 (severity A) is built on the
resulting "line 35 vs line 36 contradiction".  That contradiction is an
artefact of the transcription: in the paper the two lines agree.

``ctim/ctim.py`` implements line 35 as ``Iv[m][k-1] + dI_m`` in BOTH
``dp_tiebreak`` modes, so neither mode reproduces the printed recurrence.

This script does NOT modify ctim/ctim.py.  It re-runs the selection loop three
ways on identical inputs and reports whether the seed sets differ:

  paper-true   line 35 and line 36 both use I[C,k-1]   <- what the paper prints
  consistent   line 35 and line 36 both use I[m,k-1]   <- ctim.py default
  paper-literal line 35 uses I[m,k-1], line 36 uses I[C,k-1]  <- ctim.py's other mode

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

import os
import pickle
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ctim.ctim import _CommunitySeedState, detect_communities  # noqa: E402
from ctim.dataset import load_dataset                          # noqa: E402
from ctim.influence import MIA, EdgeWeights                    # noqa: E402


def run_dp(comm, ds, pp, K, h, mode, states=None):
    """Algorithm 2 lines 25-45 with a selectable reading of lines 35/36.

    mode:
      "paper-true"    line 35 ref = I[C,k-1],  line 36 ref = I[C,k-1]
      "consistent"    line 35 ref = I[m,k-1],  line 36 ref = I[m,k-1]
      "paper-literal" line 35 ref = I[m,k-1],  line 36 ref = I[C,k-1]
    """
    n_comm = (max(comm) + 1) if len(comm) else 0
    members = [[] for _ in range(n_comm)]
    for v in range(min(len(comm), ds.n_users)):
        members[comm[v]].append(v)

    # fresh per-community state (the incremental IncInf cache behind line 34)
    st = {}
    for m in range(1, n_comm + 1):
        mem = members[m - 1]
        if not mem:
            continue
        mia_m = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h, nodes=mem)
        st[m] = _CommunitySeedState(m, mia_m, mem)

    C = n_comm
    S = []
    Iv = [[0.0] * (K + 1) for _ in range(C + 1)]   # lines 26-28, 29-31
    sp = [[0] * (K + 1) for _ in range(C + 1)]

    for k in range(1, K + 1):                       # line 32
        for m in range(1, C + 1):                   # line 33
            s_m = st.get(m)
            dI_m = 0.0 if s_m is None else s_m.best()[1]     # line 34

            prev = Iv[m - 1][k]
            ref35 = Iv[C][k - 1] if mode == "paper-true" else Iv[m][k - 1]
            cand = ref35 + dI_m
            Iv[m][k] = prev if prev > cand else cand         # line 35

            ref36 = Iv[m][k - 1] if mode == "consistent" else Iv[C][k - 1]
            sp[m][k] = m if ref36 + dI_m >= prev else sp[m - 1][k]   # lines 36-40

        j = sp[C][k]                                # line 42
        s_j = st.get(j)
        u_k = None if s_j is None else s_j.best()[0]
        if u_k is None:                             # documented fallback
            best_g = None
            for m2 in sorted(st):
                u2, g2 = st[m2].best()
                if u2 is None:
                    continue
                if best_g is None or g2 > best_g:
                    best_g, u_k, j = g2, u2, m2
            if u_k is None:
                break
            s_j = st[j]
        s_j.add(u_k)                                # line 44
        S.append(u_k)
    return S, Iv[C][K]


def main():
    t_all = time.perf_counter()
    K, h = 20, 0.1
    ds = load_dataset(os.path.join(_REPO_ROOT, "data", "processed", "digg"))
    with open(os.path.join(_REPO_ROOT, "data", "processed", "_calib_model.pkl"), "rb") as fh:
        blob = pickle.load(fh)
    model, item = blob["model"], blob["test_items"][0]
    ew = EdgeWeights(model, ds)
    t0 = time.perf_counter()
    pp = ew.for_item(item)                          # Eq (12)
    print("[setup] Eq (12) weights            %.2fs" % (time.perf_counter() - t0))
    comm = detect_communities(model.pi)             # Eq (19)
    evaluator = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h)

    print("\n%-16s %12s %12s %10s  %s"
          % ("line-35 reading", "I(S) Eq(18)", "DP value", "secs", "seeds[:6]"))
    out = {}
    for mode in ("paper-true", "consistent", "paper-literal"):
        t0 = time.perf_counter()
        S, dpv = run_dp(comm, ds, pp, K, h, mode)
        secs = time.perf_counter() - t0
        val = evaluator.influence(S)                # Eq (18), shared evaluator
        out[mode] = S
        print("%-16s %12.4f %12.4f %9.2fs  %s"
              % (mode, val, dpv, secs, S[:6]))

    print("\nseed-set agreement:")
    base = out["paper-true"]
    for mode, S in out.items():
        same = (S == base)
        shared = len(set(S) & set(base))
        print("  %-16s identical=%-5s  shared %d/%d"
              % (mode, same, shared, len(base)))

    print("\nTOTAL %.2fs" % (time.perf_counter() - t_all))


if __name__ == "__main__":
    main()
