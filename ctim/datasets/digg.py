"""Digg benchmark preparation -- SPEC.md Section 8, benchmark #2.

The paper reports Digg as **30,358 users / 99,846 directed arcs / 7,100 items**.

Raw sources understood by this module (all auto-detected under ``data/raw/``):

``digg-friends/out.digg-friends``
    KONECT "Digg friends" network.  ``% asym unweighted`` header line, then
    ``from to weight timestamp``, whitespace separated, 1,731,653 arcs.
``digg-votes/out.digg-votes``
    KONECT "Digg votes" bipartite network, ``user story weight timestamp``.
``digg_friends.csv``
    The original Hogg & Lerman 2009 release: CRLF-terminated, quoted CSV,
    ``mutual, friend_date, user_id, friend_id``.
``digg_votes1.csv``
    The original release's votes: CR-terminated (classic-Mac line endings!),
    quoted CSV, ``vote_date, voter_id, story_id``.

**Why the original CSVs are preferred over the KONECT pair.**  KONECT renumbers
nodes per network, and it renumbers ``digg-friends`` and ``digg-votes``
*independently*: user ``1`` of the friends graph is original user 336224 while
user ``1`` of the votes graph is original user 318.  Joining votes to the graph
therefore requires the original id space, so when both original CSVs are
present we read those; the KONECT files are used as a fallback (and for the
graph alone they are equivalent).  Either way this module renumbers everything
to the dense ``[0, n)`` ids the canonical format requires.

**Arc direction.**  In both formats a row says "user_id became a fan of
friend_id", i.e. the fan subscribes to the friend's activity feed.  Influence
therefore flows ``friend_id -> user_id``, which is the reverse of the listed
column order.  That is the default (``arc_direction="influence"``);
``arc_direction="raw"`` keeps the columns as-is.

**Item attributes.**  The public Digg 2009 release contains no story topic /
container / category field -- the ``digg_votes1.csv`` and KONECT files carry
only ``(user, story, timestamp)``.  Attributes are therefore *derived* from
each story's observed diffusion profile (popularity, lifetime, time-of-day,
weekday, burstiness) plus a hashed story-id block, exactly as documented in
:func:`derive_item_attributes`.  ``meta.json["notes"]`` says so explicitly.

Standard library only.  Python 3.9 compatible.  All randomness flows through an
explicit ``random.Random`` instance.
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    # Direct execution (`python3 ctim/datasets/<mod>.py`) has no package
    # context, so put the repo root on sys.path before importing ctim.*.
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__)))))

import hashlib
import math
import os
import random
import time
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from ctim.dataset import Dataset, build_adjacency, save_dataset
from ctim.datasets import PAPER_TARGETS, paper_comparison_table
from ctim.datasets.subsample import select_dense_core, thin_arcs

__all__ = [
    "DIGG_PAPER_USERS",
    "DIGG_PAPER_LINKS",
    "DIGG_PAPER_ITEMS",
    "find_digg_files",
    "iter_friend_arcs",
    "iter_votes",
    "derive_item_attributes",
    "simulate_adoption_logs",
    "prepare_digg",
]

DIGG_PAPER_USERS, DIGG_PAPER_LINKS, DIGG_PAPER_ITEMS = PAPER_TARGETS["digg"]

# Attribute-block layout used by derive_item_attributes(); see its docstring.
_N_POPULARITY_BUCKETS = 10
_N_LIFETIME_BUCKETS = 12
_N_HOUR_BUCKETS = 8       # 3-hour bands
_N_WEEKDAY_BUCKETS = 7
_N_BURST_BUCKETS = 10
_N_DERIVED = (_N_POPULARITY_BUCKETS + _N_LIFETIME_BUCKETS + _N_HOUR_BUCKETS
              + _N_WEEKDAY_BUCKETS + _N_BURST_BUCKETS)


# ---------------------------------------------------------------------------
# Raw file discovery and parsing
# ---------------------------------------------------------------------------


def find_digg_files(raw_dir: str) -> Dict[str, Optional[str]]:
    """Locate whatever Digg raw files happen to be present under ``raw_dir``.

    Returns a dict with the keys ``friends_csv``, ``votes_csv``,
    ``friends_konect``, ``votes_konect`` (``None`` when absent) plus the
    resolved ``friends`` / ``votes`` paths actually chosen, and ``id_space``
    describing which id space the choice lives in.
    """
    def first(*rel: str) -> Optional[str]:
        for r in rel:
            p = os.path.join(raw_dir, r)
            if os.path.isfile(p):
                return p
        return None

    found: Dict[str, Optional[str]] = {
        "friends_csv": first("digg_friends.csv", "digg2009/digg_friends.csv"),
        "votes_csv": first("digg_votes1.csv", "digg_votes.csv",
                           "digg2009/digg_votes1.csv"),
        "friends_konect": first("digg-friends/out.digg-friends", "out.digg-friends"),
        "votes_konect": first("digg-votes/out.digg-votes", "out.digg-votes"),
    }

    # The original CSV pair shares one id space, so votes can be joined to the
    # graph.  Prefer it whenever both halves are there.
    if found["friends_csv"] and found["votes_csv"]:
        found["friends"] = found["friends_csv"]
        found["votes"] = found["votes_csv"]
        found["id_space"] = "original Hogg & Lerman 2009 user ids (friends and votes join)"
    else:
        found["friends"] = found["friends_konect"] or found["friends_csv"]
        if found["friends_csv"] and found["friends"] is found["friends_csv"]:
            found["votes"] = found["votes_csv"]
        else:
            # KONECT graph: only the KONECT votes file shares nothing with it,
            # so votes are usable only if they came from the original release.
            found["votes"] = found["votes_csv"]
        found["id_space"] = "KONECT digg-friends node ids"
    return found


def _iter_lines_any_eol(path: str, chunk_size: int = 1 << 20) -> Iterator[bytes]:
    """Yield lines from ``path`` accepting LF, CRLF *and* bare-CR terminators.

    ``digg_votes1.csv`` uses classic-Mac bare ``\\r`` line endings, which
    Python's text mode does not split on when the file is opened in binary and
    which universal newlines would handle but at ~3x the cost for an 85 MB
    file.  Streaming in 1 MiB chunks keeps memory flat.
    """
    with open(path, "rb") as fh:
        tail = b""
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            data = tail + chunk
            data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            parts = data.split(b"\n")
            tail = parts.pop()
            for p in parts:
                if p:
                    yield p
        if tail:
            yield tail


def _sniff_format(path: str) -> str:
    """``"csv"`` for the quoted original release, ``"konect"`` for KONECT TSV."""
    with open(path, "rb") as fh:
        head = fh.read(4096)
    for line in head.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(b"%") or line.startswith(b"#"):
            return "konect"
        return "csv" if line.startswith(b'"') else "konect"
    return "konect"


def _split_csv_row(line: bytes) -> List[bytes]:
    """Split a row of the original Digg CSVs (plain quoted integers, no escapes)."""
    return line.replace(b'"', b"").split(b",")


def iter_friend_arcs(path: str, arc_direction: str = "influence",
                     fmt: Optional[str] = None) -> Iterator[Tuple[int, int]]:
    """Yield directed arcs ``(u, v)`` meaning "u can influence v".

    ``arc_direction="influence"`` (default) emits ``friend_id -> user_id``: the
    followed user influences the fan.  ``"raw"`` emits the columns unchanged.
    Self-loops are dropped; duplicates are *not* (the caller de-duplicates).
    """
    if arc_direction not in ("influence", "raw"):
        raise ValueError("arc_direction must be 'influence' or 'raw', got {!r}"
                         .format(arc_direction))
    fmt = fmt or _sniff_format(path)
    flip = (arc_direction == "influence")

    for line in _iter_lines_any_eol(path):
        if line[:1] in (b"%", b"#"):
            continue
        if fmt == "csv":
            # mutual, friend_date, user_id, friend_id
            parts = _split_csv_row(line)
            if len(parts) < 4:
                continue
            fan = int(parts[2])
            followed = int(parts[3])
        else:
            # KONECT: <user_id> <friend_id> [weight] [timestamp]
            parts = line.split()
            if len(parts) < 2:
                continue
            fan = int(parts[0])
            followed = int(parts[1])
        if fan == followed:
            continue
        yield (followed, fan) if flip else (fan, followed)


def iter_votes(path: str, fmt: Optional[str] = None) -> Iterator[Tuple[int, int, int]]:
    """Yield adoption records ``(user_id, story_id, unix_timestamp)``."""
    fmt = fmt or _sniff_format(path)
    for line in _iter_lines_any_eol(path):
        if line[:1] in (b"%", b"#"):
            continue
        if fmt == "csv":
            # vote_date, voter_id, story_id
            parts = _split_csv_row(line)
            if len(parts) < 3:
                continue
            yield (int(parts[1]), int(parts[2]), int(parts[0]))
        else:
            # KONECT: <user> <story> [weight] [timestamp]
            parts = line.split()
            if len(parts) < 4:
                continue
            yield (int(parts[0]), int(parts[1]), int(parts[3]))


# ---------------------------------------------------------------------------
# Item attributes
# ---------------------------------------------------------------------------


def _stable_hash(*parts: object) -> int:
    """Deterministic 64-bit hash.  ``hash()`` is salted per process, md5 is not."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.md5(payload).digest()[:8], "big")


def derive_item_attributes(item_times: Dict[int, List[int]],
                           n_hash_attrs: int = 32,
                           n_hash_draws: int = 2) -> Tuple[List[List[int]], int, List[str]]:
    """Binary attribute matrix ``A`` for Digg stories, derived from real votes.

    The public Digg 2009 release has **no story topic / container / category
    field**, so there is nothing categorical to hash directly (contrast Yelp,
    where ``categories`` gives real attributes).  Instead each story's bag
    ``w_i`` (Algorithm 1 line 6) is built from its *observed diffusion
    profile*, which is genuine measured data, plus a hashed-id block that gives
    the topic model a per-story identity signal:

    ======================  =========================================  ======
    block                   feature                                    width
    ======================  =========================================  ======
    popularity              decile of vote count                       10
    lifetime                ``log2(1 + span_minutes)`` bucket          12
    time-of-day             3-hour band of the first vote (UTC)        8
    weekday                 weekday of the first vote (UTC)            7
    burstiness              decile of the share of votes in hour 1     10
    hashed id               ``n_hash_draws`` md5 buckets of story id   n_hash
    ======================  =========================================  ======

    Every story gets exactly ``5 + n_hash_draws`` attribute tokens (fewer if
    two hash draws collide).  Returns ``(item_attrs_by_story, n_attrs, labels)``
    where ``item_attrs_by_story`` is indexed the same way as ``item_times``'
    sorted keys -- i.e. a list aligned to ``sorted(item_times)``.
    """
    stories = sorted(item_times)
    counts = [len(item_times[s]) for s in stories]
    order = sorted(range(len(stories)), key=lambda j: (counts[j], stories[j]))
    pop_bucket = [0] * len(stories)
    n = len(stories)
    for rank, j in enumerate(order):
        pop_bucket[j] = min(_N_POPULARITY_BUCKETS - 1,
                            (rank * _N_POPULARITY_BUCKETS) // max(1, n))

    base_pop = 0
    base_life = base_pop + _N_POPULARITY_BUCKETS
    base_hour = base_life + _N_LIFETIME_BUCKETS
    base_wday = base_hour + _N_HOUR_BUCKETS
    base_burst = base_wday + _N_WEEKDAY_BUCKETS
    base_hash = base_burst + _N_BURST_BUCKETS
    n_attrs = base_hash + n_hash_attrs

    labels: List[str] = []
    labels += ["popularity_decile_{}".format(b) for b in range(_N_POPULARITY_BUCKETS)]
    labels += ["lifetime_log2_{}".format(b) for b in range(_N_LIFETIME_BUCKETS)]
    labels += ["first_vote_hourband_{}".format(b) for b in range(_N_HOUR_BUCKETS)]
    labels += ["first_vote_weekday_{}".format(b) for b in range(_N_WEEKDAY_BUCKETS)]
    labels += ["burstiness_decile_{}".format(b) for b in range(_N_BURST_BUCKETS)]
    labels += ["story_hash_{}".format(b) for b in range(n_hash_attrs)]

    out: List[List[int]] = []
    for j, s in enumerate(stories):
        ts = sorted(item_times[s])
        t0 = ts[0]
        span_min = (ts[-1] - t0) / 60.0
        life = min(_N_LIFETIME_BUCKETS - 1, int(math.log2(1.0 + max(0.0, span_min))))
        # UTC calendar fields without importing datetime: unix epoch was a Thursday.
        day_index = t0 // 86400
        hour = (t0 % 86400) // 3600
        weekday = int((day_index + 4) % 7)
        in_first_hour = sum(1 for t in ts if t - t0 <= 3600)
        burst = min(_N_BURST_BUCKETS - 1,
                    int(_N_BURST_BUCKETS * in_first_hour / float(len(ts))))

        bag = {
            base_pop + pop_bucket[j],
            base_life + life,
            base_hour + int(hour) // 3,
            base_wday + weekday,
            base_burst + burst,
        }
        if n_hash_attrs > 0:
            for draw in range(n_hash_draws):
                bag.add(base_hash + _stable_hash("digg-story", s, draw) % n_hash_attrs)
        out.append(sorted(bag))

    return out, n_attrs, labels


# ---------------------------------------------------------------------------
# Simulated adoption logs (used only when no votes file is available)
# ---------------------------------------------------------------------------


def simulate_adoption_logs(n_users: int, out_adj: Sequence[Sequence[int]],
                           n_items: int, rng: random.Random,
                           n_topics: int = 8, n_comms: int = 100,
                           base_p: float = 0.02, topic_boost: float = 6.0,
                           n_init_seeds: int = 3, max_steps: int = 12,
                           t_start: int = 1234567890,
                           step_seconds: int = 3600) -> List[Tuple[int, int, int]]:
    """Independent-Cascade simulation over the REAL Digg arcs.

    Used **only** when no votes file is available.  The topology stays real;
    only the adoption process is simulated, and ``meta.json["notes"]`` says so
    loudly.  The documented process:

    1. Each user ``v`` is assigned a latent community ``k(v)`` uniformly at
       random, and each community a topic-preference vector over ``n_topics``
       topics drawn from a symmetric Dirichlet(0.4) (via normalised Gamma(0.4)
       variates), so communities have sharply peaked, distinct tastes.
    2. Each item ``i`` gets a topic ``z_i`` drawn uniformly.
    3. An IC cascade for item ``i`` starts from ``n_init_seeds`` users sampled
       with probability proportional to out-degree, and propagates for at most
       ``max_steps`` rounds.  Arc ``(u, v)`` fires with probability
       ``min(0.9, base_p * (1 + topic_boost * pref[k(v)][z_i]))`` -- i.e.
       users adopt items matching their community's taste far more readily,
       which is precisely the community/topic coupling CTIM is meant to
       recover.
    4. A node activated in round ``r`` is logged at
       ``t_start + r*step_seconds + jitter``, jitter uniform in
       ``[0, step_seconds)``, so Definition 1's ``0 < t_q - t_p <= Delta``
       window is meaningful.

    Returns ``(u, i, t)`` records sorted by ``t``.
    """
    if rng is None:
        raise ValueError("simulate_adoption_logs requires an explicit random.Random")

    comm_of = [rng.randrange(n_comms) for _ in range(n_users)]
    pref: List[List[float]] = []
    for _ in range(n_comms):
        g = [rng.gammavariate(0.4, 1.0) + 1e-12 for _ in range(n_topics)]
        s = sum(g)
        pref.append([x / s for x in g])

    degrees = [len(out_adj[u]) for u in range(n_users)]
    total_deg = sum(degrees)
    if total_deg == 0:
        raise ValueError("cannot simulate diffusion on a graph with no arcs")
    cum: List[int] = []
    running = 0
    for d in degrees:
        running += d
        cum.append(running)

    def sample_hub() -> int:
        from bisect import bisect_left as _bl
        return min(n_users - 1, _bl(cum, rng.randrange(total_deg) + 1))

    logs: List[Tuple[int, int, int]] = []
    for i in range(n_items):
        z = rng.randrange(n_topics)
        active: Set[int] = set()
        frontier: List[int] = []
        for _ in range(n_init_seeds):
            s = sample_hub()
            if s not in active:
                active.add(s)
                frontier.append(s)
        for s in frontier:
            logs.append((s, i, t_start + rng.randrange(step_seconds)))

        for step in range(1, max_steps + 1):
            nxt: List[int] = []
            for u in frontier:
                for v in out_adj[u]:
                    if v in active:
                        continue
                    p = base_p * (1.0 + topic_boost * pref[comm_of[v]][z])
                    if p > 0.9:
                        p = 0.9
                    if rng.random() < p:
                        active.add(v)
                        nxt.append(v)
                        logs.append((v, i, t_start + step * step_seconds
                                     + rng.randrange(step_seconds)))
            if not nxt:
                break
            frontier = nxt

    logs.sort(key=lambda r: (r[2], r[0], r[1]))
    return logs


# ---------------------------------------------------------------------------
# The preparer
# ---------------------------------------------------------------------------


def _dedup_first(records: Iterable[Tuple[int, int, int]]) -> Dict[Tuple[int, int], int]:
    """Collapse repeated adoptions of one item by one user to the EARLIEST time.

    SPEC.md Section 8: "we disregard the unfrequent behaviors of repeated
    review for the same item".  The KONECT Digg votes README warns the raw data
    does contain such duplicates.
    """
    first: Dict[Tuple[int, int], int] = {}
    for u, i, t in records:
        key = (u, i)
        prev = first.get(key)
        if prev is None or t < prev:
            first[key] = t
    return first


def prepare_digg(raw_dir: str, out_dir: str,
                 target_users: int = DIGG_PAPER_USERS,
                 target_links: int = DIGG_PAPER_LINKS,
                 target_items: int = DIGG_PAPER_ITEMS,
                 seed: int = 42,
                 arc_direction: str = "influence",
                 n_hash_attrs: int = 32,
                 force_simulate: bool = False,
                 write: bool = True,
                 verbose: bool = True) -> Dataset:
    """Build the processed Digg benchmark and (by default) write it to ``out_dir``.

    Pipeline, all of it deterministic given ``seed``:

    1. Read the friendship arcs (real) and the story votes (real when a votes
       file exists, otherwise simulated by :func:`simulate_adoption_logs` --
       announced loudly in ``meta.json["notes"]``).
    2. Restrict to **active** users: present in the friend graph *and* holding
       at least one vote.  Only such users can ever produce a
       potential-influence log (Definition 1), so the rest are dead weight.
    3. Shrink to ``target_users`` via :func:`~ctim.datasets.subsample.select_dense_core`
       (k-core -> largest weakly connected component -> best-first snowball).
    4. Shrink the induced arcs to ``target_links`` via
       :func:`~ctim.datasets.subsample.thin_arcs` (per-source quota + global fill).
    5. Keep the ``target_items`` most-voted stories among the surviving users.
    6. Derive the attribute matrix (:func:`derive_item_attributes`).
    7. Renumber users/items/attributes to dense ``[0, n)`` ids and save.

    Any ``target_* <= 0`` disables that truncation step.
    """
    rng = random.Random(seed)
    t_start = time.time()
    files = find_digg_files(raw_dir)

    if not files.get("friends"):
        raise FileNotFoundError(
            "No Digg friendship file found under {!r}.\n"
            "Expected one of:\n"
            "  digg-friends/out.digg-friends   (KONECT, http://konect.cc/networks/digg-friends)\n"
            "  digg_friends.csv                (Hogg & Lerman 2009 release)\n"
            .format(raw_dir))

    friends_path = files["friends"]
    votes_path = None if force_simulate else files.get("votes")

    if verbose:
        print("[digg] friends : {}".format(friends_path))
        print("[digg] votes   : {}".format(votes_path or "<none -- will simulate>"))
        print("[digg] id space: {}".format(files["id_space"]))

    # -- 1. arcs ------------------------------------------------------------
    arc_set: Set[Tuple[int, int]] = set()
    for arc in iter_friend_arcs(friends_path, arc_direction=arc_direction):
        arc_set.add(arc)
    if verbose:
        print("[digg] {:,} unique directed arcs read ({:.1f}s)"
              .format(len(arc_set), time.time() - t_start))

    notes_parts: List[str] = []
    simulated = False

    # -- 2. adoption records ------------------------------------------------
    if votes_path:
        first_time = _dedup_first(iter_votes(votes_path))
        adoptions = [(u, i, t) for (u, i), t in first_time.items()]
        notes_parts.append(
            "Adoption logs are REAL Digg story votes from {}; repeated votes on the "
            "same story by the same user collapsed to the earliest (SPEC.md Sec. 8)."
            .format(os.path.basename(votes_path)))
    else:
        # No votes file: keep the topology real, simulate only the diffusion.
        simulated = True
        nodes = sorted({x for arc in arc_set for x in arc})
        idx = {n: j for j, n in enumerate(nodes)}
        sim_edges = [(idx[u], idx[v]) for u, v in arc_set]
        sim_out, _ = build_adjacency(len(nodes), sim_edges)
        n_sim_items = target_items if target_items > 0 else 1000
        sim = simulate_adoption_logs(len(nodes), sim_out, n_sim_items, rng)
        adoptions = [(nodes[u], i, t) for u, i, t in sim]
        reason = ("simulation was explicitly forced (force_simulate=True) even though a "
                  "votes file is present" if force_simulate else
                  "no Digg votes file was found under {!r}".format(raw_dir))
        notes_parts.append(
            "*** SIMULATED ADOPTION LOGS ***  These adoption logs are NOT real: {}.  They "
            "were generated by Independent-Cascade diffusion over the REAL Digg friendship "
            "arcs with per-item topic-biased activation probabilities (see "
            "ctim.datasets.digg.simulate_adoption_logs, seed={}).  The graph topology is "
            "real; the adoption process is not.  Do not report these logs as observed "
            "Digg behaviour.".format(reason, seed))

    if verbose:
        print("[digg] {:,} de-duplicated adoption records ({:.1f}s)"
              .format(len(adoptions), time.time() - t_start))

    # -- 3. active users ----------------------------------------------------
    activity: Dict[int, int] = {}
    for u, _i, _t in adoptions:
        activity[u] = activity.get(u, 0) + 1
    graph_users: Set[int] = {x for arc in arc_set for x in arc}
    active: Set[int] = graph_users & set(activity)
    active_arcs = [(u, v) for (u, v) in arc_set if u in active and v in active]
    if verbose:
        print("[digg] active users (in graph AND voting): {:,};  induced arcs {:,}"
              .format(len(active), len(active_arcs)))
    if not active_arcs:
        raise ValueError("no arcs survive the active-user restriction -- "
                         "friends and votes are probably in different id spaces")

    # -- 4. densest active core --------------------------------------------
    selected, core_info = select_dense_core(active_arcs, target_users,
                                            activity=activity, verbose=verbose)
    kept_arcs = [(u, v) for (u, v) in active_arcs if u in selected and v in selected]
    if verbose:
        print("[digg] core selection: k*={} k-core={:,} WCC={:,} selected={:,} "
              "(induced arcs {:,})".format(core_info["k_star"], core_info["n_kcore"],
                                           core_info["n_wcc"], len(selected),
                                           len(kept_arcs)))

    # -- 5. arc budget ------------------------------------------------------
    kept_arcs = thin_arcs(kept_arcs, target_links, weight=activity,
                          required_nodes=selected)
    users_with_arcs: Set[int] = {x for arc in kept_arcs for x in arc}
    # Strand repair keeps almost everybody, but a user the budget genuinely
    # cannot afford is unreachable in the diffusion model, so drop them from U.
    selected = selected & users_with_arcs if users_with_arcs else selected
    kept_arcs = [(u, v) for (u, v) in kept_arcs if u in selected and v in selected]
    if verbose:
        print("[digg] after arc thinning: {:,} users, {:,} arcs"
              .format(len(selected), len(kept_arcs)))

    # -- 6. items -----------------------------------------------------------
    item_times: Dict[int, List[int]] = {}
    for u, i, t in adoptions:
        if u in selected:
            item_times.setdefault(i, []).append(t)
    if target_items > 0 and len(item_times) > target_items:
        ranked = sorted(item_times, key=lambda s: (-len(item_times[s]), s))
        keep_items = set(ranked[:target_items])
        item_times = {s: ts for s, ts in item_times.items() if s in keep_items}
    if verbose:
        print("[digg] items kept: {:,}".format(len(item_times)))

    # -- 7. attributes ------------------------------------------------------
    attrs_by_story, n_attrs, _labels = derive_item_attributes(
        item_times, n_hash_attrs=n_hash_attrs)

    # -- 8. renumber to dense ids ------------------------------------------
    user_ids = sorted(selected)
    umap = {u: j for j, u in enumerate(user_ids)}
    story_ids = sorted(item_times)
    imap = {s: j for j, s in enumerate(story_ids)}

    edges = sorted((umap[u], umap[v]) for u, v in kept_arcs)
    logs = [(umap[u], imap[i], t) for u, i, t in adoptions
            if u in umap and i in imap]
    logs.sort(key=lambda r: (r[2], r[0], r[1]))
    item_attrs = [attrs_by_story[j] for j in range(len(story_ids))]

    # Attribute ids must be dense in [0, n_attrs): derived blocks are sparse
    # (e.g. no story may land in lifetime bucket 11), so compact them.
    used = sorted({a for bag in item_attrs for a in bag})
    amap = {a: j for j, a in enumerate(used)}
    item_attrs = [[amap[a] for a in bag] for bag in item_attrs]
    n_attrs = len(used)

    out_adj, in_adj = build_adjacency(len(user_ids), edges)

    notes_parts.append(
        "Subsampled to approximately the paper's reported Digg size by the deterministic "
        "procedure: restrict to users that are both in the friend graph and have >=1 vote; "
        "{}; then per-source-quota arc thinning to the link budget; then keep the "
        "most-voted stories.  k*={}, k-core={}, largest WCC={}."
        .format(core_info["procedure"], core_info["k_star"],
                core_info["n_kcore"], core_info["n_wcc"]))
    notes_parts.append(
        "Arc direction '{}': a raw row says user_id became a fan of friend_id, so "
        "influence flows friend_id -> user_id (the reverse of the column order)."
        .format(arc_direction))
    notes_parts.append(
        "Item attributes are DERIVED, not real categories: the public Digg 2009 release "
        "carries no story topic/container/category field.  Blocks: popularity decile, "
        "log2 lifetime, first-vote 3h band, first-vote weekday, burstiness decile, "
        "plus {} hashed-story-id buckets.".format(n_hash_attrs))
    if len(item_times) < target_items:
        notes_parts.append(
            "NOTE: this Digg snapshot contains only {} distinct stories in total, fewer "
            "than the paper's reported {} items; every available story was kept."
            .format(len(item_times), target_items))

    ds = Dataset(
        name="digg",
        n_users=len(user_ids),
        n_items=len(story_ids),
        n_attrs=n_attrs,
        edges=edges,
        out_adj=out_adj,
        in_adj=in_adj,
        logs=logs,
        item_attrs=item_attrs,
        source="Digg 2009 (Hogg & Lerman / KONECT): friends={}, votes={}".format(
            os.path.basename(friends_path),
            os.path.basename(votes_path) if votes_path else "SIMULATED"),
        notes="  ".join(notes_parts),
    )

    if write:
        save_dataset(ds, out_dir)
        if verbose:
            print("[digg] written to {}".format(out_dir))

    if verbose:
        print()
        print(paper_comparison_table(
            {"n_users": ds.n_users, "n_links": ds.n_links, "n_items": ds.n_items,
             "n_attrs": ds.n_attrs, "n_logs": ds.n_logs},
            benchmark="digg",
            title="Digg -- produced vs. paper (SPEC.md Section 8)"))
        if simulated:
            print()
            print("!! WARNING: adoption logs are SIMULATED (no votes file found). "
                  "See meta.json['notes'].")
        print()
        print("[digg] done in {:.1f}s".format(time.time() - t_start))

    return ds


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="ctim-digg-selftest-")
    try:
        raw = os.path.join(tmp, "raw")
        os.makedirs(os.path.join(raw, "digg-friends"))

        # -- parser: CRLF-quoted CSV friends, CR-only quoted CSV votes -------
        fpath = os.path.join(raw, "digg_friends.csv")
        with open(fpath, "wb") as fh:
            for mutual, ts, user, friend in (
                (1, 100, 10, 20), (0, 101, 10, 30), (1, 102, 20, 30),
                (0, 103, 30, 10), (0, 104, 40, 10), (0, 105, 40, 40),
            ):
                fh.write('"{}","{}","{}","{}"\r\n'.format(mutual, ts, user, friend)
                         .encode("ascii"))
        arcs = list(iter_friend_arcs(fpath))
        # influence flows friend_id -> user_id, self-loop (40,40) dropped
        assert arcs == [(20, 10), (30, 10), (30, 20), (10, 30), (10, 40)], arcs
        raw_arcs = list(iter_friend_arcs(fpath, arc_direction="raw"))
        assert raw_arcs == [(u, v) for v, u in arcs], raw_arcs
        assert _sniff_format(fpath) == "csv"

        vpath = os.path.join(raw, "digg_votes1.csv")
        with open(vpath, "wb") as fh:  # bare CR terminators, as in the real file
            for ts, user, story in ((900, 10, 1), (950, 20, 1), (999, 30, 1),
                                    (800, 10, 2), (860, 30, 2), (870, 40, 2),
                                    (2000, 10, 1)):
                fh.write('"{}","{}","{}"\r'.format(ts, user, story).encode("ascii"))
        votes = list(iter_votes(vpath))
        assert len(votes) == 7 and votes[0] == (10, 1, 900), votes[:2]
        first = _dedup_first(votes)
        assert first[(10, 1)] == 900, first          # earliest kept, 2000 dropped
        assert len(first) == 6

        # -- KONECT format ---------------------------------------------------
        kpath = os.path.join(raw, "digg-friends", "out.digg-friends")
        with open(kpath, "wb") as fh:
            fh.write(b"% asym unweighted\n")
            fh.write(b"10 20 1\t100\n20 30 1\t101\n")
        assert _sniff_format(kpath) == "konect"
        assert list(iter_friend_arcs(kpath)) == [(20, 10), (30, 20)]

        kv = os.path.join(raw, "out.digg-votes")
        with open(kv, "wb") as fh:
            fh.write(b"% bip positive\n5 7 1\t1246573330\n")
        assert list(iter_votes(kv)) == [(5, 7, 1246573330)]

        # -- discovery prefers the joinable original CSV pair -----------------
        found = find_digg_files(raw)
        assert found["friends"] == fpath and found["votes"] == vpath, found
        assert "original" in found["id_space"]

        # -- attributes -------------------------------------------------------
        item_times = {1: [900, 950, 999], 2: [800, 860, 870], 3: [10, 90000000]}
        bags, n_attrs, labels = derive_item_attributes(item_times, n_hash_attrs=8)
        assert len(bags) == 3 and len(labels) == n_attrs
        assert n_attrs == _N_DERIVED + 8
        for bag in bags:
            assert bag == sorted(set(bag)) and 5 <= len(bag) <= 7
            assert all(0 <= a < n_attrs for a in bag)
        # deterministic across calls / processes (md5, not salted hash())
        assert derive_item_attributes(item_times, n_hash_attrs=8)[0] == bags
        # the long-lived item 3 must land in a higher lifetime bucket than item 1
        base_life = _N_POPULARITY_BUCKETS
        life = [next(a for a in b if base_life <= a < base_life + _N_LIFETIME_BUCKETS)
                for b in bags]
        assert life[2] > life[0], life

        # -- end-to-end on the toy raw files ---------------------------------
        out = os.path.join(tmp, "processed", "digg")
        ds = prepare_digg(raw, out, target_users=0, target_links=0, target_items=0,
                          seed=1, verbose=False)
        assert ds.n_users >= 3 and ds.n_links >= 3 and ds.n_items == 2
        assert all(0 <= u < ds.n_users and 0 <= v < ds.n_users for u, v in ds.edges)
        assert all(0 <= i < ds.n_items for _u, i, _t in ds.logs)
        assert all(0 <= a < ds.n_attrs for bag in ds.item_attrs for a in bag)
        assert ds.logs == sorted(ds.logs, key=lambda r: (r[2], r[0], r[1]))
        assert "REAL Digg story votes" in ds.notes

        from ctim.dataset import load_dataset
        back = load_dataset(out)
        assert back.n_users == ds.n_users and back.n_links == ds.n_links
        assert sorted(back.edges) == sorted(ds.edges) and back.logs == ds.logs

        # -- deterministic ----------------------------------------------------
        ds2 = prepare_digg(raw, out, target_users=0, target_links=0, target_items=0,
                           seed=1, write=False, verbose=False)
        assert ds2.edges == ds.edges and ds2.logs == ds.logs
        assert ds2.item_attrs == ds.item_attrs

        # -- truncation actually binds ----------------------------------------
        ds3 = prepare_digg(raw, out, target_users=3, target_links=3, target_items=1,
                           seed=1, write=False, verbose=False)
        assert ds3.n_users <= 3 and ds3.n_links <= 3 and ds3.n_items == 1

        # -- simulated-log fallback ------------------------------------------
        sim_raw = os.path.join(tmp, "raw_nosim")
        os.makedirs(sim_raw)
        big = os.path.join(sim_raw, "digg_friends.csv")
        with open(big, "wb") as fh:  # a small dense graph so cascades survive
            for u in range(40):
                for v in range(40):
                    if u != v and (u * 7 + v) % 3 == 0:
                        fh.write('"0","0","{}","{}"\r\n'.format(u, v).encode("ascii"))
        ds4 = prepare_digg(sim_raw, out, target_users=0, target_links=0,
                           target_items=20, seed=5, write=False, verbose=False)
        assert "*** SIMULATED ADOPTION LOGS ***" in ds4.notes
        assert ds4.n_logs > 0 and ds4.n_items > 0
        ds5 = prepare_digg(sim_raw, out, target_users=0, target_links=0,
                           target_items=20, seed=5, write=False, verbose=False)
        assert ds5.logs == ds4.logs, "simulation must be seed-deterministic"
        ds6 = prepare_digg(sim_raw, out, target_users=0, target_links=0,
                           target_items=20, seed=6, write=False, verbose=False)
        assert ds6.logs != ds4.logs, "a different seed must change the simulation"

        # force_simulate on a source that DOES have votes
        ds7 = prepare_digg(raw, out, target_users=0, target_links=0, target_items=5,
                           seed=3, force_simulate=True, write=False, verbose=False)
        assert "*** SIMULATED ADOPTION LOGS ***" in ds7.notes
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("ctim/datasets/digg.py self-test: OK")


if __name__ == "__main__":
    _self_test()
