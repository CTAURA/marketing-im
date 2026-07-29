"""Deterministic graph subsampling shared by the benchmark preparers.

The paper (SPEC.md Section 8) reports datasets of a specific size:

    Yelp Dataset Challenge 2014   366,715 users   2,949,285 links   61,184 items
    Digg                           30,358 users      99,846 links    7,100 items

Neither raw source has exactly those dimensions any more (the Yelp 2014
snapshot is retired; the public Digg 2009 crawl is a *different* crawl from the
one the authors used).  To obtain a comparable benchmark we extract the
**densest active core** of the real graph and truncate it to the reported size
by the fully deterministic procedure documented in :func:`select_dense_core`
and :func:`thin_arcs`.  No randomness is involved anywhere in this module --
the same input always yields byte-identical output.

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "undirected_neighbours",
    "core_numbers",
    "largest_weakly_connected_component",
    "snowball_to_size",
    "select_dense_core",
    "thin_arcs",
]


# ---------------------------------------------------------------------------
# Basic graph utilities
# ---------------------------------------------------------------------------


def undirected_neighbours(arcs: Iterable[Tuple[int, int]]) -> Dict[int, Set[int]]:
    """Symmetrised adjacency: ``nb[x]`` = every node sharing an arc with ``x``.

    Self-loops are ignored.  Nodes with no incident arc simply do not appear.
    """
    nb: Dict[int, Set[int]] = {}
    for u, v in arcs:
        if u == v:
            continue
        s = nb.get(u)
        if s is None:
            s = set()
            nb[u] = s
        s.add(v)
        s = nb.get(v)
        if s is None:
            s = set()
            nb[v] = s
        s.add(u)
    return nb


def core_numbers(nb: Dict[int, Set[int]]) -> Dict[int, int]:
    """Core number of every node: the largest ``k`` with the node in the k-core.

    Standard peeling: repeatedly remove a node of minimum residual degree.  The
    running maximum of the removed degrees is the core number.  ``O(E log V)``
    with a lazy heap.  Deterministic: ties broken by ascending node id.
    """
    deg = {n: len(s) for n, s in nb.items()}
    heap = [(d, n) for n, d in deg.items()]
    heapq.heapify(heap)

    core: Dict[int, int] = {}
    removed: Set[int] = set()
    running = 0
    while heap:
        d, n = heapq.heappop(heap)
        if n in removed or d != deg[n]:
            continue  # stale heap entry
        if d > running:
            running = d
        core[n] = running
        removed.add(n)
        for w in nb[n]:
            if w not in removed:
                deg[w] -= 1
                heapq.heappush(heap, (deg[w], w))
    return core


def largest_weakly_connected_component(nodes: Set[int],
                                       nb: Dict[int, Set[int]]) -> Set[int]:
    """Largest connected component of the subgraph induced on ``nodes``.

    "Weakly" connected because ``nb`` is the symmetrisation of a directed graph.
    Deterministic: components are discovered in ascending node-id order, and
    ties on size are broken in favour of the component found first.
    """
    seen: Set[int] = set()
    best: Set[int] = set()
    for start in sorted(nodes):
        if start in seen:
            continue
        comp = {start}
        seen.add(start)
        stack = [start]
        while stack:
            x = stack.pop()
            for w in nb.get(x, ()):  # a node may have no neighbours inside
                if w in nodes and w not in seen:
                    seen.add(w)
                    comp.add(w)
                    stack.append(w)
        if len(comp) > len(best):
            best = comp
    return best


def snowball_to_size(pool: Set[int], nb: Dict[int, Set[int]], target: int,
                     priority: Dict[int, tuple]) -> Set[int]:
    """Best-first ("snowball") expansion inside ``pool`` down to ``target`` nodes.

    Starts from the highest-priority node of ``pool`` and repeatedly admits the
    highest-priority node adjacent to the set already admitted, so the result
    stays connected.  ``priority[n]`` is a tuple compared with ``>`` (higher is
    better); ties are broken by ascending node id, making the result unique.

    If the frontier runs dry before ``target`` (possible only when ``pool`` is
    disconnected) the remaining slots are filled by global priority order.
    """
    if len(pool) <= target:
        return set(pool)

    # heapq is a min-heap and we want "highest priority first", so negate the
    # rank: store (-priority, node) via a rank index rather than negating the
    # tuple element-wise (elements may be of mixed sign).
    ordered = sorted(pool, key=lambda n: (tuple(-x for x in priority[n]), n))
    rank = {n: r for r, n in enumerate(ordered)}

    start = ordered[0]
    chosen: Set[int] = {start}
    queued: Set[int] = {start}
    frontier: List[Tuple[int, int]] = []
    for w in nb.get(start, ()):
        if w in pool and w not in queued:
            heapq.heappush(frontier, (rank[w], w))
            queued.add(w)

    while frontier and len(chosen) < target:
        _, n = heapq.heappop(frontier)
        chosen.add(n)
        for w in nb.get(n, ()):
            if w in pool and w not in queued:
                heapq.heappush(frontier, (rank[w], w))
                queued.add(w)

    if len(chosen) < target:  # disconnected pool: top up by global priority
        for n in ordered:
            if len(chosen) >= target:
                break
            chosen.add(n)
    return chosen


# ---------------------------------------------------------------------------
# The documented subsampling procedure
# ---------------------------------------------------------------------------


def select_dense_core(arcs: Sequence[Tuple[int, int]], target_users: int,
                      activity: Optional[Dict[int, int]] = None,
                      verbose: bool = False) -> Tuple[Set[int], dict]:
    """Pick ~``target_users`` nodes forming the densest, most-active core of ``arcs``.

    The procedure, in full (deterministic, no rng):

    1. Compute every node's **core number** on the symmetrised graph.
    2. Let ``k*`` be the largest ``k`` such that the k-core has at least
       ``target_users`` nodes.
    3. Take the **largest weakly connected component** of the k*-core.  If it
       has fewer than ``target_users`` nodes, decrement ``k*`` and retry --
       peeling less keeps more of the graph connected.
    4. **Snowball** from the highest-priority node of that component, admitting
       the highest-priority frontier node each step until exactly
       ``target_users`` nodes have been admitted.

    Node priority is ``(core number, adoption activity, degree)``, descending,
    with ascending node id as the final tie-break -- i.e. we keep the users who
    are simultaneously deepest in the core and most active in the adoption log,
    which is exactly the "densest / most-active core" the benchmark wants.

    ``target_users <= 0`` disables subsampling and returns every node.
    Returns ``(selected_nodes, info_dict)``; ``info_dict`` documents the chosen
    ``k*`` and the size after each stage, for meta.json/notes.
    """
    nb = undirected_neighbours(arcs)
    all_nodes = set(nb)
    info: Dict[str, object] = {
        "procedure": "k-core -> largest weakly connected component -> best-first snowball",
        "n_nodes_in": len(all_nodes),
    }

    if target_users <= 0 or len(all_nodes) <= target_users:
        info["k_star"] = 0
        info["n_kcore"] = len(all_nodes)
        info["n_wcc"] = len(all_nodes)
        info["n_selected"] = len(all_nodes)
        info["note"] = "no subsampling needed (graph already at or below target)"
        return all_nodes, info

    core = core_numbers(nb)
    act = activity or {}

    # -- step 2: largest k whose k-core still holds >= target_users nodes ----
    sizes: Dict[int, int] = {}
    for c in core.values():
        sizes[c] = sizes.get(c, 0) + 1
    max_core = max(sizes)
    cumulative = 0
    core_size_at_least: Dict[int, int] = {}
    for k in range(max_core, -1, -1):
        cumulative += sizes.get(k, 0)
        core_size_at_least[k] = cumulative

    k_star = 0
    for k in range(max_core, -1, -1):
        if core_size_at_least[k] >= target_users:
            k_star = k
            break

    # -- step 3: largest WCC of the k*-core, backing off if it is too small --
    kcore_nodes: Set[int] = set()
    wcc: Set[int] = set()
    k = k_star
    while k >= 0:
        kcore_nodes = {n for n, c in core.items() if c >= k}
        wcc = largest_weakly_connected_component(kcore_nodes, nb)
        if verbose:
            print("    k={:<4d} k-core={:<8d} largest WCC={}".format(
                k, len(kcore_nodes), len(wcc)))
        if len(wcc) >= target_users:
            break
        k -= 1
    k_star = max(k, 0)

    # -- step 4: snowball down to exactly target_users ----------------------
    priority = {n: (core.get(n, 0), act.get(n, 0), len(nb[n])) for n in wcc}
    selected = snowball_to_size(wcc, nb, target_users, priority)

    info["k_star"] = k_star
    info["n_kcore"] = len(kcore_nodes)
    info["n_wcc"] = len(wcc)
    info["n_selected"] = len(selected)
    return selected, info


def thin_arcs(arcs: Sequence[Tuple[int, int]], target_links: int,
              weight: Optional[Dict[int, int]] = None,
              required_nodes: Optional[Set[int]] = None) -> List[Tuple[int, int]]:
    """Truncate ``arcs`` to ``target_links``, keeping the most informative ones.

    Deterministic three-pass rule:

    * **Pass 1 (fair quota).**  Every source node keeps its best
      ``q = target_links // n_nodes`` out-arcs.  This guarantees the thinned
      graph is not a hairball around a handful of hubs -- every user that had
      out-arcs keeps some, so the out-degree distribution stays broad and the
      diffusion model still has paths to work with.
    * **Pass 2 (global fill).**  The leftover budget is spent on the globally
      best remaining arcs.
    * **Pass 3 (strand repair).**  A node of ``required_nodes`` left with no
      incident arc would have to be dropped from ``U`` entirely, putting the
      produced user count below the paper's target.  Each such node gets its
      best incident arc back, paid for by evicting the worst-scoring kept arc
      that is not itself somebody's last remaining connection.  The link budget
      is therefore still met exactly.

    An arc's score is ``(weight[u] + weight[v], deg[u] + deg[v])`` descending,
    tie-broken by ``(u, v)`` ascending -- arcs between active, well-connected
    users are preferred.  ``weight`` defaults to all-zero, which reduces the
    score to pure degree.

    ``target_links <= 0`` or a graph already below budget returns the arcs
    unchanged (sorted).
    """
    arcs = [(u, v) for u, v in arcs]
    if target_links <= 0 or len(arcs) <= target_links:
        return sorted(arcs)

    w = weight or {}
    deg: Dict[int, int] = {}
    for u, v in arcs:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1

    def score(arc: Tuple[int, int]) -> tuple:
        u, v = arc
        return (-(w.get(u, 0) + w.get(v, 0)), -(deg[u] + deg[v]), u, v)

    by_source: Dict[int, List[Tuple[int, int]]] = {}
    for arc in arcs:
        by_source.setdefault(arc[0], []).append(arc)

    n_nodes = len(deg)
    quota = max(1, target_links // max(1, n_nodes))

    kept: List[Tuple[int, int]] = []
    leftover: List[Tuple[int, int]] = []
    for u in sorted(by_source):
        group = sorted(by_source[u], key=score)
        kept.extend(group[:quota])
        leftover.extend(group[quota:])

    if len(kept) > target_links:  # only if quota*|sources| overshoots
        kept.sort(key=score)
        kept = kept[:target_links]
    elif len(kept) < target_links:
        leftover.sort(key=score)
        kept.extend(leftover[:target_links - len(kept)])

    if required_nodes:
        kept = _repair_stranded(arcs, kept, required_nodes, target_links, score)

    return sorted(kept)


def _repair_stranded(arcs: Sequence[Tuple[int, int]],
                     kept: List[Tuple[int, int]],
                     required_nodes: Set[int], target_links: int,
                     score) -> List[Tuple[int, int]]:
    """Pass 3 of :func:`thin_arcs`: give every required node an incident arc.

    Adds each stranded node's best incident arc and evicts an equal number of
    the worst-scoring kept arcs, never evicting one that is some node's only
    remaining connection.  Deterministic throughout.
    """
    kept_set = set(kept)
    incident: Dict[int, int] = {}
    for u, v in kept_set:
        incident[u] = incident.get(u, 0) + 1
        incident[v] = incident.get(v, 0) + 1

    stranded = sorted(n for n in required_nodes if incident.get(n, 0) == 0)
    if not stranded:
        return kept

    # best incident arc per stranded node, over ALL arcs (not just the leftovers)
    best: Dict[int, Tuple[int, int]] = {}
    for arc in arcs:
        for n in arc:
            if n in incident and incident[n] > 0:
                continue
            if n not in required_nodes:
                continue
            cur = best.get(n)
            if cur is None or score(arc) < score(cur):
                best[n] = arc

    added: Set[Tuple[int, int]] = set()
    for n in stranded:
        arc = best.get(n)
        if arc is None or arc in kept_set:
            continue
        kept_set.add(arc)
        added.add(arc)
        for x in arc:
            incident[x] = incident.get(x, 0) + 1

    # pay for the additions by evicting the worst arcs that nobody depends on
    if len(kept_set) > target_links:
        evictable = sorted((a for a in kept_set if a not in added),
                           key=score, reverse=True)  # worst first
        for arc in evictable:
            if len(kept_set) <= target_links:
                break
            u, v = arc
            if incident[u] <= 1 or incident[v] <= 1:
                continue  # somebody's last connection: keep it
            kept_set.discard(arc)
            incident[u] -= 1
            incident[v] -= 1

    return sorted(kept_set)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    # -- undirected_neighbours ---------------------------------------------
    nb = undirected_neighbours([(0, 1), (1, 2), (2, 0), (3, 3)])
    assert nb == {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}, nb   # self-loop ignored

    # -- core_numbers -------------------------------------------------------
    # a triangle (core 2) with a pendant (core 1)
    nb2 = undirected_neighbours([(0, 1), (1, 2), (2, 0), (2, 3)])
    core = core_numbers(nb2)
    assert core == {0: 2, 1: 2, 2: 2, 3: 1}, core
    # K4 has core number 3 everywhere
    k4 = undirected_neighbours([(a, b) for a in range(4) for b in range(4) if a != b])
    assert set(core_numbers(k4).values()) == {3}
    # a path has core number 1 everywhere
    path = undirected_neighbours([(i, i + 1) for i in range(5)])
    assert set(core_numbers(path).values()) == {1}

    # -- largest_weakly_connected_component ---------------------------------
    arcs = [(0, 1), (1, 2), (2, 0), (5, 6)]
    nb3 = undirected_neighbours(arcs)
    assert largest_weakly_connected_component(set(nb3), nb3) == {0, 1, 2}
    # direction must not matter (weak connectivity)
    nb4 = undirected_neighbours([(1, 0), (2, 1)])
    assert largest_weakly_connected_component({0, 1, 2}, nb4) == {0, 1, 2}

    # -- snowball_to_size ---------------------------------------------------
    # star: node 0 at the centre, priority favours the centre
    star = undirected_neighbours([(0, k) for k in range(1, 10)])
    pri = {n: (10 - n,) for n in star}
    sel = snowball_to_size(set(star), star, 4, pri)
    assert len(sel) == 4 and 0 in sel, sel
    assert sel == {0, 1, 2, 3}, sel            # highest priority neighbours first
    # deterministic
    assert snowball_to_size(set(star), star, 4, pri) == sel
    # target >= pool is a no-op
    assert snowball_to_size(set(star), star, 99, pri) == set(star)
    # a disconnected pool still reaches the target via the global top-up
    disc = undirected_neighbours([(0, 1), (5, 6), (7, 8)])
    pri_d = {n: (0,) for n in disc}
    assert len(snowball_to_size(set(disc), disc, 5, pri_d)) == 5

    # -- select_dense_core --------------------------------------------------
    # triangle (dense) + a long sparse tail; asking for 3 must return the triangle
    dense = [(0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)]
    tail = [(10 + i, 11 + i) for i in range(8)]
    sel, info = select_dense_core(dense + tail, 3)
    assert sel == {0, 1, 2}, (sel, info)
    assert info["k_star"] == 2 and info["n_selected"] == 3, info
    # target <= 0 disables subsampling
    everything, info0 = select_dense_core(dense + tail, 0)
    assert everything == {0, 1, 2} | set(range(10, 19)), everything
    assert info0["k_star"] == 0, info0
    # deterministic across calls
    assert select_dense_core(dense + tail, 3)[0] == sel

    # -- thin_arcs ----------------------------------------------------------
    many = [(u, v) for u in range(10) for v in range(10) if u != v]
    assert len(many) == 90
    thin = thin_arcs(many, 20)
    assert len(thin) == 20 and thin == sorted(thin)
    assert len(set(thin)) == 20
    assert set(thin) <= set(many)
    # every source keeps its quota (20 // 10 = 2 arcs each)
    from collections import Counter
    per_source = Counter(u for u, _v in thin)
    assert set(per_source.values()) == {2}, per_source
    # deterministic
    assert thin_arcs(many, 20) == thin
    # below budget -> unchanged
    assert thin_arcs(many, 500) == sorted(many)
    assert thin_arcs(many, 0) == sorted(many)
    # weight steers the choice toward heavy nodes
    heavy = {9: 1000, 8: 1000}
    thin_w = thin_arcs(many, 20, weight=heavy)
    assert sum(1 for a in thin_w if 9 in a or 8 in a) > \
        sum(1 for a in thin if 9 in a or 8 in a)

    # -- thin_arcs strand repair -------------------------------------------
    # a hub clique plus pendants that only pass 1 has room to keep
    hub = [(u, v) for u in range(6) for v in range(6) if u != v]     # 30 arcs
    pend = [(100 + k, 0) for k in range(5)]                          # 5 pendants
    all_arcs = hub + pend
    need = {x for a in all_arcs for x in a}
    plain = thin_arcs(all_arcs, 12)
    repaired = thin_arcs(all_arcs, 12, required_nodes=need)
    assert len(repaired) == 12, len(repaired)
    covered_plain = {x for a in plain for x in a}
    covered_rep = {x for a in repaired for x in a}
    assert covered_rep >= covered_plain
    assert covered_rep == need, sorted(need - covered_rep)
    assert repaired == thin_arcs(all_arcs, 12, required_nodes=need)   # deterministic
    assert set(repaired) <= set(all_arcs)

    print("ctim/datasets/subsample.py self-test: OK")


if __name__ == "__main__":
    _self_test()
