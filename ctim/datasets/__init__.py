"""Benchmark dataset preparation for the CTIM reference implementation.

Turns raw third-party downloads into the canonical processed layout of API.md
(``graph.tsv`` / ``logs.tsv`` / ``items.tsv`` / ``meta.json``) so that
``ctim.dataset.load_dataset`` can read them.

Three sources:

``digg``       the real Digg 2009 friendship graph + story votes (SPEC.md
               Section 8 benchmark #2).
``yelp``       the official Yelp open dataset (benchmark #1).  The paper used
               the *Yelp Dataset Challenge 2014* snapshot, which Yelp has
               retired and no longer distributes, so absolute numbers cannot
               match exactly; see :mod:`ctim.datasets.yelp`.
``synthetic``  a planted community/topic generator, used for unit tests, the
               ``--quick`` experiment profile, and as an explicitly-labelled
               Yelp-scaled stand-in.

Every preparer is deterministic given its seed, and every one writes a
``meta.json`` whose ``notes`` field states exactly how the data was derived --
in particular whether the adoption logs are real or simulated.

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PAPER_TARGETS",
    "paper_comparison_table",
    "prepare_digg",
    "prepare_yelp",
    "prepare_synthetic",
    "generate_synthetic",
]

# SPEC.md Section 8 -- "Datasets" table, verbatim.
# name -> (users, links, items)
PAPER_TARGETS: Dict[str, Tuple[int, int, int]] = {
    "yelp": (366715, 2949285, 61184),
    "digg": (30358, 99846, 7100),
}

PAPER_SOURCE_LABEL: Dict[str, str] = {
    "yelp": "Yelp Dataset Challenge 2014 (SPEC.md Section 8)",
    "digg": "Digg (SPEC.md Section 8)",
}


def paper_comparison_table(produced: Dict[str, int],
                           benchmark: Optional[str] = None,
                           title: str = "") -> str:
    """ASCII table putting produced counts NEXT TO the paper's reported counts.

    ``produced`` needs the keys ``n_users``/``n_links``/``n_items`` (plus the
    optional ``n_attrs``/``n_logs``, which the paper does not report).
    ``benchmark`` selects the row of SPEC.md Section 8 to compare against; pass
    ``None`` for a dataset the paper never reported (e.g. synthetic), in which
    case the paper columns read ``n/a``.
    """
    target = PAPER_TARGETS.get(benchmark or "", None)
    rows: List[Tuple[str, str, str, str]] = []

    def add(label: str, got: object, want: Optional[int]) -> None:
        if want is None:
            rows.append((label, "{:,}".format(got) if isinstance(got, int) else str(got),
                         "n/a", "n/a"))
            return
        ratio = (float(got) / want * 100.0) if want else 0.0
        rows.append((label, "{:,}".format(int(got)), "{:,}".format(want),
                     "{:6.1f}%".format(ratio)))

    add("users", produced.get("n_users", 0), target[0] if target else None)
    add("links (directed arcs)", produced.get("n_links", 0), target[1] if target else None)
    add("items", produced.get("n_items", 0), target[2] if target else None)
    add("attributes (F)", produced.get("n_attrs", 0), None)
    add("adoption records", produced.get("n_logs", 0), None)

    header = ("quantity", "produced", "paper", "of paper")
    widths = [max(len(header[c]), max(len(r[c]) for r in rows)) for c in range(4)]

    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(
            cells[c].ljust(widths[c]) if c == 0 else cells[c].rjust(widths[c])
            for c in range(4)
        )

    rule = "  ".join("-" * widths[c] for c in range(4))
    lines: List[str] = []
    if title:
        lines.append(title)
    if benchmark:
        lines.append("paper reference: " + PAPER_SOURCE_LABEL.get(benchmark, benchmark))
    lines.append(fmt(header))
    lines.append(rule)
    for r in rows:
        lines.append(fmt(r))
    return "\n".join(lines)


# Imported last: the submodules import PAPER_TARGETS from this module.
from .digg import prepare_digg            # noqa: E402
from .yelp import prepare_yelp            # noqa: E402
from .synthetic import (                  # noqa: E402
    generate_synthetic,
    prepare_synthetic,
)
