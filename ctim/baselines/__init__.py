"""Baseline influence-maximization methods (SPEC.md Section 7, Table 2).

The baselines the paper compares CTIM against:

============  ================  ============  ==========================
module        entry point       community?    topic-aware?
============  ================  ============  ==========================
``greedy``    greedy_select     no            no
``cga``       cga_select_seeds  yes           no
``cinema``    cinema_select_..  yes           no
``air_cga``   air_cga_select_.. yes           yes
============  ================  ============  ==========================

Every baseline returns a :class:`RunResult`.

Submodules are imported **lazily** (PEP 562 ``__getattr__``) so that importing
``ctim.baselines`` stays free and, importantly, so that a not-yet-written
sibling baseline does not break the ones that already exist.

Standard library only.  Python 3.9 compatible.
"""

from __future__ import annotations

import importlib

__all__ = [
    "RunResult",
    "greedy_select",
    "ic_simulate",
    "topic_averaged_pp",
    "cga_select_seeds",
    "ctim_cga_select",
    "cinema_select_seeds",
    "air_cga_select_seeds",
]


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------
#
# API.md puts the canonical `RunResult` in `ctim/ctim.py`.  That module may not
# exist yet (the baselines are written independently of it), so we import it
# when we can and fall back to a structurally identical dataclass otherwise.
# Consumers must duck-type on the four fields rather than isinstance-check:
# the two classes are field-for-field identical but not the same object if this
# fallback is ever taken.
try:  # pragma: no cover - depends on sibling module availability
    from ctim.ctim import RunResult  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - ctim.ctim not written yet
    from dataclasses import dataclass, field

    @dataclass
    class RunResult:  # type: ignore[no-redef]
        """Result of one seed-selection run (API.md, `ctim/ctim.py`).

        seeds    selected seed nodes, in selection order
        spread   influence spread of `seeds` as measured by the method itself
        seconds  wall-clock seconds spent selecting the seeds
        extra    free-form per-method diagnostics
        """

        seeds: list
        spread: float
        seconds: float
        extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Lazy submodule / symbol access
# ---------------------------------------------------------------------------

_SUBMODULES = ("greedy", "cga", "cinema", "air_cga")

# public symbol -> submodule that defines it
_EXPORTS = {
    "greedy_select": "greedy",
    "ic_simulate": "greedy",
    "topic_averaged_pp": "greedy",
    "cga_select_seeds": "cga",
    "ctim_cga_select": "cga",
    "cinema_select_seeds": "cinema",
    "air_cga_select_seeds": "air_cga",
}


def __getattr__(name):
    """PEP 562 lazy attribute access for submodules and their entry points."""
    if name in _SUBMODULES:
        return importlib.import_module("." + name, __name__)
    modname = _EXPORTS.get(name)
    if modname is not None:
        mod = importlib.import_module("." + modname, __name__)
        return getattr(mod, name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


def __dir__():
    return sorted(set(list(globals().keys()) + list(__all__) + list(_SUBMODULES)))
