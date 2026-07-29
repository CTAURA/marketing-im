"""CTIM — Community-based Topic-aware Influence Maximization.

Reference implementation of

    Huimin Huang, Hong Shen, Zaiqiao Meng, Huajian Chang, Huaiwen He.
    "Community-based influence maximization for viral marketing",
    Applied Intelligence (2019). DOI 10.1007/s10489-018-1387-8

Standard library only, Python 3.9 compatible.

This package root deliberately imports nothing: submodules are heavy enough
that importing `ctim` should stay free.  Import what you need explicitly::

    from ctim.dataset import Dataset, load_dataset, build_potential_influence_logs
    from ctim.gibbs import train_model
    from ctim.influence import EdgeWeights, MIA
    from ctim.ctim import ctim_select_seeds
"""

__version__ = "1.0.0"

__all__ = []  # nothing re-exported on purpose; import submodules directly
