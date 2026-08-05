#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script to run CTIM and CTIM-EA with:
- K = 20
- Population size = 20
- Generations = 50
- Mode: quick (using quick config settings to run fast)
- Dataset: data/processed/digg
"""

import os
import sys
import time
import random

# Ensure workspace root is in sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ctim.dataset import load_dataset, build_potential_influence_logs, split_logs
from ctim.gibbs import train_model
from ctim.influence import EdgeWeights, MIA
from ctim.ctim import ctim_run
from ctim.citm_ea import run_ea

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run CTIM vs CTIM-EA comparison.")
    parser.add_argument("--dataset", type=str, default="data/processed/digg",
                        help="Path to the dataset directory (default: data/processed/digg)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 72)
    print(f"Running CTIM vs CTIM-EA comparison on {args.dataset}")
    print("Parameters: K=20, PopSize=20, Generations=100, 200x200 Gibbs sweeps")
    print("=" * 72)

    dataset_path = args.dataset
    seed = args.seed
    rng = random.Random(seed)

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    ds = load_dataset(dataset_path)
    print(f"Dataset loaded: {ds.name}")
    print(f"Users: {ds.n_users}, Links: {ds.n_links}, Items: {ds.n_items}")

    # 2. Build logs and split (using quick mode settings: cap logs per item to 20)
    print("\n[2/5] Building potential-influence logs (Quick mode cap: 20 per item)...")
    delta = 30 * 24 * 3600
    logs_d = build_potential_influence_logs(ds, delta, max_per_item=20, rng=rng)
    print(f"Total potential influence logs: {len(logs_d)}")
    
    train, valid, test = split_logs(logs_d, rng)
    print(f"Split logs -> Train: {len(train)}, Valid: {len(valid)}, Test: {len(test)}")

    # Pick the top test items
    counts = {}
    for (_u, _v, i) in test:
        counts[i] = counts.get(i, 0) + 1
    if not counts:
        for (_u, i, _t) in ds.logs:
            counts[i] = counts.get(i, 0) + 1
    ranked = sorted(counts.keys(), key=lambda i: (-counts[i], i))
    test_items = ranked[:3] # evaluate on 3 items like quick config does
    print(f"Evaluating on items: {test_items}")

    # 3. Train the reference Gibbs model (using 200 iterations for both stages)
    C = 100
    Z = 8
    
    print(f"\n[3/5] Training Gibbs model (C={C}, Z={Z}, 200 iters)...")
    t_train = time.perf_counter()
    model = train_model(ds, train, C, Z, n_iter_topic=200, n_iter_comm=200, rng=rng, sampler="mh")
    train_time = time.perf_counter() - t_train
    print(f"Model training completed in {train_time:.2f}s")

    ew = EdgeWeights(model, ds)

    # 4. Run CTIM
    print("\n[4/5] Running CTIM...")
    ctim_spreads = []
    ctim_times = []
    
    for item in test_items:
        t0 = time.perf_counter()
        res = ctim_run(model, ds, item, K=20, h=0.1, edge_weights=ew)
        ctim_times.append(time.perf_counter() - t0)
        ctim_spreads.append(res.spread)
        print(f"  Item {item} - Spread: {res.spread:.2f}, Time: {ctim_times[-1]:.4f}s")

    avg_ctim_spread = sum(ctim_spreads) / len(ctim_spreads)
    avg_ctim_time = sum(ctim_times) / len(ctim_times)

    # 5. Run CTIM-EA (EA)
    print("\n[5/5] Running CTIM-EA...")
    ea_spreads = []
    ea_times = []
    
    for item in test_items:
        t0 = time.perf_counter()
        # EA has pop_size=20 and num_generations=100
        res_ea = run_ea(model, ds, item, K=20, pop_size=20, num_generations=100, h=0.1, verbose=False, rng=rng)
        ea_times.append(time.perf_counter() - t0)
        ea_spreads.append(res_ea['fitness'])
        print(f"  Item {item} - Spread: {res_ea['fitness']:.2f}, Time: {ea_times[-1]:.4f}s")

    avg_ea_spread = sum(ea_spreads) / len(ea_spreads)
    avg_ea_time = sum(ea_times) / len(ea_times)

    print("\n" + "=" * 72)
    print("RESULTS SUMMARY (Averaged over evaluated items)")
    print("=" * 72)
    print(f"{'Method':<15} | {'Average Spread (Influence)':<26} | {'Average Time (seconds)':<22}")
    print("-" * 72)
    print(f"{'CTIM':<15} | {avg_ctim_spread:<26.2f} | {avg_ctim_time:<22.4f}")
    print(f"{'CTIM-EA':<15} | {avg_ea_spread:<26.2f} | {avg_ea_time:<22.4f}")
    print("=" * 72)

if __name__ == "__main__":
    main()
