"""Evolutionary Algorithm for Community-based Topic-aware Influence Maximization (CTIM-EA).

This module implements the initialization phase of the evolutionary algorithm (EA)
based on community relevance and node influence, using 20 individuals and 100 generations.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from ctim.ctim import detect_communities
from ctim.influence import EdgeWeights, MIA


def initialize_population(
    model: Any,
    ds: Any,
    item: int,
    K: int,
    pop_size: int = 20,
    h: float = 0.1,
    top_influential_per_c: int = 1,
    rng: random.Random | None = None
) -> List[List[int]]:
    """
    Initializes the population of size `pop_size` using degree-based initialization.
    For each individual:
      1. Initialize with the top K nodes of the highest out-degrees.
      2. For each node in the individual, with 50% probability (random > 0.5),
         replace it with a random node from U (0 to ds.n_users - 1) not in the individual.
    """
    if rng is None:
        rng = random.Random()

    # Calculate out-degrees for all users
    degrees = []
    for u in range(ds.n_users):
        deg = len(ds.out_adj[u]) if ds.out_adj[u] is not None else 0
        degrees.append((u, deg))

    # Sort users by degree descending
    degrees.sort(key=lambda x: (-x[1], x[0]))
    top_k_nodes = [u for u, _ in degrees[:K]]

    population = []
    for _ in range(pop_size):
        individual = list(top_k_nodes)
        seen = set(individual)
        for j in range(K):
            if rng.random() > 0.5:
                old_node = individual[j]
                seen.remove(old_node)
                # Find a random node from U (0 to n_users - 1) not in seen
                while True:
                    candidate = rng.randrange(ds.n_users)
                    if candidate not in seen:
                        individual[j] = candidate
                        seen.add(candidate)
                        break
        population.append(individual)

    return population


def crossover_population(
    population_with_fitness: List[Dict[str, Any]],
    ds: Any,
    mia: MIA,
    p_c: float = 0.8,
    rng: random.Random | None = None
) -> List[Dict[str, Any]]:
    """
    Performs Uniform Crossover on the population according to the pairing:
    - 1st (best) with last (worst)
    - 2nd with second-to-last
    - ...
    
    For duplicate nodes in offspring, new nodes are randomly selected from the 
    entire user set U (0 to ds.n_users - 1) instead of Pool[c].

    Parameters
    ----------
    population_with_fitness : List[Dict[str, Any]]
        List of dicts representing individuals, e.g., [{'seeds': [...], 'fitness': float}, ...]
    ds : Any
        The dataset object (needs ds.n_users).
    mia : MIA
        The MIA model for evaluating fitness I(S).
    p_c : float
        The crossover probability (default: 0.8).
    rng : random.Random or None
        Random number generator.

    Returns
    -------
    List[Dict[str, Any]]
        The new population of offspring with their evaluated fitness.
    """
    if rng is None:
        rng = random.Random()

    # Sort population by fitness descending
    sorted_pop = sorted(population_with_fitness, key=lambda x: x['fitness'], reverse=True)
    n_pop = len(sorted_pop)
    
    offspring_pop = []
    
    # Pair individuals: i-th with (n_pop - 1 - i)-th
    for i in range(n_pop // 2):
        parent1 = sorted_pop[i]
        parent2 = sorted_pop[n_pop - 1 - i]
        
        # Clone seed lists for crossover
        child1_seeds = list(parent1['seeds'])
        child2_seeds = list(parent2['seeds'])
        
        K = len(child1_seeds)
        
        # Uniform Crossover: check each gene position
        for t in range(K):
            if rng.random() < p_c:
                # Swap genes/nodes at position t
                child1_seeds[t], child2_seeds[t] = child2_seeds[t], child1_seeds[t]
                
        # Resolve duplicates for child 1 by sampling from set U
        child1_unique = []
        seen1 = set()
        for u in child1_seeds:
            if u not in seen1:
                child1_unique.append(u)
                seen1.add(u)
        
        while len(child1_unique) < K:
            # Select random node from U (0 to n_users-1)
            candidate = rng.randrange(ds.n_users)
            if candidate not in seen1:
                child1_unique.append(candidate)
                seen1.add(candidate)
                
        # Resolve duplicates for child 2 by sampling from set U
        child2_unique = []
        seen2 = set()
        for u in child2_seeds:
            if u not in seen2:
                child2_unique.append(u)
                seen2.add(u)
                
        while len(child2_unique) < K:
            # Select random node from U (0 to n_users-1)
            candidate = rng.randrange(ds.n_users)
            if candidate not in seen2:
                child2_unique.append(candidate)
                seen2.add(candidate)
                
        # Calculate new fitness I(S)
        fit1 = mia.influence(child1_unique)
        fit2 = mia.influence(child2_unique)
        
        offspring_pop.append({'seeds': child1_unique, 'fitness': fit1})
        offspring_pop.append({'seeds': child2_unique, 'fitness': fit2})
        
    # If odd population size, pass the middle one unchanged
    if n_pop % 2 != 0:
        offspring_pop.append(sorted_pop[n_pop // 2])
        
    return offspring_pop


def mutate_population(
    population_with_fitness: List[Dict[str, Any]],
    model: Any,
    ds: Any,
    mia: MIA,
    p_m: float = 0.1,
    rng: random.Random | None = None
) -> List[Dict[str, Any]]:
    """
    Performs Mutation on the population.
    For each node in an individual, mutate it with probability p_m.
    A mutated node is replaced by another node belonging to the same community 
    chosen randomly from the entire user set U (all nodes in community), 
    excluding nodes already in the individual's seed set.
    
    If the new fitness is strictly greater than the old fitness, the mutation is accepted.
    Otherwise, the individual reverts to its original seed set and fitness.

    Parameters
    ----------
    population_with_fitness : List[Dict[str, Any]]
        List of dicts representing individuals, e.g., [{'seeds': [...], 'fitness': float}, ...]
    model : Any
        The trained Gibbs model (used to detect user communities).
    ds : Any
        The dataset object (needs ds.n_users, ds.out_adj, ds.in_adj).
    mia : MIA
        The MIA model for evaluating fitness I(S).
    p_m : float
        The mutation probability (default: 0.1).
    rng : random.Random or None
        Random number generator.

    Returns
    -------
    List[Dict[str, Any]]
        The mutated population with updated fitness values.
    """
    if rng is None:
        rng = random.Random()

    # Detect communities for all users
    user_communities = detect_communities(model.pi)
    
    # Group all users in U by community
    members_by_c = defaultdict(list)
    for u, c in enumerate(user_communities):
        members_by_c[c].append(u)

    mutated_pop = []

    for ind in population_with_fitness:
        original_seeds = list(ind['seeds'])
        original_fit = ind['fitness']
        
        # Working copy for mutation
        current_seeds = list(original_seeds)
        current_set = set(current_seeds)
        
        mutated = False
        K = len(current_seeds)
        
        for j in range(K):
            if rng.random() < p_m:
                old_node = current_seeds[j]
                comm = user_communities[old_node]
                
                # Find all nodes in the same community (from U) not in current_set
                candidates = [v for v in members_by_c[comm] if v not in current_set]
                
                if candidates:
                    new_node = rng.choice(candidates)
                    current_seeds[j] = new_node
                    current_set.remove(old_node)
                    current_set.add(new_node)
                    mutated = True
                    
        if mutated:
            # Calculate new fitness
            new_fit = mia.influence(current_seeds)
            if new_fit > original_fit:
                # Accept mutation
                mutated_pop.append({'seeds': current_seeds, 'fitness': new_fit})
            else:
                # Revert to original
                mutated_pop.append({'seeds': original_seeds, 'fitness': original_fit})
        else:
            # No mutation happened
            mutated_pop.append({'seeds': original_seeds, 'fitness': original_fit})
            
    return mutated_pop

def local_search_population(
    population_with_fitness: List[Dict[str, Any]],
    model: Any,
    ds: Any,
    mia: MIA,
    local_search_rate: float = 0.1,
    max_passes: int = 1,
    rng: random.Random | None = None
) -> List[Dict[str, Any]]:
    """
    Applies Local Search on the top portion of the population (Algorithm 3).
    For each selected individual, it iterates through each seed node and attempts
    to replace it with a random node from its outgoing neighborhood (N^(1)).
    If fitness improves, the change is accepted and search continues on other neighbors.
    Otherwise, the search for this seed node stops.
    """
    if rng is None:
        rng = random.Random()

    # Sort population by fitness descending
    sorted_pop = sorted(population_with_fitness, key=lambda x: x['fitness'], reverse=True)
    n_pop = len(sorted_pop)
    
    # Identify top portion of individuals to apply local search
    n_selected = max(1, int(round(local_search_rate * n_pop)))
    
    for i in range(n_selected):
        ind = sorted_pop[i]
        X_b = list(ind['seeds'])
        X_a = list(X_b)
        K = len(X_b)
        
        for idx in range(K):
            flag = False
            x_bi = X_b[idx]
            
            # Outgoing neighbors of x_bi
            neighbors = ds.out_adj[x_bi] if x_bi < len(ds.out_adj) and ds.out_adj[x_bi] is not None else []
            if not neighbors:
                continue
            
            neighbors_list = list(neighbors)
            
            while not flag:
                current_set = set(X_b)
                candidates = [v for v in neighbors_list if v not in current_set]
                
                if not candidates:
                    flag = True
                    break
                
                # Replace x_bi with a random neighbor
                new_node = rng.choice(candidates)
                X_b[idx] = new_node
                
                # Compare fitness
                fit_b = mia.influence(X_b)
                fit_a = mia.influence(X_a)
                
                if fit_b > fit_a:
                    X_a = list(X_b)
                else:
                    flag = True
            
            X_b = list(X_a)
            
        ind['seeds'] = X_a
        ind['fitness'] = mia.influence(X_a)

    return sorted_pop


def run_ea(
    model: Any,
    ds: Any,
    item: int,
    K: int,
    pop_size: int = 20,
    num_generations: int = 100,
    p_c: float = 0.8,
    p_m: float = 0.1,
    h: float = 0.1,
    local_search_freq: int = 10,
    local_search_rate: float = 0.1,
    verbose: bool = False,
    rng: random.Random | None = None
) -> Dict[str, Any]:
    """
    Runs the complete Evolutionary Algorithm for Community-based Topic-aware
    Influence Maximization (CTIM-EA).

    Parameters
    ----------
    model : Any
        The trained Gibbs model.
    ds : Any
        The dataset.
    item : int
        The target item.
    K : int
        The number of seeds to select.
    pop_size : int
        The size of the population (default: 20).
    num_generations : int
        The number of generations to evolve (default: 100).
    p_c : float
        Crossover probability.
    p_m : float
        Mutation probability.
    h : float
        MIA threshold.
    local_search_freq : int
        Local search frequency in generations (default: 10).
    local_search_rate : float
        Ratio of individuals undergoing local search (default: 0.1).
    verbose : bool
        If True, prints progress statistics per generation.
    rng : random.Random or None
        Random number generator.

    Returns
    -------
    Dict[str, Any]
        The best individual found represented as: {'seeds': [...], 'fitness': float}
    """
    if rng is None:
        rng = random.Random()

    # Step 1-6: Initialize the seed sets of the population
    if verbose:
        print(f"Initializing population of size {pop_size}...")
    initial_pop_seeds = initialize_population(
        model, ds, item, K, pop_size=pop_size, h=h, rng=rng
    )

    # Compute fitness of initial population
    ew = EdgeWeights(model, ds)
    pp = ew.for_item(item)
    mia = MIA(ds.n_users, ds.out_adj, ds.in_adj, pp, h=h)

    # Cache mia.influence evaluations to speed up crossover/mutation/local search
    original_influence = mia.influence
    influence_cache = {}
    def cached_influence(seeds):
        key = tuple(sorted(seeds))
        if key not in influence_cache:
            influence_cache[key] = original_influence(seeds)
        return influence_cache[key]
    mia.influence = cached_influence

    population = []
    for seeds in initial_pop_seeds:
        fit = mia.influence(seeds)
        population.append({'seeds': seeds, 'fitness': fit})

    # Keep track of best individual
    population.sort(key=lambda x: x['fitness'], reverse=True)
    best_individual = {
        'seeds': list(population[0]['seeds']),
        'fitness': population[0]['fitness']
    }

    if verbose:
        print(f"Initial population best fitness: {best_individual['fitness']:.4f}")

    # Evolution loop
    for gen in range(1, num_generations + 1):
        # 1. Crossover
        offspring = crossover_population(population, ds, mia, p_c=p_c, rng=rng)
        
        # 2. Mutation
        offspring = mutate_population(offspring, model, ds, mia, p_m=p_m, rng=rng)
        
        # Combine parents and offspring
        combined = population + offspring
        
        # 3. Local Search (every local_search_freq generations)
        if gen % local_search_freq == 0:
            if verbose:
                print(f"[Gen {gen}] Applying Local Search to top {local_search_rate*100:.0f}%...")
            combined = local_search_population(
                combined, model, ds, mia, 
                local_search_rate=local_search_rate, rng=rng
            )
            
        # 4. Selection (survival of the fittest: keep top pop_size)
        combined.sort(key=lambda x: x['fitness'], reverse=True)
        population = combined[:pop_size]
        
        # Update overall best
        if population[0]['fitness'] > best_individual['fitness']:
            best_individual = {
                'seeds': list(population[0]['seeds']),
                'fitness': population[0]['fitness']
            }

        if verbose and (gen % 10 == 0 or gen == 1):
            print(f"Generation {gen}/{num_generations} - Best Fitness: {population[0]['fitness']:.4f}")

    return best_individual



