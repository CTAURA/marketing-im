"""CTIM-EA -- Community-based Topic-aware Influence Maximization with Evolutionary Algorithm.

Replaces the Dynamic Programming (DP) part of Algorithm 2 with an Evolutionary Algorithm (EA).
Highly optimized version.
"""

from __future__ import annotations

import math
import random

try:
    from .ctim import detect_communities
except ImportError:
    # Support running directly
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from ctim.ctim import detect_communities


def initialize_l_comm(model, pi, K) -> list:
    """Helper to select top communities by size/membership and gather ALL their member nodes.
    
    Fix: Use min(K, C) to avoid Index/Range error when K > C (number of communities).
    """
    C = len(model.eta)
    comm = detect_communities(pi)
    members_by_m = [[] for _ in range(C)]
    for v, c in enumerate(comm):
        if 0 <= c < C:
            members_by_m[c].append(v)
            
    # Sort communities by size descending
    sorted_comms = sorted(range(C), key=lambda c: len(members_by_m[c]), reverse=True)
    # Take top min(K, C) communities
    top_comms = sorted_comms[:min(K, C)]
    l_comm = []
    for c in top_comms:
        l_comm.extend(members_by_m[c])
    return sorted(list(set(l_comm)))


class CTIM_EA2:
    """CTIM-EA Evolutionary Algorithm for seed selection."""

    def __init__(self, G, K, pop, g_max=150, Pc=0.6, Pm=0.1, L_comm=None, L_MIA=None, L_MIOA=None, mia=None, seed=42):
        self.G = G
        self.K = K
        self.pop_size = pop
        self.g_max = g_max
        self.Pc = Pc
        self.Pm = Pm
        self.L_comm = list(L_comm) if L_comm else []
        self.L_MIA = list(L_MIA) if L_MIA else []
        self.L_MIOA = list(L_MIOA) if L_MIOA else []
        self.mia = mia
        self.rng = random.Random(seed)

        # Retrieve the set of all users U
        if hasattr(G, "nodes"):
            self.U = list(G.nodes)
        elif hasattr(G, "n_users"):
            self.U = list(range(G.n_users))
        else:
            raise ValueError("G must be a NetworkX Graph or a Dataset object with n_users/nodes attribute")

        # Optimization 1: Pre-initialize sets for C-level fast Set Difference operations
        self.U_set = set(self.U)
        self.L_top_set = set(self.L_comm) | set(self.L_MIA) | set(self.L_MIOA)
        if not self.L_top_set:
            self.L_top_set = self.U_set.copy()

        self.L_top = list(self.L_top_set)
        
        # Pre-cache set versions of input pools to avoid repeated conversion in sample_unique
        self.L_MIA_set = set(self.L_MIA)
        self.L_comm_set = set(self.L_comm)
        self.L_MIOA_set = set(self.L_MIOA)
        
        # Optimization 4: Fitness Evaluation Cache (Memoization)
        # Avoids recalculating influence for the same seed set (represented as a sorted tuple)
        self.fitness_cache = {}

    def sample_unique_from_set(self, pool_set, count, exclude) -> list:
        """Sample `count` unique nodes from `pool_set` that are not in `exclude` using Set Difference."""
        available = list(pool_set - exclude)
        if len(available) < count:
            fallback_pool = list(self.U_set - exclude - set(available))
            if len(fallback_pool) < count - len(available):
                return available + fallback_pool
            return available + self.rng.sample(fallback_pool, count - len(available))
        return self.rng.sample(available, count)

    def create_individual(self) -> list:
        """Create a single individual with exactly K distinct nodes using the 40-30-20-10 rule."""
        ind_set = set()

        K1 = math.ceil(0.4 * self.K)
        K2 = math.ceil(0.3 * self.K)
        K3 = math.ceil(0.2 * self.K)
        
        # Adjust group sizes so that their sum does not exceed self.K
        while K1 + K2 + K3 > self.K:
            if K3 > 0:
                K3 -= 1
            elif K2 > 0:
                K2 -= 1
            elif K1 > 0:
                K1 -= 1
                
        K4 = self.K - (K1 + K2 + K3)
        if K4 < 0:
            K4 = 0

        # Sample from each group using pre-initialized sets for efficiency
        g1 = self.sample_unique_from_set(self.L_MIA_set, K1, ind_set)
        ind_set.update(g1)

        g2 = self.sample_unique_from_set(self.L_comm_set, K2, ind_set)
        ind_set.update(g2)

        g3 = self.sample_unique_from_set(self.L_MIOA_set, K3, ind_set)
        ind_set.update(g3)

        g4 = self.sample_unique_from_set(self.U_set, K4, ind_set)
        ind_set.update(g4)

        # Fill remaining spots to guarantee exactly K nodes
        if len(ind_set) < self.K:
            fill = self.sample_unique_from_set(self.U_set, self.K - len(ind_set), ind_set)
            ind_set.update(fill)

        return list(ind_set)

    def rectify(self, individual) -> list:
        """Rectification function to repair duplicate nodes using fast shuffle-pop Set Difference."""
        seen = set()
        unique_ind = []
        for node in individual:
            if node not in seen:
                seen.add(node)
                unique_ind.append(node)

        needed = self.K - len(unique_ind)
        if needed <= 0:
            return unique_ind

        available_U = None
        available_L = None

        while len(unique_ind) < self.K:
            r_type = self.rng.random()
            if r_type > 0.5:
                if available_U is None:
                    available_U = list(self.U_set - seen)
                    self.rng.shuffle(available_U)
                
                chosen = None
                while available_U:
                    candidate = available_U.pop()
                    if candidate not in seen:
                        chosen = candidate
                        break
                if chosen is None:
                    remaining = list(self.U_set - seen)
                    if remaining:
                        chosen = self.rng.choice(remaining)
                    else:
                        break
            else:
                if available_L is None:
                    available_L = list(self.L_top_set - seen)
                    self.rng.shuffle(available_L)
                
                chosen = None
                while available_L:
                    candidate = available_L.pop()
                    if candidate not in seen:
                        chosen = candidate
                        break
                if chosen is None:
                    # Fallback to U
                    if available_U is None:
                        available_U = list(self.U_set - seen)
                        self.rng.shuffle(available_U)
                    while available_U:
                        candidate = available_U.pop()
                        if candidate not in seen:
                            chosen = candidate
                            break
                    if chosen is None:
                        remaining = list(self.U_set - seen)
                        if remaining:
                            chosen = self.rng.choice(remaining)
                        else:
                            break

            seen.add(chosen)
            unique_ind.append(chosen)

        return unique_ind

    def crossover(self, pop_sorted) -> list:
        """Crossover strategy using pairwise opposition crossover."""
        offspring = []
        num_pairs = self.pop_size // 2
        for i in range(num_pairs):
            pA = list(pop_sorted[i])
            pB = list(pop_sorted[self.pop_size - i - 1])

            for j in range(self.K):
                if self.rng.random() < self.Pc:
                    pA[j], pB[j] = pB[j], pA[j]

            offspring.append(self.rectify(pA))
            offspring.append(self.rectify(pB))

        if len(offspring) < self.pop_size:
            offspring.append(list(pop_sorted[num_pairs]))

        return offspring

    def mutate(self, individual) -> list:
        """Mutation strategy using gene-level mutation with probability Pm."""
        ind = list(individual)
        seen = set(ind)
        
        available_U = None
        available_L = None

        for j in range(self.K):
            if self.rng.random() < self.Pm:
                old_val = ind[j]
                seen.remove(old_val)

                r_type = self.rng.random()
                chosen = None
                if r_type > 0.5:
                    if available_U is None:
                        available_U = list(self.U_set - seen)
                        self.rng.shuffle(available_U)
                    while available_U:
                        candidate = available_U.pop()
                        if candidate not in seen:
                            chosen = candidate
                            break
                    if chosen is None:
                        remaining = list(self.U_set - seen)
                        if remaining:
                            chosen = self.rng.choice(remaining)
                else:
                    if available_L is None:
                        available_L = list(self.L_top_set - seen)
                        self.rng.shuffle(available_L)
                    while available_L:
                        candidate = available_L.pop()
                        if candidate not in seen:
                            chosen = candidate
                            break
                    if chosen is None:
                        if available_U is None:
                            available_U = list(self.U_set - seen)
                            self.rng.shuffle(available_U)
                        while available_U:
                            candidate = available_U.pop()
                            if candidate not in seen:
                                chosen = candidate
                                break
                        if chosen is None:
                            remaining = list(self.U_set - seen)
                            if remaining:
                                chosen = self.rng.choice(remaining)

                if chosen is not None:
                    ind[j] = chosen
                    seen.add(chosen)
                else:
                    seen.add(old_val)

        return ind

    def evaluate(self, ind) -> float:
        """Evaluate the fitness I(S) using MIA influence model with cache lookup."""
        key = tuple(sorted(ind))
        if key in self.fitness_cache:
            return self.fitness_cache[key]

        if self.mia is not None:
            if hasattr(self.mia, "influence"):
                val = self.mia.influence(ind)
            elif callable(self.mia):
                val = self.mia(ind)
            else:
                val = float(len(set(ind)))
        else:
            val = float(len(set(ind)))
            
        self.fitness_cache[key] = val
        return val

    def run(self) -> list:
        """Run the CTIM-EA algorithm."""
        # 1. Initialize population
        population = [self.create_individual() for _ in range(self.pop_size)]
        fitnesses = [self.evaluate(ind) for ind in population]

        # Track the global best individual
        best_idx = max(range(self.pop_size), key=lambda i: fitnesses[i])
        global_best_ind = list(population[best_idx])
        global_best_fit = fitnesses[best_idx]

        for gen in range(self.g_max):
            # Sort population and fitnesses by descending fitness
            sorted_indices = sorted(range(self.pop_size), key=lambda i: fitnesses[i], reverse=True)
            pop_sorted = [population[i] for i in sorted_indices]
            fits_sorted = [fitnesses[i] for i in sorted_indices]

            # Crossover
            offspring = self.crossover(pop_sorted)

            # Mutation
            offspring = [self.mutate(ind) for ind in offspring]

            # Evaluate offspring
            fits_offspring = [self.evaluate(ind) for ind in offspring]

            # Survivor selection: pairwise comparison (parent vs offspring)
            next_population = []
            next_fitnesses = []
            for i in range(self.pop_size):
                if fits_offspring[i] > fits_sorted[i]:
                    next_population.append(offspring[i])
                    next_fitnesses.append(fits_offspring[i])
                else:
                    next_population.append(pop_sorted[i])
                    next_fitnesses.append(fits_sorted[i])

                # Update global best
                if next_fitnesses[-1] > global_best_fit:
                    global_best_fit = next_fitnesses[-1]
                    global_best_ind = list(next_population[-1])

            # Elitism: ensure the global best is in the next population
            best_of_next_idx = max(range(self.pop_size), key=lambda i: next_fitnesses[i])
            if next_fitnesses[best_of_next_idx] < global_best_fit:
                worst_of_next_idx = min(range(self.pop_size), key=lambda i: next_fitnesses[i])
                next_population[worst_of_next_idx] = list(global_best_ind)
                next_fitnesses[worst_of_next_idx] = global_best_fit

            population = next_population
            fitnesses = next_fitnesses

        # Return the best individual found
        best_idx = max(range(self.pop_size), key=lambda i: fitnesses[i])
        return population[best_idx]


# ---------------------------------------------------------------------------
# Self-test code
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running CTIM_EA2 self-test...")
    
    # Create dummy Dataset-like and Model-like objects
    class DummyGraph:
        def __init__(self, n_users):
            self.n_users = n_users
            self.nodes = list(range(n_users))
            
    class DummyModel:
        def __init__(self):
            self.eta = [[0.1, 0.2], [0.3, 0.4]]
            self.pi = [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9], [0.8, 0.2]]
            
    # Test L_comm initialization
    dummy_model = DummyModel()
    l_comm = initialize_l_comm(dummy_model, dummy_model.pi, K=2)
    print("Initial L_comm:", l_comm)
    assert len(l_comm) > 0, "L_comm must not be empty"
    
    # Initialize EA class
    dummy_g = DummyGraph(n_users=5)
    ea = CTIM_EA2(
        G=dummy_g,
        K=3,
        pop=6,
        g_max=10,
        L_comm=[0, 1],
        L_MIA=[1, 2],
        L_MIOA=[3],
        mia=lambda S: float(len(set(S)))  # Simple size fitness
    )
    
    # Test population initialization
    pop = [ea.create_individual() for _ in range(ea.pop_size)]
    for ind in pop:
        assert len(ind) == 3, f"Individual size must be exactly 3, got {len(ind)}"
        assert len(set(ind)) == 3, f"Individual must contain unique nodes, got {ind}"
        
    print("Initialization passed.")
    
    # Test crossover and mutation
    sorted_pop = sorted(pop, key=lambda ind: ea.evaluate(ind), reverse=True)
    offspring = ea.crossover(sorted_pop)
    for ind in offspring:
        assert len(ind) == 3 and len(set(ind)) == 3
        
    mutated = [ea.mutate(ind) for ind in offspring]
    for ind in mutated:
        assert len(ind) == 3 and len(set(ind)) == 3
        
    print("Crossover and Mutation passed.")
    
    # Test run
    best_seeds = ea.run()
    print("Best seeds found:", best_seeds)
    assert len(best_seeds) == 3 and len(set(best_seeds)) == 3
    print("Self-test completed successfully.")