import numpy as np


# a function to initialize the population
def initialize_pop(dna_size, pop_size, fixed_tot_features=False, fixed_tot_feature_size=5):
    if fixed_tot_features:
        a = (np.random.choice(dna_size, size=fixed_tot_feature_size, replace=False) for _ in range(pop_size))
        b = np.vstack(a)
        pop = np.zeros((pop_size, dna_size), dtype=int)
        for i in range(pop_size):
            pop[i, tuple(b[i, :])] = 1
    else:
        pop = np.array([np.random.randint(0, 2, dna_size) for _ in range(pop_size)])
    return pop


# a function to select the top few performing chromosomes
# repeating these to reach the original population size
def select_top(pop, acc_list, top_count=5):
    idx = np.array(acc_list).argsort()[-top_count:][::-1]
    repeat_count = int(len(pop)/top_count)
    repeat_remainder = len(pop) % top_count
    remainder_idx = idx[:repeat_remainder]
    idx = np.hstack((np.repeat(idx, repeat_count), remainder_idx))
    selected_pop = pop[idx]
    np.random.shuffle(selected_pop)
    return selected_pop


# a function to select the top few performing chromosomes
# repeating these to reach the original population size
def select_top_dual(pop, acc_list, top_count=5, len_factor=0.1):
    acc = np.array(acc_list) - len_factor*np.sum(pop, axis=1)
    idx = acc.argsort()[-top_count:][::-1]
    repeat_count = int(len(pop)/top_count)
    repeat_remainder = len(pop) % top_count
    remainder_idx = idx[:repeat_remainder]
    idx = np.hstack((np.repeat(idx, repeat_count), remainder_idx))
    selected_pop = pop[idx]
    np.random.shuffle(selected_pop)
    return selected_pop


# function to choose chromosomes according to their accuracy
# done with replacement so as to allow repeats
def select_probability(pop, acc_list, len_factor=0.1):
    pop_size = len(pop)
    acc = np.array(acc_list) - len_factor * np.sum(pop, axis=1)
    # less than zero gets zero probability, avoid divide by zero (add epsilon)
    acc[np.where(acc<0)] = 0 + 1e-5
    idx = np.random.choice(np.arange(pop_size), size=pop_size, replace=True,
                           p=acc / np.sum(acc))
    return pop[idx]


# crossover for a particular gene w.r.t. the whole population
def crossover(parent, pop, cross_rate):
    pop_size, dna_size = pop.shape
    if np.random.rand() < cross_rate:
        # randomly select another individual from population
        i = np.random.randint(0, pop_size, size=1)
        # choose crossover points(bits)
        cross_points = np.random.randint(0, 2, size=dna_size).astype(np.bool)
        # produce one child
        parent[cross_points] = pop[i, cross_points]
    return parent


# mutate function
def mutate(child, mutation_rate, dna_size):
    for point in range(dna_size):
        if np.random.rand() < mutation_rate:
            child[point] = 1 if child[point] == 0 else 0
    return child


# crossover and mutate
def cross_mutate(pop, cross_rate, mutation_rate):
    pop_copy = pop.copy()
    pop_size, dna_size = pop.shape
    for parent in pop:
        # produce a child by crossover operation
        child = crossover(parent, pop_copy, cross_rate)
        # mutate child
        child = mutate(child, mutation_rate, dna_size)
        # replace parent with its child
        parent[:] = child
    return pop