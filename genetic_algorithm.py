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

# function to choose chromosomes according to their accuracy
# done with replacement so as to allow repeats
def select_probability(pop, acc_list):
    pop_size = len(pop)
    acc = np.array(acc_list)
    idx = np.random.choice(np.arange(pop_size), size=pop_size, replace=True,
                           p=acc / np.sum(acc))
    return pop[idx]