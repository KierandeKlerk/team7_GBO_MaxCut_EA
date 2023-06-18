import numpy as np

from FitnessFunction import FitnessFunction
from Individual import Individual


def uniform_crossover(individual_a: Individual, individual_b: Individual, p=0.5):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(l)
    offspring_b = Individual(l)

    m = np.random.choice((0, 1), p=(p, 1 - p), size=l)
    offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
    offspring_b.genotype = np.where(1 - m, individual_a.genotype, individual_b.genotype)

    return [offspring_a, offspring_b]


def one_point_crossover(individual_a: Individual, individual_b: Individual):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(l)
    offspring_b = Individual(l)

    l = len(individual_a.genotype)
    m = np.arange(l) < np.random.randint(l + 1)
    offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
    offspring_b.genotype = np.where(~m, individual_a.genotype, individual_b.genotype)

    return [offspring_a, offspring_b]


def two_point_crossover(individual_a: Individual, individual_b: Individual):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    offspring_a = Individual()
    offspring_b = Individual()

    l = len(individual_a.genotype)
    m = (np.arange(l) < np.random.randint(l + 1)) ^ (np.arange(l) < np.random.randint(l + 1))
    offspring_a.genotype = np.where(m, individual_b.genotype, individual_a.genotype)
    offspring_b.genotype = np.where(~m, individual_b.genotype, individual_a.genotype)

    return [offspring_a, offspring_b]


def grouped_one_point_crossover(fitness: FitnessFunction, individual_a: Individual, individual_b: Individual):
    """Splits the parents only at the boundaries between the groups.
    Applicable only to set D."""
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(l)
    offspring_b = Individual(l)

    # Implement your custom crossover here
    
    # Define the group intersection points
    intersection_points = [i for i in range(5, l, 5)]

    split_point = intersection_points[np.random.randint(0, len(intersection_points))]

    m = np.arange(l) < split_point
    # Apply crossover to create offspring
    offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
    offspring_b.genotype = np.where(~m, individual_a.genotype, individual_b.genotype)

    return [offspring_a, offspring_b]

def grouped_uniform_crossover(fitness: FitnessFunction, individual_a: Individual, individual_b: Individual):
    """Modified uniform crossover to group-level. At the boundary between two groups,
    the colors of the vertices are forced to be different."""
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(l)
    offspring_b = Individual(l)
    offspring_a.genotype = np.zeros(l)
    offspring_b.genotype = np.zeros(l)

    # Implement your custom crossover here
    
    # Define the group size and calculate the number of groups
    group_size = 5
    num_groups = l // group_size

    # Set the first group
    if np.random.rand() < 0.5:
        offspring_a.genotype[0:group_size] = individual_a.genotype[0:group_size]
        offspring_b.genotype[0:group_size] = individual_b.genotype[0:group_size]
    else:
        offspring_a.genotype[0:group_size] = individual_b.genotype[0:group_size]
        offspring_b.genotype[0:group_size] = individual_a.genotype[0:group_size]

    # Sequentially check all other groups of 5
    for group in range(1, num_groups):
        start = group * group_size
        end = start + group_size
        
        if np.random.rand() < 0.5:
            # Offspring A
            if offspring_a.genotype[start-1] == individual_a.genotype[start]:
                offspring_a.genotype[start:end] = np.logical_not(individual_a.genotype[start:end])
            else:
                offspring_a.genotype[start:end] = individual_a.genotype[start:end]
            
            # Offspring B
            if offspring_b.genotype[start-1] == individual_b.genotype[start]:
                offspring_b.genotype[start:end] = np.logical_not(individual_b.genotype[start:end])
            else:
                offspring_b.genotype[start:end] = individual_b.genotype[start:end]
        else:
            # Offspring A
            if offspring_a.genotype[start-1] == individual_b.genotype[start]:
                offspring_a.genotype[start:end] = np.logical_not(individual_b.genotype[start:end])
            else:
                offspring_a.genotype[start:end] = individual_b.genotype[start:end]
            
            # Offspring B
            if offspring_b.genotype[start-1] == individual_a.genotype[start]:
                offspring_b.genotype[start:end] = np.logical_not(individual_a.genotype[start:end])
            else:
                offspring_b.genotype[start:end] = individual_a.genotype[start:end]

    return [offspring_a, offspring_b]

def grouped_uniform_crossover_with_mutation(fitness: FitnessFunction, individual_a: Individual, individual_b: Individual):
    """Modified uniform crossover to group-level. At the boundary between two groups,
    the colors of the vertices are forced to be different."""
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(l)
    offspring_b = Individual(l)
    offspring_a.genotype = np.zeros(l)
    offspring_b.genotype = np.zeros(l)

    # Mutation probability
    pm = 1/l

    # Implement your custom crossover here
    
    # Define the group size and calculate the number of groups
    group_size = 5
    num_groups = l // group_size

    # Set the first group
    if np.random.rand() < 0.5:
        offspring_a.genotype[0:group_size] = individual_a.genotype[0:group_size]
        offspring_b.genotype[0:group_size] = individual_b.genotype[0:group_size]
    else:
        offspring_a.genotype[0:group_size] = individual_b.genotype[0:group_size]
        offspring_b.genotype[0:group_size] = individual_a.genotype[0:group_size]

    # Sequentially check all other groups of 5
    for group in range(1, num_groups):
        start = group * group_size
        end = start + group_size
        
        if np.random.rand() < 0.5:
            # Offspring A
            if offspring_a.genotype[start-1] == individual_a.genotype[start]:
                offspring_a.genotype[start:end] = np.logical_not(individual_a.genotype[start:end])
            else:
                offspring_a.genotype[start:end] = individual_a.genotype[start:end]
            
            # Offspring B
            if offspring_b.genotype[start-1] == individual_b.genotype[start]:
                offspring_b.genotype[start:end] = np.logical_not(individual_b.genotype[start:end])
            else:
                offspring_b.genotype[start:end] = individual_b.genotype[start:end]
        else:
            # Offspring A
            if offspring_a.genotype[start-1] == individual_b.genotype[start]:
                offspring_a.genotype[start:end] = np.logical_not(individual_b.genotype[start:end])
            else:
                offspring_a.genotype[start:end] = individual_b.genotype[start:end]
            
            # Offspring B
            if offspring_b.genotype[start-1] == individual_a.genotype[start]:
                offspring_b.genotype[start:end] = np.logical_not(individual_a.genotype[start:end])
            else:
                offspring_b.genotype[start:end] = individual_a.genotype[start:end]

    for i in range(l):
        for offspring in [offspring_a, offspring_b]:
            if np.random.rand() < pm:
                offspring.genotype[i] = np.logical_not(offspring.genotype[i])
    return [offspring_a, offspring_b]
