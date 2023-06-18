import numpy as np

import Individual
from Utils import ValueToReachFoundException


class FitnessFunction:
    def __init__(self):
        self.dimensionality = 1
        self.number_of_evaluations = 0
        self.value_to_reach = np.inf

    def evaluate(self, individual: Individual):
        self.number_of_evaluations += 1
        if individual.fitness >= self.value_to_reach:
            raise ValueToReachFoundException(individual)


class MaxCut(FitnessFunction):
    def __init__(self, instance_file):
        super().__init__()
        self.weights = None
        self.edge_list = None
        self.read_problem_instance(instance_file)
        self.read_value_to_reach(instance_file)

    def read_problem_instance(self, instance_file):
        with open(instance_file, "r") as f_in:
            lines = f_in.readlines()
            first_line = lines[0].split()
            self.dimensionality = int(first_line[0])
            number_of_edges = int(first_line[1])
            self.edge_list = np.zeros((number_of_edges, 2), dtype=np.int32)
            self.weights = np.zeros((number_of_edges,), dtype=float)
            for i, line in enumerate(lines[1:]):
                splt = line.split()
                v0 = int(splt[0]) - 1
                v1 = int(splt[1]) - 1
                assert (0 <= v0 < self.dimensionality)
                assert (0 <= v1 < self.dimensionality)
                w = float(splt[2])
                self.edge_list[i] = (v0, v1)
                self.weights[i] = w

    def read_value_to_reach(self, instance_file):
        bkv_file = instance_file.replace(".txt", ".bkv")
        with open(bkv_file, "r") as f_in:
            lines = f_in.readlines()
            first_line = lines[0].split()
            self.value_to_reach = float(first_line[0])

    def evaluate(self, individual: Individual):
        genotypes_edges = individual.genotype[self.edge_list]
        not_equal_indices = genotypes_edges[:, 0] != genotypes_edges[:, 1]
        random_check = self.weights[not_equal_indices]
        individual.fitness = np.sum(self.weights[not_equal_indices])
        super().evaluate(individual)

    def get_weight( self, v0, v1 ):
        if( not (v0,v1) in self.weights ):
            return 0

        return self.weights[(v0,v1)]

    def get_degree( self, v ):
        return len(self.adjacency_list(v))

    def evaluate_single_node_flip(self, individual: Individual, node_index):
        # Check if node_index is in the genotype and give meaningful error message
        assert (0 <= node_index < self.dimensionality), f"Node index {node_index} is not in the genotype"

        # Create a copy of the individual
        new_individual = individual.copy()

        # Flip the node
        new_individual.genotype[node_index] = 1 - new_individual.genotype[node_index]

        # Evaluate the new individual
        self.number_of_evaluations -= (1 - 1/self.dimensionality)

        self.evaluate(new_individual)

        if new_individual.fitness < individual.fitness:
            raise ValueError("Fitness of new individual is lower than fitness of original individual")

        return new_individual

    # TODO: Speed up this function
    def evaluate_single_node_flip_slow(self, individual: Individual, node_index):
        # Check if node_index is in the genotype and give meaningful error message
        assert (0 <= node_index < self.dimensionality), f"Node index {node_index} is not in the genotype"

        # Create a copy of the individual
        new_individual = individual.copy()

        edge_list = np.any(self.edge_list == node_index, axis=1)

        # Get the edges that are affected by the node flip
        edges_to_flip = self.edge_list[edge_list]

        # Get the genotypes of the edges that are affected by the node flip
        genotypes_edges_to_flip = individual.genotype[edges_to_flip]

        # Get the indices of the edges that are affected by the node flip
        indices_edges_to_flip = np.where(edge_list)[0]

        # Get the weights of the edges that are affected by the node flip
        weights_edges_to_flip = self.weights[indices_edges_to_flip]

        # Add the weights where indices are the same and subtract the weights where indices are different
        fitness_addition = np.sum(weights_edges_to_flip[genotypes_edges_to_flip[:, 0] == genotypes_edges_to_flip[:, 1]]) - \
                  np.sum(weights_edges_to_flip[genotypes_edges_to_flip[:, 0] != genotypes_edges_to_flip[:, 1]])

        if fitness_addition < 0:
            raise ValueError("Fitness addition is negative")

        # Flip the genotype of the node
        new_individual.genotype[node_index] = 1 - new_individual.genotype[node_index]

        # Update the fitness of the new individual
        new_individual.fitness = individual.fitness + fitness_addition

        if new_individual.fitness >= self.value_to_reach:
            raise ValueToReachFoundException(new_individual)

        self.number_of_evaluations += (1/self.dimensionality)

        # Return the new individual
        return new_individual

    def __repr__(self):
        return f"MaxCut(dimensionality={self.dimensionality}, value_to_reach={self.value_to_reach})"

    def __str__(self):
        return self.__repr__()
