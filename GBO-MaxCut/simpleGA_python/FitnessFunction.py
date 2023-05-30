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
            self.weights = np.zeros((number_of_edges, ), dtype=float)
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
        individual.fitness = np.sum(self.weights[not_equal_indices])
        super().evaluate(individual)

