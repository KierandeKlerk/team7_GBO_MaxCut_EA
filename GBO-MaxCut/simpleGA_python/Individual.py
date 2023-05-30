import numpy as np


class Individual:
    def __init__(self, genotype=[]):
        self.genotype = np.array(genotype)
        self.fitness = 0

    @staticmethod
    def initialize_uniform_at_random(genotype_length):
        individual = Individual()
        individual.genotype = np.random.choice((0, 1), p=(0.5, 0.5), size=genotype_length)
        return individual

    def __repr__(self):
        return f"Individual(genotype={self.genotype}, fitness={self.fitness})"

    def __str__(self):
        return self.__repr__()
