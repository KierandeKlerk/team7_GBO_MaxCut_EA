from functools import partial

import numpy as np

import Selection
import Variation
from FitnessFunction import FitnessFunction
from Individual import Individual
from Utils import ValueToReachFoundException


class GeneticAlgorithm:
    def __init__(self, fitness: FitnessFunction, population_size, **options):
        self.fitness = fitness
        self.evaluation_budget = options.get("evaluation_budget", 1000000)
        self.variation_operator = self._get_variation_operator(options.get("variation", "UniformCrossover"))
        self.selection_operator = Selection.tournament_selection
        self.population_size = population_size
        self.population = []
        self.number_of_generations = 0
        self.verbose = options.get("verbose", False)
        self.print_final_results = True
        self.partial_evaluations = options.get("partial_evaluations", False)

    def _get_variation_operator(self, variation):
        match variation:
            case "UniformCrossover":
                return Variation.uniform_crossover
            case "OnePointCrossover":
                return Variation.one_point_crossover
            case "TwoPointCrossover":
                return Variation.two_point_crossover
            case "CustomCrossover":
                return partial(Variation.grouped_uniform_crossover, self.fitness)
            case _:
                raise ValueError(f"Unknown variation operator: {variation}")

    def initialize_population(self):
        self.population = [Individual.initialize_uniform_at_random(self.fitness.dimensionality) for i in
                           range(self.population_size)]
        for individual in self.population:
            self.fitness.evaluate(individual)

    def make_offspring(self):
        offspring = []
        order = np.random.permutation(self.population_size)
        for i in range(len(order) // 2):
            offspring = offspring + self.variation_operator(self.population[order[2 * i]],
                                                            self.population[order[2 * i + 1]])
        for individual in offspring:
            self.fitness.evaluate(individual)
        return offspring

    def make_selection(self, offspring):
        return self.selection_operator(self.population, offspring)

    def print_statistics(self):
        fitness_list = [ind.fitness for ind in self.population]
        print(f"Generation {self.number_of_generations}:"
              f"Best_fitness: {max(fitness_list)},"
              f"Avg._fitness: {np.mean(fitness_list)},"
              f"Nr._of_evaluations: {self.fitness.number_of_evaluations}")

    def get_best_fitness(self):
        return max(ind.fitness for ind in self.population)

    def run(self):
        try:
            self.initialize_population()
            while self.fitness.number_of_evaluations < self.evaluation_budget:
                self.number_of_generations += 1
                if self.verbose and self.number_of_generations % 100 == 0:
                    self.print_statistics()

                offspring = self.make_offspring()
                if self.partial_evaluations:
                    for individual in offspring:
                        try:
                            new_individual = self.fitness.evaluate_single_node_flip_slow(individual, np.random.randint(0, self.fitness.dimensionality))
                            # If the new individual is better than the old one, replace it
                            if new_individual.fitness > individual.fitness:
                                individual.fitness = new_individual.fitness
                                individual.genotype = new_individual.genotype
                        except ValueToReachFoundException as exception:
                            raise exception
                        except ValueError:
                            pass
                selection = self.make_selection(offspring)
                self.population = selection
            if self.verbose:
                self.print_statistics()
        except ValueToReachFoundException as exception:
            if self.print_final_results:
                print(exception)
                print(f"Best fitness: {exception.individual.fitness},"
                      f"Nr._of_evaluations: {self.fitness.number_of_evaluations}")
            return exception.individual.fitness, self.fitness.number_of_evaluations
        if self.print_final_results:
            self.print_statistics()
        return self.get_best_fitness(), self.fitness.number_of_evaluations
