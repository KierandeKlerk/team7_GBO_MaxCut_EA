import numpy as np

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction
import time
import numpy as np

if __name__ == "__main__":
    # Set the seed for reproducibility
    np.random.seed(42)
    start_time = time.time()
    crossovers = ["CustomCrossover", "UniformCrossover", "OnePointCrossover"]
    inst = "maxcut-instances/setB/n0000196i02.txt"
    for cx in crossovers:
        with open(f"output-{cx}.txt", "w") as f:
            population_size = 500
            num_evaluations_list = []
            num_runs = 1
            num_success = 0
            for i in range(num_runs):
                fitness = FitnessFunction.MaxCut(inst)
                genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=100000,
                                                     verbose=True, partial_evaluations=True)
                best_fitness, num_evaluations = genetic_algorithm.run()
                if best_fitness == fitness.value_to_reach:
                    num_success += 1
                num_evaluations_list.append(num_evaluations)
            print("{}/{} runs successful".format(num_success, num_runs))
            print("{} evaluations (median)".format(np.median(num_evaluations_list)))
            percentiles = np.percentile(num_evaluations_list, [10, 50, 90])
            f.write("{} {} {} {} {}\n".format(population_size, num_success / num_runs, percentiles[0], percentiles[1],
                                              percentiles[2]))

    print(f"Took {time.time() - start_time} seconds")
