import numpy as np
import os
import datetime
import multiprocessing as mp

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

def run_genetic_algorithm(run_params):
    inst, population_size, cx = run_params
    fitness = FitnessFunction.MaxCut(inst)
    genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=100000, verbose=False)
    best_fitness, num_evaluations = genetic_algorithm.run()
    return best_fitness == fitness.value_to_reach, num_evaluations, fitness.dimensionality, len(fitness.edge_list)

if __name__ == "__main__":
    crossovers = ["UniformCrossover", "OnePointCrossover"]
    set = "A"
    currentDT = datetime.datetime.now()

    dirOut = "output/set{}".format(set)
    dirIn = "maxcut-instances/set{}".format(set)

    filesIn = [os.path.join(dirIn, file) for file in os.listdir(dirIn) if file.endswith(".txt")]
    num_files_in = len(filesIn)
    os.makedirs(dirOut,exist_ok=True)

    for cx in crossovers:
        print("\nCrossover: {}\n".format(cx))
        file_out = "output-{}_{}.csv".format(cx, currentDT.strftime("%d-%m-%Y_%H-%M"))
        with open(os.path.join(dirOut,file_out),"w") as f:
            f.write("dimensionality, num_edges, population_size,success_rate, min, 10th_percentile,50th_percentile,90th_percentile, max\n")
            population_size = 500
            pool = mp.Pool()  # creates a pool of worker processes

            run_params = [(inst, population_size, cx) for inst in filesIn]
            results = pool.map(run_genetic_algorithm, run_params)

            for i, result in enumerate(results):
                print("\nInstance {}/{} \n".format(i+1,num_files_in))
                success, num_evaluations, dimensionality, num_edges = result
                print("{}/{} run successful".format(success, 1))
                print("{} evaluations (total)".format(num_evaluations))
                f.write("{},{},{},{},{},{},{},{},{}\n".format(dimensionality,num_edges,population_size,success,num_evaluations,num_evaluations,num_evaluations,num_evaluations,num_evaluations))
                f.flush()
                os.fsync(f.fileno())
            pool.close()  # close the pool to prevent any more tasks from being submitted
            pool.join()  # wait for all the worker processes to terminate
