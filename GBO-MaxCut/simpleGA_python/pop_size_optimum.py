import os
import datetime
import multiprocessing as mp
import numpy as np  

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

def run_genetic_algorithm_pop_size_optimization(run_params):
    inst, population_size, cx, n_runs = run_params
    success = True
    list_num_evaluations = []
    for _ in range(n_runs):
        fitness = FitnessFunction.MaxCut(inst)
        genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=100000, verbose=False)
        best_fitness, num_evaluations = genetic_algorithm.run()
        list_num_evaluations.append(num_evaluations)
        if best_fitness != fitness.value_to_reach: 
            print("Population size: {} failed".format(population_size)) 
            success = False
            break

    return success, np.median(list_num_evaluations), fitness.dimensionality, len(fitness.edge_list), population_size

def pop_size_optimization(set):
    crossovers = ["UniformCrossover", "OnePointCrossover"]


    dirIn = "maxcut-instances/set{}".format(set)
    dirOut = "output/set{}".format(set)
    currentDT = datetime.datetime.now()
    os.makedirs(dirOut,exist_ok=True)

    filesIn = [os.path.join(dirIn, file) for file in os.listdir(dirIn) if file.endswith(".txt")]
    filesIn.sort()

    n_runs = 10

    for cx in crossovers:
        file_out = "output-pop_size-{}_{}.csv".format(cx, currentDT.strftime("%d-%m-%Y_%H-%M"))
        with open(os.path.join(dirOut,file_out),"w") as f:
            f.write("dimensionality,num_edges,population_size,success,median_num_eval\n")
            
            run_params = [(inst, population_size, cx, n_runs) for inst in filesIn[0::10] for population_size in [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000]]
            
            pool = mp.Pool()  # creates a pool of worker processes
            results = pool.map(run_genetic_algorithm_pop_size_optimization, run_params)
            for i, result in enumerate(results):
                success, median_num_evaluations, dimensionality, num_edges, population_size = result
                print("\nDimensionality: {}, Population size: {}, Success, {} \n".format(dimensionality, population_size, success))
                f.write("{},{},{},{},{}\n".format(dimensionality,num_edges,population_size,success,median_num_evaluations))
                f.flush()
                os.fsync(f.fileno())

            pool.close()  # close the pool to prevent any more tasks from being submitted
            pool.join()  # wait for all the worker processes to terminate
            
if __name__ == "__main__":
    pop_size_optimization("A")
