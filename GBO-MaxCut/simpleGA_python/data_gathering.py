import numpy as np
import os
import datetime
import multiprocessing as mp

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

def run_genetic_algorithm(run_params):
    i, inst, population_size, cx = run_params
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
            num_runs = 10
            pool = mp.Pool()  # creates a pool of worker processes

            for i, inst in enumerate(filesIn):
                print("\nInstance {}/{} \n".format(i+1,num_files_in))
                run_params = [(i, inst, population_size, cx) for _ in range(num_runs)]
                results = pool.map(run_genetic_algorithm, run_params)
                num_success = sum(success for success, _, _, _ in results)
                num_evaluations_list = [evaluations for _, evaluations, _, _ in results]
                print("{}/{} runs successful".format(num_success,num_runs))
                print("{} evaluations (median)".format(np.median(num_evaluations_list)))
                dimensionality = results[0][2]
                num_edges = results[0][3]
                percentiles = np.percentile(num_evaluations_list,[10,50,90])
                f.write("{},{},{},{},{},{},{},{},{}\n".format(dimensionality,num_edges,population_size,num_success/num_runs,np.min(num_evaluations_list),percentiles[0],percentiles[1],percentiles[2],np.max(num_evaluations_list)))
                f.flush()
                os.fsync(f.fileno())
            pool.close()  # close the pool to prevent any more tasks from being submitted
            pool.join()  # wait for all the worker processes to terminate
