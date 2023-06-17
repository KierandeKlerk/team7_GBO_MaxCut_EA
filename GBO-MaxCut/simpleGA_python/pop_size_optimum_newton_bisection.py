import os
import datetime
import multiprocessing as mp
import numpy as np  

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

def run_genetic_algorithm_pop_size_optimization(run_params):
    inst, population_size, cx, evaluation_budget = run_params
    success = False
    fitness = FitnessFunction.MaxCut(inst)
    genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=evaluation_budget, verbose=False)
    best_fitness, num_evaluations = genetic_algorithm.run()
    if best_fitness == fitness.value_to_reach: 
        success = True
        print("Population size: {} successful".format(population_size))
    else:
        success = False
        print("Population size: {} failed".format(population_size)) 
        

    return success, num_evaluations

   
def newton_bisection_run(inst, cx, precision, n_runs):
    with open(inst,"r") as f:
        lines = f.readlines()
        line = lines[0].split()
        dimensionality = int(line[0])
        num_edges = int(line[1])
    evaluation_budget = 200*dimensionality*num_edges

    lower_bound = int(4)
    upper_bound = int(round(10*num_edges*dimensionality, -1))
    width = upper_bound - lower_bound

    optimal = int(upper_bound)
    success_lower=False
    success_upper=False
    print("Lower, Dimensionality: {}, cx: {}".format(dimensionality, cx))
    with mp.Pool() as pool:
        results_lower = np.array(list(pool.imap_unordered(run_genetic_algorithm_pop_size_optimization, [(inst, lower_bound, cx, evaluation_budget)]*n_runs) ))
        success_lower = np.all(results_lower[:,0]==1)


    if not success_lower:
        print("Upper, Dimensionality: {}, cx: {}".format(dimensionality, cx))
        with mp.Pool() as pool:
            results_upper = np.array(list(pool.imap_unordered(run_genetic_algorithm_pop_size_optimization, [(inst, upper_bound, cx, evaluation_budget)]*n_runs) ))
            success_upper = np.all(results_upper[:,0]==1)
            median_num_evaluations_upper = np.median(results_upper[:,1])
        if not success_upper:
            print("Upper bound was too low")
            return cx, optimal, dimensionality, median_num_evaluations_upper, False
    else:
        print("Lower bound was too high")
        return cx, optimal, dimensionality, median_num_evaluations_upper, False
        
    
    while width/optimal > precision:
        middle_bound = int(round((lower_bound + upper_bound) / 2, -1))

        print("Middle, Dimensionality: {}, cx: {}".format(dimensionality, cx))
        with mp.Pool() as pool:
            results_middle = np.array(list(pool.imap_unordered(run_genetic_algorithm_pop_size_optimization, [(inst, middle_bound, cx, evaluation_budget)]*n_runs) ))
            success_middle = np.all(results_middle[:,0]==1)
            median_num_evaluations_middle = np.median(results_middle[:,1])
        
        if success_middle:
            upper_bound = middle_bound
            optimal = middle_bound
            median_num_evaluations_upper = median_num_evaluations_middle
            median_num_evaluations_optimal = median_num_evaluations_middle
        else:
            lower_bound = middle_bound
            optimal = upper_bound
            median_num_evaluations_lower = median_num_evaluations_middle
            median_num_evaluations_optimal = median_num_evaluations_upper
        width = upper_bound - lower_bound
    return cx, optimal, dimensionality, median_num_evaluations_optimal, True
    


    


def pop_size_optimization_bisection(set):
    crossovers = ["UniformCrossover", "OnePointCrossover"]
    dirIn = "maxcut-instances/set{}".format(set)
    dirOut = "output/set{}".format(set)
    currentDT = datetime.datetime.now()
    os.makedirs(dirOut,exist_ok=True)

    filesIn = [os.path.join(dirIn, file) for file in os.listdir(dirIn) if file.endswith(".txt")]
    filesIn.sort()
    filesIn = filesIn[0::10]
    
    
    n_runs = 10
    precision = 0.1
    
    with open(os.path.join(dirOut,"output-pop_size_newton-{}.csv".format(currentDT.strftime("%d-%m-%Y_%H-%M"))),"w") as f:
        bisection_params = [(inst, cx, precision, n_runs) for inst in filesIn for cx in crossovers]
        f.write("Crossover, Dimensionality, Population size, Success, Median num evaluations\n")
        for cx in crossovers:
            for inst in filesIn:
                result = newton_bisection_run(inst, cx, precision, n_runs)
        
                
                cx, optimal_pop_size, dimensionality, median_num_evaluations_optimal, success = result
                f.write("{},{},{},{},{}\n".format(cx, dimensionality, optimal_pop_size, success, median_num_evaluations_optimal))
                f.flush()
                os.fsync(f.fileno())
            
        

     
    
if __name__ == "__main__":
    pop_size_optimization_bisection("A")

