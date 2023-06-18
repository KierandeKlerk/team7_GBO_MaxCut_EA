import os
import datetime
import multiprocessing as mp
import numpy as np  

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

def round_even(x):
    return round(x / 2) * 2

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

   
def newton_bisection_run(insts, cx, precision, n_runs):
    with open(insts[0],"r") as f:
        lines = f.readlines()
        line = lines[0].split()
        dimensionality = int(line[0])
        num_edges = int(line[1])
    evaluation_budget = 300*dimensionality*num_edges

    lower_bound = int(0)
    upper_bound = round_even(10*num_edges*dimensionality)
    median_num_evaluations_upper = None
    width = upper_bound - lower_bound

    optimal = int(upper_bound)
    success_upper=False
        
    
    print("Upper, Dimensionality: {}, cx: {}".format(dimensionality, cx))
    with mp.Pool() as pool:
        results_upper = np.array(list(pool.imap_unordered(run_genetic_algorithm_pop_size_optimization, [(inst, upper_bound, cx, evaluation_budget) for inst in insts]) ))
        success_upper = np.all(results_upper[:,0]==1)
        median_num_evaluations_upper = np.median(results_upper[:,1])
    if not success_upper:
        print("Upper bound was too low")
        return cx, optimal, dimensionality, median_num_evaluations_upper, False
    
        
    
    while width/optimal >= precision:
        middle_bound = round_even((lower_bound + upper_bound) / 2)

        print("Middle, Dimensionality: {}, cx: {}".format(dimensionality, cx))
        with mp.Pool() as pool:
            results_middle = np.array(list(pool.imap_unordered(run_genetic_algorithm_pop_size_optimization, [(inst, middle_bound, cx, evaluation_budget) for inst in insts]) ))
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
        print("Lower bound: {}, Upper bound: {}, Optimal: {}, Width: {}, Ratio {}".format(lower_bound, upper_bound, optimal, width, width/optimal))
    return cx, optimal, dimensionality, median_num_evaluations_optimal, True
    


    


def pop_size_optimization_bisection(set):
    crossovers = ["UniformCrossover", "OnePointCrossover"]
    dirIn = "maxcut-instances/set{}".format(set)
    dirOut = "output/set{}".format(set)
    currentDT = datetime.datetime.now()
    os.makedirs(dirOut,exist_ok=True)

    filesIn = [os.path.join(dirIn, file) for file in os.listdir(dirIn) if file.endswith(".txt")]
    filesIn.sort()    
    
    n_runs = 10
    precision = 0.2501
    
    with open(os.path.join(dirOut,"output-pop_size_newton-{}.csv".format(currentDT.strftime("%d-%m-%Y_%H-%M"))),"w") as f:

        f.write("Crossover, Dimensionality, Population size, Success, Median num evaluations\n")
        
        for i in range(int(len(filesIn)/10)):
            for cx in crossovers:
                result = newton_bisection_run(filesIn[0+i*10:10+i*10], cx, precision, n_runs)
                
                cx, optimal_pop_size, dimensionality, median_num_evaluations_optimal, success = result
                f.write("{},{},{},{},{}\n".format(cx, dimensionality, optimal_pop_size, success, median_num_evaluations_optimal))
                f.flush()
                os.fsync(f.fileno())
            
        

     
    
if __name__ == "__main__":
    pop_size_optimization_bisection("D")

