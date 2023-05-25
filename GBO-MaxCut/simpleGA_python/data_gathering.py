import numpy as np
import os
import datetime

from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

if __name__ == "__main__":
    crossovers = ["UniformCrossover", "OnePointCrossover"]
    set = "A"
    currentDT = datetime.datetime.now()
    
    dirOut = "output/set{}".format(set)
    dirIn = "maxcut-instances/set{}".format(set)

    filesIn = []
    for file in os.listdir(dirIn):
        if file.endswith(".txt"):
            filesIn.append(os.path.join(dirIn,file))
    num_files_in = len(filesIn)
    os.makedirs(dirOut,exist_ok=True)

    for cx in crossovers:
        print("\nCrossover: {}\n".format(cx))
        file_out = "output-{}_.csv".format(cx, currentDT.strftime("%d-%m-%Y_%H-%M"))
        with open(os.path.join(dirOut,file_out),"w") as f:
            f.write("dimensionality, num_edges, population_size,success_rate, min, 10th_percentile,50th_percentile,90th_percentile, max\n")
            population_size = 500
            num_evaluations_list = []
            num_runs = 10   
            for i, inst in enumerate(filesIn):
                print("\nInstance {}/{} \n".format(i+1,num_files_in))
                num_success = 0
                for i in range(num_runs):
                    fitness = FitnessFunction.MaxCut(inst)
                    genetic_algorithm = GeneticAlgorithm(fitness,population_size,variation=cx,evaluation_budget=100000,verbose=False)
                    best_fitness, num_evaluations = genetic_algorithm.run()
                    if best_fitness == fitness.value_to_reach:
                        num_success += 1
                    num_evaluations_list.append(num_evaluations)
                print("{}/{} runs successful".format(num_success,num_runs))
                print("{} evaluations (median)".format(np.median(num_evaluations_list)))
                percentiles = np.percentile(num_evaluations_list,[10,50,90])
                f.write("{},{},{},{},{},{},{},{},{}\n".format(fitness.dimensionality,len(fitness.edge_list),population_size,num_success/num_runs,np.min(num_evaluations_list),percentiles[0],percentiles[1],percentiles[2],np.max(num_evaluations_list)))
                f.flush()
                os.fsync(f)