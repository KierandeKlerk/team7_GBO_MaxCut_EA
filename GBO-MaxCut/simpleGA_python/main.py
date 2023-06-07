from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction
import time
import numpy as np

import os
import multiprocessing as mp
import matplotlib.pyplot as plt


def original_main():
    crossovers = ["CustomCrossover", "UniformCrossover", "OnePointCrossover"]
    inst = "maxcut-instances/setB/n0000160i09.txt"
    for cx in crossovers:
        with open(f"output-{cx}.txt", "w") as f:
            population_size = 500
            num_evaluations_list = []
            num_runs = 1
            num_success = 0
            for i in range(num_runs):
                fitness = FitnessFunction.MaxCut(inst)
                genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=100000,
                                                     verbose=True, partial_evaluations=False)
                best_fitness, num_evaluations = genetic_algorithm.run()
                if best_fitness == fitness.value_to_reach:
                    num_success += 1
                num_evaluations_list.append(num_evaluations)
            print("{}/{} runs successful".format(num_success, num_runs))
            print("{} evaluations (median)".format(np.median(num_evaluations_list)))
            percentiles = np.percentile(num_evaluations_list, [10, 50, 90])
            f.write("{} {} {} {} {}\n".format(population_size, num_success / num_runs, percentiles[0], percentiles[1],
                                              percentiles[2]))


def process_file(filepath, partial_evaluations):
    population_size = 500
    num_evaluations_list = []
    num_runs = 10
    num_success = 0
    cx = "CustomCrossover"
    for i in range(num_runs):
        fitness = FitnessFunction.MaxCut(filepath)
        genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=100000,
                                             verbose=True, partial_evaluations=partial_evaluations)
        best_fitness, num_evaluations = genetic_algorithm.run()
        if best_fitness == fitness.value_to_reach:
            num_success += 1
        num_evaluations_list.append(num_evaluations)
    return num_success, np.mean(num_evaluations_list)  # Use np.mean for average/mean calculation

def multiprocessing_main():
    folder_path = "maxcut-instances/setD"
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    results_partial_eval = []
    results_no_partial_eval = []

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for file in file_list:
            filepath = os.path.join(folder_path, file)
            results_partial_eval.append(pool.apply_async(process_file, args=(filepath, True)))
            results_no_partial_eval.append(pool.apply_async(process_file, args=(filepath, False)))

        pool.close()
        pool.join()

    with open("mean_evaluations.txt", "w") as file:  # Rename the output file to "mean_evaluations.txt"
        for i, (result_partial, result_no_partial) in enumerate(zip(results_partial_eval, results_no_partial_eval)):
            success_partial, mean_eval_partial = result_partial.get()  # Rename median_eval_partial to mean_eval_partial
            success_no_partial, mean_eval_no_partial = result_no_partial.get()  # Rename median_eval_no_partial to mean_eval_no_partial
            filename = file_list[i]
            file.write("File: {}\n".format(filename))
            file.write("Partial Evaluations: True\n")
            file.write("Success Rate: {}/2 runs successful\n".format(success_partial))
            file.write("Mean Evaluations: {}\n\n".format(mean_eval_partial))  # Rename Median Evaluations to Mean Evaluations
            file.write("Partial Evaluations: False\n")
            file.write("Success Rate: {}/2 runs successful\n".format(success_no_partial))
            file.write("Mean Evaluations: {}\n\n".format(mean_eval_no_partial))  # Rename Median Evaluations to Mean Evaluations
            print("File: {}".format(filename))
            print("Partial Evaluations: True")
            print("Success Rate: {}/2 runs successful".format(success_partial))
            print("Mean Evaluations: {}\n".format(mean_eval_partial))  # Rename Median Evaluations to Mean Evaluations
            print("Partial Evaluations: False")
            print("Success Rate: {}/2 runs successful".format(success_no_partial))
            print("Mean Evaluations: {}\n".format(mean_eval_no_partial))  # Rename Median Evaluations to Mean Evaluations

    # Plotting
    plot_comparison(file_list, results_partial_eval, results_no_partial_eval)

def plot_comparison(file_list, results_partial_eval, results_no_partial_eval):
    success_rates_partial = []
    mean_evaluations_partial = []  # Rename median_evaluations_partial to mean_evaluations_partial
    success_rates_no_partial = []
    mean_evaluations_no_partial = []  # Rename median_evaluations_no_partial to mean_evaluations_no_partial

    for result_partial, result_no_partial in zip(results_partial_eval, results_no_partial_eval):
        success_partial, mean_eval_partial = result_partial.get()  # Rename median_eval_partial to mean_eval_partial
        success_no_partial, mean_eval_no_partial = result_no_partial.get()  # Rename median_eval_no_partial to mean_eval_no_partial
        success_rates_partial.append(success_partial)
        mean_evaluations_partial.append(mean_eval_partial)  # Rename median_evaluations_partial to mean_evaluations_partial
        success_rates_no_partial.append(success_no_partial)
        mean_evaluations_no_partial.append(mean_eval_no_partial)  # Rename median_evaluations_no_partial to mean_evaluations_no_partial

    x = np.arange(len(file_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, success_rates_partial, width, label='Partial Evaluations: True')
    rects2 = ax.bar(x + width/2, success_rates_no_partial, width, label='Partial Evaluations: False')

    ax.set_xticks(x)
    ax.set_xticklabels(file_list, rotation=45, ha='right')
    ax.set_ylabel('Success Rate')
    ax.set_xlabel('File')
    ax.set_title('Success Rate Comparison')
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    plt.tight_layout()
    plt.savefig("success_rate_comparison.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, mean_evaluations_partial, width, label='Partial Evaluations: True')  # Rename median_evaluations_partial to mean_evaluations_partial
    rects2 = ax.bar(x + width/2, mean_evaluations_no_partial, width, label='Partial Evaluations: False')  # Rename median_evaluations_no_partial to mean_evaluations_no_partial

    ax.set_xticks(x)
    ax.set_xticklabels(file_list, rotation=45, ha='right')
    ax.set_ylabel('Mean Evaluations')  # Rename Median Evaluations to Mean Evaluations
    ax.set_xlabel('File')
    ax.set_title('Mean Evaluations Comparison')  # Rename Median Evaluations to Mean Evaluations
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    plt.tight_layout()
    plt.savefig("mean_evaluations_comparison.png", dpi=300)  # Rename the output file to "mean_evaluations_comparison.png"
    plt.close()

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

if __name__ == "__main__":
    # Set the seed for reproducibility
    # np.random.seed(42)
    start_time = time.time()
    multiprocessing_main()
    print(f"Took {time.time() - start_time} seconds")
