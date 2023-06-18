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
    return num_success, np.mean(num_evaluations_list), fitness.dimensionality


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

    with open("mean_evaluations.txt", "w") as file:
        for i, (result_partial, result_no_partial) in enumerate(zip(results_partial_eval, results_no_partial_eval)):
            success_partial, mean_eval_partial, dimensionality_partial = result_partial.get()
            success_no_partial, mean_eval_no_partial, dimensionality_no_partial = result_no_partial.get()
            filename = file_list[i]
            file.write("File: {}\n".format(filename))
            file.write("Dimensionality: {}\n".format(dimensionality_partial))
            file.write("Partial Evaluations: True\n")
            file.write("Success Rate: {}/2 runs successful\n".format(success_partial))
            file.write("Mean Evaluations: {}\n\n".format(mean_eval_partial))
            file.write("Partial Evaluations: False\n")
            file.write("Success Rate: {}/2 runs successful\n".format(success_no_partial))
            file.write("Mean Evaluations: {}\n\n".format(mean_eval_no_partial))
            print("File: {}".format(filename))
            print(f"Dimensionality: {dimensionality_partial}")
            print("Partial Evaluations: True")
            print("Success Rate: {}/2 runs successful".format(success_partial))
            print("Mean Evaluations: {}\n".format(mean_eval_partial))
            print("Partial Evaluations: False")
            print("Success Rate: {}/2 runs successful".format(success_no_partial))
            print("Mean Evaluations: {}\n".format(mean_eval_no_partial))

    plot_comparison(file_list, results_partial_eval, results_no_partial_eval)


def plot_comparison(file_list, results_partial_eval, results_no_partial_eval):
    mean_evaluations_partial = []
    success_rates_partial = []
    mean_evaluations_no_partial = []
    success_rates_no_partial = []
    dimensionality_partial_array = []

    for result_partial, result_no_partial in zip(results_partial_eval, results_no_partial_eval):
        success_partial, mean_eval_partial, dimensionality_partial = result_partial.get()
        success_no_partial, mean_eval_no_partial, dimensionality_no_partial = result_no_partial.get()
        success_rates_partial.append(success_partial / 10 * 100)
        mean_evaluations_partial.append(mean_eval_partial)
        success_rates_no_partial.append(success_no_partial / 10 * 100)
        mean_evaluations_no_partial.append(mean_eval_no_partial)
        dimensionality_partial_array.append(dimensionality_partial)

    # Sort the dimensionalities and corresponding data lists
    sorted_indices = np.argsort(dimensionality_partial_array)
    dimensionality_partial_array = np.array(dimensionality_partial_array)[sorted_indices]
    mean_evaluations_partial = np.array(mean_evaluations_partial)[sorted_indices]
    mean_evaluations_no_partial = np.array(mean_evaluations_no_partial)[sorted_indices]
    success_rates_partial = np.array(success_rates_partial)[sorted_indices]
    success_rates_no_partial = np.array(success_rates_no_partial)[sorted_indices]

    x_ticks = list(range(len(dimensionality_partial_array)))

    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = 'tab:red'
    ax1.set_xlabel('Dimensionality')
    ax1.set_ylabel('Mean Evaluations', color=color)
    ax1.plot(x_ticks, mean_evaluations_partial, color=color, label='Partial Evaluations: True')
    ax1.plot(x_ticks, mean_evaluations_no_partial, '--', color=color, label='Partial Evaluations: False')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Success Rate (%)', color=color)
    ax2.plot(x_ticks, success_rates_partial, color=color, label='Partial Evaluations: True')
    ax2.plot(x_ticks, success_rates_no_partial, '--', color=color, label='Partial Evaluations: False')
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(dimensionality_partial_array, rotation=45)
    ax1.set_title('Mean Evaluations and Success Rate Comparison')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig("mean_evaluations_and_success_rate_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Set the seed for reproducibility
    np.random.seed(42)
    start_time = time.time()
    multiprocessing_main()
    print(f"Took {time.time() - start_time} seconds")
