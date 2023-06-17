import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_eval_budget_optimization():
    sets = ["A", "B", "C", "D", "E"]
    crossovers = ["UniformCrossover", "OnePointCrossover"]
    crossover_labels = ['UX', '1P']
    markers = ["o", "s"]

    fig, axes = plt.subplots(1, len(sets), figsize=(50, 6))
    fig.suptitle("Evaluation budget optimization", fontsize=16)
    
    for i, set in enumerate(sets):
        for j, cx in enumerate(crossovers):
            dirIn = "output/set{}".format(set)
            filesIn = [os.path.join(dirIn, file) for file in os.listdir(dirIn) if file.startswith("output-num_eval-{}".format(cx))]
            filesIn.sort()
            df = pd.read_csv(filesIn[0], header = 0, index_col = False, skipinitialspace = True)
            df = df[df["success"]]
            df = df.groupby(["dimensionality"]).first().reset_index()
            print(df)
            df.plot(x = "dimensionality", y = "median_num_eval", ax = axes[i], ylabel = 'Needed number of evaluations', label = crossover_labels[j], marker = markers[j], markerfacecolor = 'none', linestyle = "-")
            axes[i].set_title("Set {}".format(set))
    plt.show()       

if __name__ == "__main__":
    plot_eval_budget_optimization()