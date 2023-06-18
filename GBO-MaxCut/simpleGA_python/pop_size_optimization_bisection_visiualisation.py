import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
import datetime

def plot_pop_size_optimization():
    sets = ["A", "B", "C", "D", "E"]
    crossovers = ["UniformCrossover", "OnePointCrossover"]
    crossover_labels = ['UX', '1P']
    markers = ["o", "s"]
    DT  = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    fig, axes = plt.subplots(1, len(sets), figsize=(50, 6), sharey=True, sharex=True)
    fig.suptitle("Population size optimization", fontsize=16)
    
    for i, set in enumerate(sets):
        for j, cx in enumerate(crossovers):
            dirIn = "output/set{}".format(set)
            filesIn = [os.path.join(dirIn, file) for file in os.listdir(dirIn) if file.startswith("output-pop_size_newton")]
            filesIn.sort()
            df = pd.read_csv(filesIn[-1], header = 0, index_col = False, skipinitialspace = True)
            df = df[df["Crossover"] == cx]
            df = df[df["Success"]]
            print(df)

            df.plot(x = "Dimensionality", y = "Population size", xlabel='', ax = axes[i], marker = markers[j], markerfacecolor = 'none', linestyle = "-", legend=False)
            axes[i].set_title("Set {}".format(set), fontsize=14)
            axes[i].set_yscale('log')
            axes[i].set_xscale('log')
            axes[i].tick_params(axis='both', labelsize=14)
    custom_lines = [Line2D([0], [0], color="tab:blue", lw=2, ls="-", markerfacecolor= 'none', marker="o"),
    Line2D([0], [0], color="tab:orange", lw=2, ls="-", markerfacecolor = 'none', marker="s")]
    fig.legend(custom_lines, crossover_labels, loc='lower right', ncol=2, fontsize=14)
    fig.supylabel(r'$n$ [-]', fontsize = 14)
    fig.supxlabel(r'$D$ [-]', fontsize = 14)
    plt.subplots_adjust(
        top=0.9,
        bottom=0.11,
        left=0.035,
        right=0.98,
        hspace=0.2,
        wspace=0.2
    )
    plt.show()

if __name__ == "__main__":
    plot_pop_size_optimization()