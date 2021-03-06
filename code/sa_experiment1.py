####################################################################################################
# CSE 6140 - Fall 2019
#   Rodrigo Alves Lima
#   Shichao Liang
#   Jeevanjot Singh
#   Kevin Tynes
####################################################################################################


"""
This file contains the QRTD experiment for Simulated Annealing.

For a set of fixed qualities, the algorithm iterates through specified time constraints 
for 50 iterations, calculating the probability of a solution of at least the specified quality being found.

Then, plots the result.
"""


import simulated_annealing as sa
import random
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d

def run_qrtd_experiment(city, optimal):
    """
    Plots the QRTD for a given city instance and the optimal path for that city.
    """
    file_path = "Data/{}.tsp".format(city)
    times = [0.001, 0.01, 0.1, 1, 10]
    qualities = [0, 0.1, 0.5]
    df2 = pd.DataFrame(index = times, columns = qualities)
    for quality in qualities:
        print("Running quality",quality)
        test_quality = math.floor((quality+1)*optimal)
        p_values = []
        for max_time in times:
            print("\tRunning times",max_time)
            experiment = []
            for i in range(50):
                print("\t\tRunning iteration",i)
                sol, _, _ = sa.simulated_annealing_single(file_path, random.randint(1,100), time.time(), max_time, test_quality = test_quality)
                print(max_time, quality, i, sol)
                experiment.append(sol<=test_quality)
            t_count = experiment.count(True)
            p = t_count / len(experiment)
            p_values.append(p)
        df2[quality] = p_values
    
    print("Smoothing out splines...")
    for quality in qualities:
        df2[quality] = gaussian_filter1d(df2[quality].values.tolist(), sigma = 1.0)

    print("Plotting")
    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.xscale("log")
    plt.axis([min(times), max(times),-0.1,1.1])
    plt.plot(df2[0], color = 'b', linewidth = 1.0)
    plt.plot(df2[0.1], color = 'g', linewidth = 1.0)
    plt.plot(df2[0.5], color = 'b', linewidth = 1.0, linestyle = '--')
    plt.legend(["Opt", "0.01 err", "0.5 err"])
    plt.title("Qualified RTDs for {}".format(city), fontsize = 10)
    plt.ylabel("Probability(Solve)", fontsize = 8)
    plt.xlabel("Run-time [CPU sec]", fontsize = 8)
    plt.savefig("qrtd_ls1_{}.png".format(city))

if __name__ == "__main__":
    run_qrtd_experiment("Atlanta", 2003763)
    #run_qrtd_experiment("Champaign", 52643)
