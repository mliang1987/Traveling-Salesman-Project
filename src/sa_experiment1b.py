####################################################################################################
# CSE 6140 - Fall 2019
#   Rodrigo Alves Lima
#   Shichao Liang
#   Jeevanjot Singh
#   Kevin Tynes
####################################################################################################


"""
This file contains the SQD experiment for Simulated Annealing.

For a set of fixed times, the algorithm iterates through specified qualities for 50 iterations,
calculating the probability of a solution of at least the specified quality being found.

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

def run_sqd_experiment(city, optimal):
    '''
    Plots the SQD for a given city instance and the optimal path for that city.
    '''
    file_path = "Data/{}.tsp".format(city)
    times = [1, 5, 10, 20]
    qualities = [0, 0.05, 0.1, 0.2, 0.3]
    df2 = pd.DataFrame(index = qualities, columns = times)
    for max_time in times:
        print("Running time",max_time)
        p_values = []
        for quality in qualities:
            print("\tRunning quality",quality)
            test_quality = math.floor((quality+1)*optimal)
            experiment = []
            for i in range(10):
                sol, _, _ = sa.simulated_annealing_single(file_path, random.randint(1,100), time.time(), max_time, test_quality = test_quality)
                experiment.append(sol<=test_quality)
            t_count = experiment.count(True)
            p = t_count / len(experiment)
            p_values.append(p)
        df2[max_time] = p_values
    
    print("Smoothing out splines...")

    for t in times:
        df2[t] = gaussian_filter1d(df2[t].values.tolist(), sigma = 1)
    print("Plotting...")
    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.axis([min(qualities),max(qualities),-0.1,1.1])
    plt.plot(df2[1], color = 'b', linewidth = 1.0)
    plt.plot(df2[5], color = 'g', linewidth = 1.0)
    plt.plot(df2[10], color = 'r', linewidth = 1.0)
    plt.plot(df2[20], color = 'b', linewidth = 1.0, linestyle = "--")
    #plt.plot(df2[1], color = 'g', linewidth = 1.0, linestyle = "--")
    #plt.plot(df2[5], color = 'r', linewidth = 1.0, linestyle = "--")
    plt.legend(["{}s".format(item) for item in times])
    plt.title("Solution Quality Distributions for {}".format(city), fontsize = 10)
    plt.ylabel("Probability(Solve)", fontsize = 8)
    plt.xlabel("Relative Solution Quality [ratio]", fontsize = 8)
    plt.savefig("sqd_ls1_{}.png".format(city))

if __name__ == "__main__":
    run_sqd_experiment("Champaign", 52643)
    #run_sqd_experiment("Atlanta", 2003763)
	