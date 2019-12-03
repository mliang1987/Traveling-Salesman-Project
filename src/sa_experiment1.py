import simulated_annealing as sa
import random
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d


if __name__ == "__main__":
    optimal = 2003763
    file_path = "Data/Atlanta.tsp"
    times = [0.01, 0.1, 1, 10, 100]
    qualities = [0, 0.2, 0.4, 0.6, 0.8]
    df2 = pd.DataFrame(index = times, columns = qualities)
    for quality in qualities:
        print("Running quality",quality)
        test_quality = math.floor((quality+1)*optimal)
        p_values = []
        for max_time in times:
            print("\tRunning times",max_times)
            experiment = []
            for i in range(30):
                sol, _, _ = sa.simulated_annealing_single(file_path, random.randint(1,100), time.time(), max_time, test_quality = test_quality)
                experiment.append(sol<=test_quality)
            t_count = experiment.count(True)
            p = t_count / len(experiment)
            p_values.append(p)
        df2[quality] = p_values
    
    for quality in qualities:
        df2[quality] = gaussian_filter1d(df2[quality].values.tolist(), sigma = 0.5)

    plt.figure()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.axis([0,1.2,-0.1,1.1])
    plt.plot(df2[0.01], color = 'b', linewidth = 1.0)
    plt.plot(df2[0.05], color = 'g', linewidth = 1.0)
    plt.plot(df2[0.1], color = 'r', linewidth = 1.0)
    plt.plot(df2[0.5], color = 'b', linewidth = 1.0, linestyle = '--')
    plt.plot(df2[1], color = 'g', linewidth = 1.0, linestyle = '--')
    plt.legend([0.01, 0.05, 0.1, 0.5, 1])
    plt.title("Qualified RTDs for Atlanta", fontsize = 10)
    plt.ylabel("Probability(Solve)", fontsize = 8)
    plt.xlabel("Run-time [CPU sec]", fontsize = 8)
    plt.savefig("qrtd_ls1_atlanta.png")
