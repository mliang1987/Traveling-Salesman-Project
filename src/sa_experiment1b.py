import simulated_annealing as sa
import random
import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
	optimal = 2003763
	file_path = "Data/Atlanta.tsp"
	times = [0.01, 0.05, 0.1, 0.5, 1]
	qualities = [0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
	df2 = pd.DataFrame(index = qualities, columns = times)
	for max_time in times:
		print("Running time",max_time)
		p_values = []
		for quality in qualities:
			print("\tRunning quality",quality)
			test_quality = math.floor((quality+1)*optimal)
			experiment = []
			for i in range(50):
				sol, _, _ = sa.simulated_annealing_single(file_path, random.randint(1,100), time.time(), max_time, test_quality = test_quality)
				experiment.append(sol<=test_quality)
			t_count = experiment.count(True)
			p = t_count / len(experiment)
			p_values.append(p)
		df2[max_time] = p_values
	
	plt.figure()
	plt.gcf().subplots_adjust(bottom=0.2)
	plt.axis([0,1.2,-0.1,1.1])
	plt.plot(df2[0.01], color = 'b', linewidth = 1.0)
	plt.plot(df2[0.05], color = 'g', linewidth = 1.0)
	plt.plot(df2[0.1], color = 'r', linewidth = 1.0)
	plt.plot(df2[0.5], color = 'b', linewidth = 1.0, linestyle = '--')
	plt.plot(df2[1], color = 'g', linewidth = 1.0, linestyle = '--')
	plt.legend([0.01, 0.05, 0.1, 0.5, 1])
	plt.title("Solution Quality Distributions for Atlanta", fontsize = 10)
	plt.ylabel("Probability(Solve)", fontsize = 8)
	plt.xlabel("Relative Solution Quality [%]", fontsize = 8)
	plt.savefig("sqd_ls1_atlanta.png")
