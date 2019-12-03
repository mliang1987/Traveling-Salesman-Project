import simulated_annealing as sa
import random
import time
import math
import pandas as pd
import numpy as np

class TimeoutException(Exception): pass

def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

if __name__ == "__main__":
	optimal = 2003763
	file_path = "Data/Atlanta.tsp"
	times = [0.01, 0.05, 0.1, 0.5, 1]
	qualities = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
	df = pd.DataFrame(index = qualities, columns = times)
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
		df[max_time] = p_values
	print(df)
