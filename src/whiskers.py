import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_trace_data(output_folder, instances = ['Atlanta','Berlin','Boston','Champaign','Cincinnati','Denver','NYC','Philadelphia', 'Roanoke', 'SanFrancisco', 'Toronto', 'UKansasState','UMissouri']):
    script_dir = os.path.dirname(__file__)
    rel_path = '../output/{}'.format(output_folder)
    path = os.path.join(script_dir, rel_path)
    time_values = {}
    fitness_values = {}
    for instance in instances:
        times = []
        fitness = []
        for filename in glob.glob(os.path.join(path, '{}*.trace'.format(instance))):
            df = pd.read_csv(filename, header = None, names = ["Time", "Fit"])
            last_time, last_val = df.iloc[-1]
            times.append(last_time)
            fitness.append(last_val)
        time_values[instance] = times
        fitness_values[instance] = fitness
    return time_values, fitness_values

def calculate_averages(output_folder, instances = ['Atlanta','Berlin','Boston','Champaign','Cincinnati','Denver','NYC','Philadelphia', 'Roanoke', 'SanFrancisco', 'Toronto', 'UKansasState','UMissouri']):
    optimal = {'Atlanta':2003763,'Berlin':7542,'Boston':893536,'Champaign':52643,'Cincinnati':277952,'Denver':100431,'NYC':1555060,'Philadelphia':1395981, 'Roanoke':655454, 'SanFrancisco':810196, 'Toronto':1176151, 'UKansasState':62962,'UMissouri':132709}
    time_vals, fitness_vals = read_trace_data(output_folder, instances)
    for instance in instances:
        times = time_vals[instance]
        fitnesses = fitness_vals[instance]
        average_time = np.mean(times)
        average_fit = np.mean(fitnesses)
        error = average_fit/optimal[instance]-1
        print("{},{},{},{}".format(instance,average_fit, error, average_time))

if __name__ == "__main__":
    calculate_averages('LS1-SA')