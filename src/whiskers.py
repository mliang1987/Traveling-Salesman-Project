####################################################################################################
# CSE 6140 - Fall 2019
#   Rodrigo Alves Lima
#   Shichao Liang
#   Jeevanjot Singh
#   Kevin Tynes
####################################################################################################


"""
Generates box and whisker plots for specified instances.  In the main method: specify instances 
desired and the name of the folder containing the traces.
"""


import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 

def read_trace_data(output_folder, instances = ['Atlanta','Berlin','Boston','Champaign','Cincinnati','Denver','NYC','Philadelphia', 'Roanoke', 'SanFrancisco', 'Toronto', 'UKansasState','UMissouri']):
    '''
    Reads trace data from a specified folder.  
    Optional instances array are particular instances in which we're interested.
    '''
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
    '''
    Calculates table metrics for instances.  
    Optional instances array are particular instances in which we're interested.
    '''
    optimal = {'Atlanta':2003763,'Berlin':7542,'Boston':893536,'Champaign':52643,'Cincinnati':277952,'Denver':100431,'NYC':1555060,'Philadelphia':1395981, 'Roanoke':655454, 'SanFrancisco':810196, 'Toronto':1176151, 'UKansasState':62962,'UMissouri':132709}
    time_vals, fitness_vals = read_trace_data(output_folder, instances)
    time_averages = []
    for instance in instances:
        times = time_vals[instance]
        time_averages.append(times)
        fitnesses = fitness_vals[instance]
        average_time = np.mean(times)
        average_fit = np.mean(fitnesses)
        error = average_fit/optimal[instance]-1
        print("{},{},{},{}".format(instance,average_fit, error, average_time))
    return time_averages

def box_plot(time_averages, name, instances = ['Atlanta','Berlin','Boston','Champaign','Cincinnati','Denver','NYC','Philadelphia', 'Roanoke', 'SanFrancisco', 'Toronto', 'UKansasState','UMissouri']):
    '''
    Given the time averages from calculate_averages(), and the name of folder, generate a boxplot.
    '''
    n = len(time_averages[0])
    instance_col = []
    time_col = []
    for i, instance in enumerate(instances):
        instance_col+=[instance]*n
        time_col+=time_averages[i]
    df = pd.DataFrame(columns = ["Instance", "Time"])
    df["Instance"] = instance_col
    df["Time"] = time_col
    df.boxplot(column='Time', by='Instance')
    plt.title(instances, fontsize = 12)
    plt.suptitle("Time Data for Specificed Instances", fontsize = 14)
    plt.ylabel("Time (s)", fontsize = 10)
    plt.savefig('boxplot_{}.png'.format(name))

if __name__ == "__main__":
    instances = ['Atlanta', 'Champaign']
    name = 'LS1-SA'
    time_averages = calculate_averages(name, instances = instances)
    box_plot(time_averages, name, instances = instances)
