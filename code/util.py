"""
Utilities file for CSE6140 Traveling Salesman Problem Project
Author: Shichao Liang
"""
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

def read_tsp_file(file_name):
    '''
    Reads TSP file into dictionary of coordinates

    Parameters:
    file_name: String of filename of TSP coordinate information

    Returns: Dictionary of coordinates mapping node ID to euclidean coordinates (tuple)
    '''
    f = open(file_name, "r")
    contents = list(filter(None, [re.findall(r'[\d-]+\s[\d.-]+\s[\d.-]+',line) for line in f]))
    f.close()
    contents = [(line[0].rstrip()).split(" ") for line in contents]
    coordinates = {int(coord[0])-1:(float(coord[1]),float(coord[2])) for coord in contents}
    return coordinates

def get_tour_distance(tour, coords):
    '''
    Calculates the tour distance, given the tour and the dictionary of coordinates

    Parameters:
    tour: list of node IDs that represent the tour.
    coords: dictionary of coordinates for all node IDs.

    Returns: An integer value for the distance of the tour.

    Note: Python rounding rounds split floats to next even value.
    '''
    coord_array = np.asarray([coords[i] for i in tour])
    return distance_helper(coord_array)

def distance_helper(coord_array):
    '''
    Helper function for calculating the tour distance.  Can be used if coord_array is already
    generated.
    '''
    coord_array = np.vstack((coord_array,coord_array[0]))
    deltas = np.diff(coord_array, axis = 0)
    segdists = np.hypot(deltas[:,0], deltas[:,1])
    return int(np.sum(np.around(segdists, decimals = 0)))

def plotTSP(tour, coords, title = None, subtitled = True, save_path = None, show_plots = False, verbose = False):
    '''
    Generates a plot of a tour, given the tour and the dictionary of coordinates.

    Parameters:
    tour: list of node IDs that represent the tour.
    coords: dictionary of coordinates for all node IDs.
    
    Optional Parameters:
    title: string for the title of the plot. 
    subtitled: boolean for if a subtitle for the tour should be added.
    save_path: file path in which to save the figure.
    '''
    plt.figure()
    # Preprocessing tour data.
    coord_array = np.asarray([coords[i] for i in tour])
    distance = distance_helper(coord_array)
    x, y = coord_array.T

    # Title processing for plot
    if title != None:
        plt.axes([.1,.1,.8,.7])
        plt.figtext(.5,.9,title, fontsize=18, ha='center')
        if subtitled:
            plt.figtext(.5,.85,"\nTour with distance {}\n".format(distance),fontsize=8,ha='center')

    # Plot data
    plt.scatter(x,y, c='r', marker = 'o')
    arrow_scale = float(abs(max(x))-abs(min(x)))/float(60)
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = arrow_scale,
            color ='b', length_includes_head=True)
    for i in range(0,len(x)-1):
        plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = arrow_scale,
                color = 'b', length_includes_head = True)
    # Figure saving
    if verbose:
        print("\tDistance: ",distance)
    if save_path != None:
        plt.savefig(save_path)
    if show_plots:
        plt.show()

def calculate_distance_matrix(coordinates):
    '''
    Calculates distance matrix using provided coordinates.

    Parameters:
    coordinates: Dictionary of coordinates mapping n locations to (x,y) coordinates

    Returns: n x n (int) matrix of distances.
    '''
    pts = np.asarray(list(coordinates.values()))
    return np.around(np.sqrt(np.sum((pts[None, :] - pts[:, None])**2, -1))).astype(int)

def get_all_files(path = 'Data/'):
    '''
    Given a folder path, reads all TSP files in the directory into a dictionary of coordinates.

    Parameters:
    path: folder path of the TSP data files.

    Returns: Dictionary of coordinates mapping file name to dictionary of coordinates

    TODO: Currently hard-coded paths, need to conform to spec on project desc.
    '''
    all_coordinates = {}
    os.chdir("../Data/")
    files = glob.glob('*.tsp')
    for file in files:
        all_coordinates[file.split(".")[0]] = read_tsp_file(file)
    return all_coordinates

if __name__ == "__main__":
    all_coordinates = (get_all_files())
    for city in all_coordinates.keys():
        dm = calculate_distance_matrix_scipy(all_coordinates[city])
        print(dm)
        print(dm.iloc[3][2])