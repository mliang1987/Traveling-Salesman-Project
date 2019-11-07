"""
Utilities file for CSE6140 Traveling Salesman Problem Project
Author: Shichao Liang
"""
import matplotlib.pyplot as plt
import glob
import os
import re

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
    coordinates = {int(coord[0]):(float(coord[1]),float(coord[2])) for coord in contents}
    return coordinates

def plotTSP(path, coords):
    pass

def get_all_files(path = 'Data/'):
    '''
    '''
    all_coordinates = {}
    os.chdir("Data/")
    files = glob.glob('*.tsp')
    for file in files:
        all_coordinates[file.split(".")[0]] = read_tsp_file(file)
    return all_coordinates

if __name__ == "__main__":
    get_all_files()