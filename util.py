"""
Utilities file for CSE6140 Traveling Salesman Problem Project
Author: Shichao Liang
"""
import matplotlib.pyplot as plt
import glob
import os

def read_tsp_file(file_name):
    '''
    Reads TSP file into dictionary of coordinates

    Parameters:
    file_name: String of filename of TSP coordinate information

    Returns: Dictionary of coordinates mapping node ID to euclidean coordinates (tuple)
    '''
    f = open(file_name, "r")
    if f.mode == "r":
        contents = f.readlines()
    f.close()
    contents = [(line.rstrip()).split(" ") for line in contents[5:-1]]
    coordinates = {int(coord[0]):(float(coord[1]),float(coord[2])) for coord in contents}
    return coordinates

def plotTSP(path, coords):
    pass

def test_util(path):
    '''
    Test function
    '''
    coordinates = read_tsp_file(path)
    print(coordinates)

def get_all_files(path = 'Data/'):
    os.chdir("Data/")
    files = glob.glob('*.tsp')
    print(files)
    for file in files:
        test_util(file)

if __name__ == "__main__":
    get_all_files()