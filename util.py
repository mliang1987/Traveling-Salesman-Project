"""
Utilities file for CSE6140 Traveling Salesman Problem Project
Author: Shichao Liang
"""

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
    contents = [ (line.rstrip()).split(" ") for line in contents[5:-1]]
    coordinates = {int(coord[0]):(float(coord[1]),float(coord[2])) for coord in contents}
    return coordinates

def test_util():
    '''
    Test function
    '''
    coordinates = read_tsp_file("Data\Atlanta.tsp")
    print(coordinates)

if __name__ == "__main__":
    test_util()