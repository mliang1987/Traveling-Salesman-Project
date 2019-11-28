import math
import random
import time
import util as ut
import numpy as np
import pandas as pd

class RestartAnnealing(object):

    def __init__(self, name, coordinates, distance_matrix, default_temp = 1.0e100, default_stop = 1.0e-4, default_alpha = 0.99):
        self.name = name
        self.distance_matrix = distance_matrix
        self.N = distance_matrix.shape[0]
        self.coordinates = coordinates
        pass

    def get_tour_distance(self, tour):
        return ut.get_tour_distance(tour, self.coordinates)

    def distance(self, n1, n2):
        return self.distance_matrix[n1-1,n2-1]

if __name__ == "__main__":
    coordinates = ut.read_tsp_file("Data\\Atlanta.tsp")

    t_start = time.time();
    distance_matrix = ut.calculate_distance_matrix(coordinates)
    t_end = time.time();
    print("Numpy Distance Matrix Time Elapsed:",t_end-t_start)


    print("Arrays equal?", np.array_equal(distance_matrix.to_numpy(), distance_matrix2))

    n1 = 3
    n2 = 2

    print(distance_matrix2[n1-1,n2-1])

    ra = RestartAnnealing("Atlanta", coordinates, distance_matrix)

    tour = [8, 14, 18, 5, 20, 11, 13, 17, 3, 2, 1, 7, 9, 12, 6, 10, 16, 4, 19, 15]

    t_start = time.time();
    fit = ut.get_tour_distance(tour, coordinates)
    print(fit)
    t_end = time.time();
    print("Tour Distance Time Elapsed:",t_end-t_start)

    pass