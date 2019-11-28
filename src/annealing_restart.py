import math
import random
import time
import util as ut
import numpy as np
import pandas as pd
import copy

class RestartAnnealing(object):

    def __init__(self, name, 
                 coordinates, 
                 distance_matrix, 
                 time_start = time.time(), 
                 time_max = 600, 
                 default_temp = 1.0e10, 
                 default_stop = 1.0e-8, 
                 default_alpha = 0.999):
        
        # Problem Parameters
        self.name = name
        self.distance_matrix = distance_matrix
        self.coordinates = coordinates
        self.N = distance_matrix.shape[0]
        self.nodes = list(self.coordinates.keys())
        self.time_start = time_start
        self.time_max = time_max


        # Annealing Parameters
        self.default_temp = default_temp
        self.default_stop = default_stop
        self.default_alpha = default_alpha
        self.best_fits = []
        self.best_tours = []

        # Trace information
        self.fitness_list = []
        self.tour_list = []

    def get_tour_distance(self, tour):
        '''
        Calculates tour distance.

        Parameter (tour) - List of cities visited in order.
        Returns: integer for distance of the tour
        '''
        distances = [self.distance(tour[i], tour[(i+1)%self.N]) for i in range(self.N)]
        return sum(distances)

    def distance(self, n1, n2):
        '''
        Determines the distance between two nodes.
        
        Parameters
        n1 - label for node 1
        n2 - label for node 2

        Returns:
        distance - float distance between n1 and n2
        '''
        return self.distance_matrix[n1,n2]

    def random_tour(self):
        '''
        Random tour generated.  Identifies all nodes, then shuffles order.

        Returns:
        solution: tour path
        fitness: tour length
        '''
        path = list(self.coordinates.keys())
        random.shuffle(path)
        return path

    def nearest_neighbors_tour(self):
        '''
        Tour generated with greedy heuristic: Picks a random starting node, then appends
        next nearest neighbor.

        Returns:
        solution: tour path
        fitness: tour length
        '''
        solution = []
        unassigned_nodes = set(self.nodes)
        node = random.choice(self.nodes)
        solution.append(node)
        unassigned_nodes.remove(node)
        while unassigned_nodes:
            next = min(unassigned_nodes, key=lambda x: self.distance(node, x))
            unassigned_nodes.remove(next)
            solution.append(next)
            node = next
        return solution

    def compute_swap_indices(self, index):
        previous = (index - 1 + self.N) % self.N
        next = (index + 1) % self.N
        return previous, next

    def distance_swap(self, tour, n1, n2):
        """Compute the distance inferred by the two given indices."""
        a = min(n1, n2)
        b = max(n1, n2)
    
        previous_a, next_a = self.compute_swap_indices(a)
        previous_b, next_b = self.compute_swap_indices(b)
  
        distances = []
        # These two distances are common to the two sub-cases
        #distances.append(tour[previous_a].distance_to_city_in_km(tour[a]))
        #distances.append(tour[b].distance_to_city_in_km(tour[next_b]))
        distances.append(self.distance(tour[previous_a], tour[a]))
        distances.append(self.distance(tour[b], tour[next_b]))
        
        if a == previous_b:
            # B is following A in the list: the distance between A and B must not
            # be counted twice.
            # ---x---A---B---x---
            #distances.append(tour[a].distance_to_city_in_km(tour[b]))
            distances.append(self.distance(tour[a], tour[b]))
        else:
            # B is not following A in the list: all distances must be counted
            # ---x---A---x--- ... ---x---B---x---
            #distances.append(tour[a].distance_to_city_in_km(tour[next_a]))
            #distances.append(tour[previous_b].distance_to_city_in_km(tour[b]))
            distances.append(self.distance(tour[a], tour[next_a]))
            distances.append(self.distance(tour[previous_b], tour[b]))
        return sum(distances)

    def anneal(self, tour = None, tour_type = 0, 
               temperature = None, 
               stop = None, 
               alpha = None, 
               iterations = 1 ):
        '''
        Simulated annealing step.  Considers neighboring solutions, and accepts candidates for
        further search if they are better than the currently-known best solution, or, accepts 
        worse candidates with decreasing probability (geometric).

        Keyword Parameters:
        tour: (list) Starting tour.  If left as None, will either generate a random tour or nearest-neighbors tour, depending on tour_type.
        tour-type: (int: 0 or 1) Type of starting tour to generate. 0-random, 1-nearest-neighbors
        temperature: (float) Starting temperature for annealing. If None, will use default values.
        stop: (float) Ending temperature for annealing.  If None, will use default values.
        alpha: (float) Geometric change in temperature per iteration. If None, will use default values.
        iterations: (int) The number of iterations to anneal.
        '''
        # Setup initial annealing paramaters.
        if tour == None:
            if tour_type == 0:
                tour = self.random_tour()
            else:
                tour = self.nearest_neighbors_tour()
        if temperature == None:
            temperature = self.default_temp
        if stop == None:
            stop = self.default_stop
        if alpha == None:
            alpha = self.default_alpha

        # Initial guesses for tour and fitness
        current_tour = list(tour)
        current_fit = self.get_tour_distance(current_tour)
        self.best_fits.append(current_fit)
        self.best_tours.append(current_tour)

        while temperature > stop:
            candidate_tour = self.candidate_generator_b(current_tour)
            candidate_fit = self.get_tour_distance(candidate_tour)

            delta = candidate_fit - current_fit
            if delta < 0 or math.exp(-delta/temperature) > random.random():
                current_fit = candidate_fit
                current_tour = candidate_tour
            if current_fit < self.best_fits[-1]:
                self.best_fits.append(current_fit)
                self.best_tours.append(current_tour)

            temperature = alpha*temperature

        # Return end-result of annealing step.
        return self.best_tours, self.best_fits

    def candidate_generator_a(self, current_tour):
        a, b = random.sample(range(self.N-1),2)
        a += 1
        b += 1
        candidate_tour = list(current_tour)
        candidate_tour[a], candidate_tour[b] = current_tour[b], current_tour[a]
        return candidate_tour

    def candidate_generator_b(self, current_tour):
        candidate_tour = list(current_tour)
        l = random.randint(2, self.N - 1)
        i = random.randint(0, self.N - l)
        candidate_tour[i : (i + l)] = reversed(candidate_tour[i : (i + l)])
        return candidate_tour

if __name__ == "__main__":
    time_start = time.time()
    coordinates = ut.read_tsp_file("Data\\Atlanta.tsp")
    distance_matrix = ut.calculate_distance_matrix(coordinates)
    ra = RestartAnnealing("Atlanta", coordinates, distance_matrix, time_start = time_start)
    
    solution = list(np.asarray([6, 10, 16, 4, 19, 15, 8, 14, 18, 5, 20, 11, 13, 17, 3, 2, 1, 7, 9, 12])-1)

    t_start = time.time()
    tour, fit = ra.anneal(tour_type=1)
    t_end = time.time();
    t = t_end-t_start
    print(tour, fit)
    print(t)
    
    pass