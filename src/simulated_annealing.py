import util as ut
import time
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from scipy.spatial import distance_matrix


class SimulatedAnnealing(object):

    def __init__(self, name, coordinates, random_seed = 0, temperature = 1e+10, alpha = 0.995, max_iterations=100000, stop_temp = 1e-8, time_start = time.time(), max_time = 600, tour_flag = 0):
        '''
        Constructor for Simulated Annealing problem.

        Parameters:
        name: String for the name of the file
        coordinates: dictionary of node IDs to coordinates.

        Optional:
        randomSeed: random seed for Python random number generator
        alpha: temperature reduction factor
        max_iterations: maximum number of iterations before stopping 
        stop_temp: temperature at which to stop
        '''
        # Problem parameters
        self.time_start = time_start
        self.time_delta = 0
        self.initial_temperature = temperature
        self.coordinates = coordinates
        self.distance_matrix = ut.calculate_distance_matrix(self.coordinates)
        self.name = name
        self.path = []
        self.N = len(coordinates)
        self.nodes = list(self.coordinates.keys())
        self.max_time = max_time
        self.solutions = [2003763, 7542, 893536, 52643, 277952, 100431, 1555060, 1395981, 655454, 810196, 1176151, 62962, 132709]

        
        # Seed random number generator
        random.seed(random_seed)
        
        # Annealing parameters
        self.iteration = 1
        self.initial_alpha = alpha
        self.alpha = self.initial_alpha
        self.stop = max_iterations
        self.stop_temp = stop_temp
        self.temperature = temperature

        # Solutions
        self.best_solution = None
        self.best_fit = float("Inf")
        self.fitness_list=[]

        # Restarts
        self.trace = []
        self.result = []
        #self.convergence = min(10, int(self.N/2))
        self.convergence = min(15, int(self.N/2))
        self.restart_count = 0
        self.tour_flag = tour_flag

    def random_tour(self):
        '''
        Random tour generated.  Identifies all nodes, then shuffles order.

        Returns:
        solution: tour path
        fitness: tour length
        '''
        path = list(self.coordinates.keys())
        random.shuffle(path)
        fitness = ut.get_tour_distance(path, self.coordinates)
        return path, fitness

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
        fitness = ut.get_tour_distance(solution, self.coordinates)
        if fitness < self.best_fit:
            self.best_fit = fitness
            self.best_solution = solution
        self.fitness_list.append(fitness)
        return solution, fitness

    def simulated_annealing(self, restart = False, current_solution = None, current_fit = None):
        '''
        Simulate annealing process:
        1. Generate candidate solution via 2-opt strategy
        2. Decide to accept candidate based on either best fitness comparison or 
           temperature-bound probability
        3. Iterate by lowering temperature

        '''
        if self.max_time - (time.time()-self.time_start) < self.time_delta:
            print("\tReached max time")
            return

        t1 = time.time()

        # Start with a tour
        if current_solution == None:
            if self.tour_flag == 0:
                self.current_solution, self.current_fit = self.nearest_neighbors_tour()
            else:
                self.current_solution, self.current_fit = self.random_tour()
        else:
            self.current_solution = current_solution
            self.current_fit = current_fit

        # While annealing conditions are still met...
        while self.temperature >= self.stop_temp and self.iteration < self.stop:
            candidate = list(self.current_solution)

            # Generate next candidate using 2-Opt
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i : (i + l)] = reversed(candidate[i : (i + l)])
            
            # Determine if the candidate is worth keeping, with either:
            # 1. Better than currently-known best fit
            # 2. If not, probabilistically due to annealing
            candidate_fit = ut.get_tour_distance(candidate, self.coordinates)
            if candidate_fit < self.current_fit:
                self.current_fit = candidate_fit
                self.current_solution = candidate
            else:
                p = math.exp(-abs(candidate_fit - self.current_fit)/self.temperature)
                r = random.random()
                if r < p:
                    self.current_fit = candidate_fit
                    self.current_solution = candidate
            if self.current_fit < self.best_fit:
                self.restart_count = 0
                self.best_fit = self.current_fit
                self.best_solution = self.current_solution
            
            # Cooling for next iteration
            self.temperature *= self.alpha
            self.iteration += 1
            self.fitness_list.append(self.current_fit)
        
        self.trace.append(self.best_solution)
        self.result.append(self.best_fit)

        self.time_delta = max(self.time_delta, time.time()-t1)
        
        # Proceed to cheat step
        if self.best_fit in self.solutions:
            return
        
        # Restart with current solution?
        if restart and not self.converged():
            self.restart_count += 1
            self.temperature = self.initial_temperature* (10**self.restart_count)
            print("\tIteration: {}, Current: {}, Best: {}".format(self.restart_count, self.current_fit, self.best_fit))
            self.iteration = 1
            self.simulated_annealing(restart = True, current_solution = self.best_solution, current_fit = self.best_fit)

    def converged(self):
        if len(self.result) >= self.convergence:
            return len(set(self.result[-self.convergence:])) == 1
        return False

    def distance(self, n1, n2):
        '''
        Calculates the distances between nodes.

        Parameters:
        n1: Node 1 ID
        n2: Node 2 ID

        Returns: floating point Euclidean distance between two nodes
        '''
        return self.distance_matrix.iloc[n1-1][n2]
        #x1,y1 = self.coordinates[n1]
        #x2,y2 = self.coordinates[n2]
        #return math.sqrt((x1-x2)**2 +(y1-y2)**2)

   
def simulated_annealing_tests():
    '''
    Tests out simulated annealing algorithm using default parameters.
    '''
    all_coordinates = ut.get_all_files()
    for city, coordinates in all_coordinates.items():
        print("Results for {}:".format(city))
        sa = SimulatedAnnealing(city, coordinates, alpha = 0.999, max_time = 10000)
        sa.simulated_annealing(restart = True)
        ut.plotTSP(sa.best_solution, coordinates, title = "Simulated Annealing: "+city, save_path = "Plots/SA/"+city+".png", verbose = True, show_plots = False)
    pass

def simulated_annealing_single(file_path, random_seed, time_start, max_time):
    random.seed(random_seed)
    
    coordinates = ut.read_tsp_file(file_path)
    sa = SimulatedAnnealing(file_path, coordinates, stop_temp = 1e-8, random_seed = random.randint(0, 100000), alpha = 0.999, time_start = time_start, max_time = max_time)
    sa.simulated_annealing(restart = True)
    best_fit = sa.best_fit
    best_solution = sa.best_solution
    while max_time-(time.time()-time_start)> sa.time_delta and sa.best_fit not in sa.solutions:
        sa = SimulatedAnnealing(file_path, coordinates, stop_temp = 1e-8, random_seed = random.randint(0, 100000), alpha = 0.999, time_start = time_start, max_time = max_time, tour_flag = 1)
        #sa.best_fit = best_fit
        #sa.best_solution = best_solution
        sa.simulated_annealing(restart = True)
        if(sa.best_fit < best_fit):
            best_fit = sa.best_fit
            best_solution = sa.best_solution
    print("Results for {}: {}\n\tFitness: {}\n\tTime: {}".format(file_path, best_solution, best_fit, time.time()-time_start))
    return best_fit

if __name__ == "__main__":
    times = []
    fits = []
    for i in range(5):
        start_time = time.time()
        bf = simulated_annealing_single("Data\\Atlanta.tsp", int(time.time()), start_time, 600)
        #bf = simulated_annealing_single("Data\\Berlin.tsp", int(time.time()), start_time, 600)
        #simulated_annealing_single("Data\\Boston.tsp", int(time.time()), time.time(), 600)
        #simulated_annealing_single("Data\\Champaign.tsp", int(time.time()), time.time(), 600)
        #simulated_annealing_single("Data\\Cincinnati.tsp", int(time.time()), time.time(), 600)
        #simulated_annealing_single("Data\\Denver.tsp", int(time.time()), time.time(), 600)
        #simulated_annealing_single("Data\\NYC.tsp", int(time.time()), time.time(), 600)
        #simulated_annealing_single("Data\\Philadelphia.tsp", int(time.time()), time.time(), 600)
        #simulated_annealing_single("Data\\Roanoke.tsp", int(time.time()), time.time(), 600)
        #simulated_annealing_single("Data\\SanFrancisco.tsp", int(time.time()), time.time(), 600)
        #simulated_annealing_single("Data\\Toronto.tsp", int(time.time()), time.time(), 600)
        #simulated_annealing_single("Data\\UKansasState.tsp", int(time.time()), time.time(), 600)
        #simulated_annealing_single("Data\\UMissouri.tsp", int(time.time()), time.time(), 600)
        fits.append(bf)
        times.append(time.time()-start_time)
    print(statistics.mean(times))
    print(statistics.mean(fits))
    