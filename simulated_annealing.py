import util as ut
import random
import math
import matplotlib.pyplot as plt

class SimulatedAnnealing(object):

    def __init__(self, name, coordinates, randomSeed = 0, temperature = 1e+10, alpha = 0.995, max_iterations=100000, stop_temp = 1e-8):
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
        self.coordinates = coordinates
        self.name = name
        self.path = []
        self.N = len(coordinates)
        self.nodes = list(self.coordinates.keys())
        
        # Seed random number generator
        random.seed(randomSeed)
        
        # Annealing parameters
        self.iteration = 1
        self.alpha = alpha
        self.stop = max_iterations
        self.stop_temp = stop_temp
        self.temperature = temperature

        # Solutions
        self.best_solution = None
        self.best_fit = float("Inf")
        self.fitness_list=[]

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

        TODO: Restarts
        '''
        # Start with a random tour
        if not restart:
            self.current_solution, self.current_fit = self.nearest_neighbors_tour()
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
                self.best_fit = self.current_fit
                self.best_solution = self.current_solution
            
            # Cooling for next iteration
            self.temperature *= self.alpha
            self.iteration += 1
            self.fitness_list.append(self.current_fit)

    def distance(self, n1, n2):
        '''
        Calculates the distances between nodes.

        Parameters:
        n1: Node 1 ID
        n2: Node 2 ID

        Returns: floating point Euclidean distance between two nodes
        '''
        x1,y1 = self.coordinates[n1]
        x2,y2 = self.coordinates[n2]
        return math.sqrt((x1-x2)**2 +(y1-y2)**2)

   
def simulated_annealing_tests():
    '''
    Tests out simulated annealing algorithm using default parameters.
    '''
    all_coordinates = ut.get_all_files()
    for city, coordinates in all_coordinates.items():
        sa = SimulatedAnnealing(city, coordinates, alpha = 0.999)
        sa.simulated_annealing()
        print("Results for {}:".format(city))
        ut.plotTSP(sa.best_solution, coordinates, title = "Simulated Annealing: "+city, save_path = "Plots/SA/"+city+".png", verbose = True)
    pass

if __name__ == "__main__":
    simulated_annealing_tests()