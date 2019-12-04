####################################################################################################
# CSE 6140 - Fall 2019
#   Kevin Tynes
#   Rodrigo Alves Lima
#   Shichao Liang
#   Jeevanjot Singh
####################################################################################################

'''
This file has the implementation of the MST 2-approximation algorithm.

The Minimum Spanning Tree (MST)is a 2-approximate algorithm whose
approximation guarantee hingeson the Triangle Inequality property of
the TSP problem instance.The problem is represented as a weighted
undirected graph,G=(V,E), with cities as nodes and paths between
cities as edges. Costof an edge (u,v) is defined as Euclidean distance
from city u to cityv. This algorithm chooses the city with index 0,c0,
to be the start-ing point of the cycle. Using c_0 as a root, a Minimum
Spanning Treeis constructed using Prim’s Algorithm. We greedily add
the mini-mum edge cost, and update all minimum distances as needed
untilall nodes are in set T. Using the parent node data, we backtrack
thechildren of each node starting with the root node to construct
anadjacency list. We then perform Depth First Search (DFS) to cre-ate
a euler tour, a tree traversal where nodes are added as they
arereached in the DFS algorithm. Duplicates of nodes are
removed,andc0is appended to the end to find the hamiltonian tour
solution.
'''
import util as ut
import math
import matplotlib.pyplot as plt
import numpy as np
from heapq import heappush, heappop
import time

class MST(object):

    def __init__(self, name, coordinates):
        '''
        Constructor for MST approximation algorithm.

        Parameters:
        name: String for the name of the file
        coordinates: dictionary of node IDs to coordinates.
        '''
        # Problem parameters
        self.coordinates = coordinates
        self.name = name
        self.path = []
        self.N = len(coordinates)
        self.nodes = set(self.coordinates.keys())

        self.root = 0
        self.non_root_nodes = self.nodes.copy()
        self.non_root_nodes.discard(self.root)

        # Solutions
        self.solution = [] #tour path

    def MST(self):
        ''' 
        Tour generated with greedy heuristic: 2-approximation algorithm based on MST

        Returns:
        solution: tour path
        fitness: tour length
        '''
        parent_nodes = self.prim(self.coordinates)
        child_nodes = self.parents_to_children(parent_nodes) #adjacency list
        euler_walk = []
        euler_walk = self.DFS(euler_walk,child_nodes,self.root)
        solution = self.remove_duplicates(euler_walk)
        solution.append(self.root)
        self.solution = solution

    def prim(self, coordinates):
        ''' 
        Implementation of Prim's algorithm to create a MST
        
        Input:
        coordinates: dictionary of node IDs to coordinates

        Returns:
        Pred: Predecessor data for each node in generated 
        '''
        Q = []      # Priority queue
        cost = {}   # Cost to add a vertex to the tree
        pred = {}   # Predecessor Nodes
        pred[self.root] = None

        #Initialize all other costs as distance to root
        for v in self.non_root_nodes:
            cost[v] = self.distance(self.root,v)
            heappush(Q, (self.distance(self.root,v), v))
            pred[v] = self.root
        tree = set([self.root])
        while Q:
            (weight, v) = heappop(Q)
            tree.add(v)
            for u in self.non_root_nodes:
                if u not in tree and cost[u] > self.distance(v,u):
                    cost[u] = self.distance(v,u)
                    heappush(Q, (self.distance(v,u), u))
                    pred[u] = v
        return pred

    def parents_to_children(self, parent_nodes):
        ''' 
        Converts pred datastructure to a dictionary that stores child_nodes for each node, essentially an adjacency list
        
        Input:
        parent nodes: dictionary where key is each node and value is all the parent nodes

        Returns:
        child nodes: Predecessor dictionary where key is each node and value is all the parent nodes
        '''
        child_nodes = {}
        for parent in self.nodes:
            child_nodes[parent] = []
        for child in parent_nodes:
            parent = parent_nodes[child]
            if parent is not None:
                child_nodes[parent].append(child)
        return child_nodes


    def DFS(self, visited, child_nodes, node):
        ''' 
        Depth first search implementation across the MST
        
        Input:
        visited: list of currently visited nodes, initially empty
        child_nodes: dictionary of all child_nodes for each node
        node: Starts as root node, becomes child of current node in recursive call

        Returns:
        visited: list of nodes visited with duplicates
        '''
        if node not in visited:
            visited.append(node)
            for child in child_nodes[node]:
                self.DFS(visited, child_nodes, child)
        return visited

    def remove_duplicates(self, full_walk):
        ''' 
        Depth first search implementation across the MST
        
        Input:
        visited: list of nodes visited with duplicates

        Returns:
        preorder_walk: list of nodes visited without duplicates, tour without starting node at end
        '''
        nodes_seen = set()
        preorder_walk = []
        for node in full_walk:
            if(node not in nodes_seen):
                preorder_walk.append(node)
                nodes_seen.add(node)
        return preorder_walk

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


def MST_tests():
    '''
    Tests out MST algorithm using default parameters.
    '''
    start = time.time() 
    all_coordinates = ut.get_all_files()
    for city, coordinates in all_coordinates.items():
        time_arr = []
        print('city: ',city)
        #print('coordinates: ',coordinates)
        #print(coordinates)
        print("Results for {}:".format(city))
        #print("N: ",mst.N)
        for i in range(10):
            start = time.time() 
            mst = MST(city, coordinates)
            mst.MST()
            #print('solution: ',mst.solution)
            #ut.plotTSP(mst.solution, coordinates, title = "MST: "+city, save_path = "Plots/MST/"+city+".png", verbose = True)
            end = time.time()
            time_arr.append(end-start)
        #print(time_arr)
        print('Average Time elapsed:',np.mean(time_arr))
    pass
    # end = time.time()
    # print('Time elapsed:',end-start)

def MSTSolver(inst_arg,time_arg):
    '''
        Runs MST approximation algorithm for inst_arg instance and 
        outputs cost, tour, and trace to return to the tsp_main file.

        Parameters:
        inst_arg: filepath string of a single input instance.
        time_arg: cutoff time in

        Returns:
        cost: tour cost
        tour: list of cities in tour
        trace: tuple of (time, solution)
    '''
    start = time.time()
    inst_coordinates = ut.read_tsp_file(inst_arg)
    #all_coordinates = ut.get_all_files()
    #inst_coordinates = all_coordinates[inst_arg]
    N = len(inst_coordinates)
    #print('city: ',inst_arg)
    #print('coordinates: ',inst_coordinates)
    #print("Results for {}:".format(inst_arg))
    #init data structures

    #trivial solution
    mst = MST(inst_arg, inst_coordinates)
    best_tour = list(mst.nodes)
    best_tour.append(mst.root)
    best_cost = ut.get_tour_distance(best_tour, inst_coordinates)
    trace = [(0,best_cost)]

    start = time.time()
    for i in range(N):

        #update root node
        mst.non_root_nodes.add(mst.root)
        mst.root = i
        mst.non_root_nodes.discard(mst.root)

        mst.MST()
        curr_time = time.time()
        duration = curr_time-start

        tour = mst.solution
        cost = ut.get_tour_distance(tour, inst_coordinates)
        
        if cost < best_cost and duration < time_arg:
            best_cost = cost
            best_tour = tour
            trace_line = (duration,cost)
            trace.append(trace_line)
        #print('solution: ',tour)
        #print('cost: ',cost)
    #print('solution: ',best_tour)
    #print('cost: ',best_cost)
    #print('time: ',duration)
    #print('trace: ',trace)
    return best_cost, best_tour, trace

if __name__ == "__main__":
    #MST_tests()
    MSTSolver("../Data/Cincinnati.tsp",0.00001)