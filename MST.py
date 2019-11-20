import util as ut
import math
import matplotlib.pyplot as plt
from heapq import heappush, heappop

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
        self.nodes = list(self.coordinates.keys())

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
        full_walk = []
        full_walk = self.DFS(full_walk,child_nodes,1)
        solution = self.remove_duplicates(full_walk)
        return solution

    def prim(self, coordinates):
        root = 1
        Q = []      # Priority queue
        cost = {}   # Cost to add a vertex to the tree
        pred = {}   # Predecessor Nodes
        pred[root] = None

        #Initialize all other costs as distance to root
        for v in range(2,self.N+1):
            cost[v] = self.distance(root,v)
            heappush(Q, (self.distance(root,v), v))
            pred[v] = root
        tree = set([root])
        while Q:
            (weight, v) = heappop(Q)
            tree.add(v)
            for u in range(1,self.N+1):
                if u not in tree and cost[u] > self.distance(v,u):
                    cost[u] = self.distance(v,u)
                    heappush(Q, (self.distance(v,u), u))
                    pred[u] = v
        return pred

    def parents_to_children(self, parent_nodes):
        child_nodes = {}
        for child in parent_nodes:
            parent = parent_nodes[child]
            child_nodes[parent] = child
        return child_nodes

    def DFS(self, visited, child_nodes, node):
        if node not in visited:
            visited.append(node)
            for child in child_nodes[node]:
                DFS(visited, child_nodes, child)
        return visited

    def remove_duplicates(self, full_walk):
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
    all_coordinates = ut.get_all_files()
    for city, coordinates in all_coordinates.items():
        #print(coordinates)
        mst = MST(city, coordinates)
        mst.MST()
        print("Results for {}:".format(city))
        ut.plotTSP(mst.solution, coordinates, title = "MST: "+city, save_path = "Plots/mst/"+city+".png", verbose = True)
    pass

if __name__ == "__main__":
    MST_tests()