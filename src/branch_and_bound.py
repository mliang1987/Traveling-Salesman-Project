 ####################################################################################################
# CSE 6140 - Fall 2019
#   Rodrigo Alves Lima
#   Shichao Liang
#   Jeevanjot Singh
#   Kevin Tynes
####################################################################################################

from heapq import heappush, heappop
import time

from solver import Solver


class BranchAndBoundConfiguration:
  """A configuration of the BranchAndBoundSolver."""

  def __init__(self, G, N, path):
    """Initialize a configuration: a path in the graph.

    G -- [list of list of integers] A graph represented as an adjacency matrix.
    N -- [integer] Number of vertices in the graph.
    path -- [list of integers] A path represented as a sequence of vertices.
    """
    # Set attributes.
    self._G = G
    self._N = N
    self._path = path

    if self.is_solution():
      self._lower_bound = None
    else:
      # Prim's algorithm: Initialize the tree with any vertex that is not in the path. Then, grow the
      # tree one edge at a time until all the vertices that are not in the path are covered.
      root = None
      Q = []      # Priority queue
      cost = {}   # Cost to add a vertex to the tree
      mst_cost = 0
      for v in range(N):
        if root is None and v not in path:
          root = v
        elif root is not None and v not in path:
          cost[v] = G[root][v]
          heappush(Q, (G[root][v], v))
      tree = set([root]) if root is not None else None
      while Q:
        (weight, v) = heappop(Q)
        if v not in tree:
          mst_cost += weight
          tree.add(v)
          for u in range(N):
            if u not in path and u not in tree and cost[u] > G[v][u]:
              cost[u] = G[v][u]
              heappush(Q, (G[v][u], u))

      # Calculate a lower bound of any solution derived from this configuration:
      #   cost of the path +
      #   cost of the minimum spanning tree covering the vertices that are not in the path +
      #   minimum cost of joining the path ends to that minimum spanning tree
      self._lower_bound = self.get_path_cost() + mst_cost + \
          min([G[path[0]][v] for v in range(N) if v not in path]) if len(path) > 0 else 0 + \
          min([G[path[-1]][v] for v in range(N) if v not in path]) if len(path) > 1 else 0

  def expand(self, v):
    """Return an expanded configuration with the specified vertex appended to the path.

    v -- [integer] Vertex to be appended to the path.
    """
    # Check if the vertex is not in the path already.
    if v in self._path:
      raise ValueError
    return BranchAndBoundConfiguration(self._G, self._N, self._path + [v])
    
  def is_solution(self):
    """Return True if the path is a solution. Return False, otherwise."""
    # Only need to check the length because the configuration expansion assesses the feasibility.
    return len(self._path) == self._N

  def get_path_cost(self):
    """Return the cost of the path: sum of the costs of its edges."""
    return sum([self._G[self._path[i - 1]][self._path[i]] for i in range(1, len(self._path))])

  def get_cycle_cost(self):
    """Return the cost of the cycle: sum of the costs of its edges."""
    return self.get_path_cost() + self._G[self._path[-1]][self._path[0]]

  def get_lower_bound(self):
    """Return the lower bound."""
    return self._lower_bound

  def get_path(self):
    """Return the path."""
    return self._path

  def __lt__(self, other):
    """Return True if this configuration is more promising than the other configuration. Return
    False, otherwise.

    other -- [Configuration] A configuration to be compared against.
    """
    return self._lower_bound < other._lower_bound


class BranchAndBoundSolver(Solver):
  """Implementation of the branch and bound algorithm using the MST-based lower bound function."""

  def solve(self):
    """Return an exact solution to the TSP problem.

    TODO: Detailed explanation.
    """
    # Set an undefined global upper bound.
    upper_bound = None

    # Start from an empty configuration.
    frontier = [BranchAndBoundConfiguration(self._G, self._N, [])]

    # Branch and bound until the frontier set is empty.
    while frontier:
      # Fetch the most promising configuration.
      config = heappop(frontier)

      # Expand configuration by appending a vertex to the path.
      for v in range(self._N):
        try:
          expanded_config = config.expand(v)
        except ValueError:
          # Expanded configuration is not valid.
          continue
        if expanded_config.is_solution():
          # Update the global upper bound, if needed.
          upper_bound = min(upper_bound, expanded_config.get_cycle_cost()) \
              if upper_bound is not None else expanded_config.get_cycle_cost()
          print("[%s] Found a solution.\n\tupper bound = %s" % (time.time(), upper_bound))
        elif upper_bound is None or expanded_config.get_lower_bound() < upper_bound:
          # Add to the frontier set.
          heappush(frontier, expanded_config)
    print("[%s] Done.\n\tcost = %s" % (time.time(), upper_bound))
