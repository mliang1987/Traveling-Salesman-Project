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


# Configuration
LOWER_BOUND_METHOD = "MST"   # "SHORTEST_EDGES" or "MST"


class BranchAndBoundConfiguration:
  """A configuration of the BranchAndBoundSolver."""

  def __init__(self, G, N, path, lower_bound_method):
    """Initialize a configuration: a path in the graph.

    G -- [list of list of integers] A graph represented as an adjacency matrix.
    N -- [integer] Number of vertices in the graph.
    path -- [list of integers] A path represented as a sequence of vertices.
    lower_bound_method -- [string] "MST" or "SHORTEST_EDGES".
    """
    # Set attributes.
    self._G = G
    self._N = N
    self._path = path

    if self.is_solution():
      self._lower_bound = None
    elif lower_bound_method == "MST":
      # Prim's algorithm: Initialize the tree with any vertex that is not in the path.
      # Then, grow the tree one edge at a time until all the vertices that are not in the path are
      # covered.
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
    elif lower_bound_method == "SHORTEST_EDGES":
      # Calculate a lower bound of any solution derived from this configuration:
      #   cost of the path +
      #   mean of the 2 shortest edges that can be in a tour for every vertex that is not in the
      #   path
      self._lower_bound = self.get_path_cost()
      for u in range(N):
        if u in path:
          continue
        edge_weights = [
            G[u][v] for v in range(N) if u != v and (v not in path or v in (path[0], path[-1]))
        ]
        edge_weights.sort()
        self._lower_bound += (edge_weights[0] + edge_weights[1]) / 2
    else:
      raise NotImplementedError

  def expand(self, v):
    """Return an expanded configuration with the specified vertex appended to the path.

    v -- [integer] Vertex to be appended to the path.
    """
    # Check if the vertex is not in the path already.
    if v in self._path:
      raise ValueError
    return BranchAndBoundConfiguration(self._G, self._N, self._path + [v], LOWER_BOUND_METHOD)
    
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

    The branch and bound algorithm maintains a set `frontier' of partial solutions to be expanded,
    where each partial solution is a path in the graph. A lower bound `l(p)' is calculated for each
    partial solution `p': no solution derived from `p' can be better than `l(p)'. A MST-based
    approach is used to calculate this lower bound `l(p)' of a path `p': the sum of the cost of `p'
    itself, the cost of the minimum spanning tree covering the vertices not in `p', and the cost of
    connecting `p' to this minimum spanning tree.

    At each iteration, the branch and bound algorithm selects the most promising path from the
    `frontier' (i.e. the one with the lowest lower bound value) and expand it by appending an
    unvisited vertex to the end. The new partial solution `q' is only added to `frontier' if `l(q)'
    is less than a global upper bound value with the best solution found so far (i.e. a branch can
    be pruned if no solution derived from it can be better than a solution already found).
    """
    # Use a trivial tour (1-2-3-...-N-1) to set the global upper bound.
    tour = list(range(self._N))
    upper_bound = sum([self._G[i][(i + 1) % self._N] for i in range(self._N)])
    trace = []

    # Start from a configuration with a single vertex.
    frontier = [BranchAndBoundConfiguration(self._G, self._N, [0], LOWER_BOUND_METHOD)]

    # Set the start time.
    start_time = time.time()

    # Branch and bound until the frontier set is empty or the time has not expired.
    while frontier and (time.time() - start_time) < self._cutoff_time:
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
          this_solution = expanded_config.get_cycle_cost()
          if this_solution < upper_bound:
            # Log it.
            trace.append((time.time() - start_time, this_solution))
            # Update the best solution.
            upper_bound = this_solution
            tour = list(expanded_config.get_path())
            print("[%s] Found a better solution: %s" % (time.time(), upper_bound))
        elif expanded_config.get_lower_bound() < upper_bound:
          # Add to the frontier set.
          heappush(frontier, expanded_config)
    return (upper_bound, [self._vertex_to_city[v] for v in tour], trace)
