####################################################################################################
# CSE 6140 - Fall 2019
#   Rodrigo Alves Lima
#   Shichao Liang
#   Jeevanjot Singh
#   Kevin Tynes
####################################################################################################

import decimal
import math
import time


class Solver:
  """Define a Solver interface and implement utilities common to all solvers."""

  def __init__(self, filename, cutoff_time):
    """Initialize a Solver.

    filename -- [string] Filepath of a single input instance.
    cutoff_time -- [int] Cut-off time.
    """ 
    # Parse the input file.
    with open(filename) as input_file:
      # List of tuples representing cities, where the 1st element is the city's
      # id, 2nd is the city's x coordinate, and 3rd is the city's y coordinate.
      cities = []
      # Iterate over the lines, excluding the header.
      for input_line in input_file.readlines()[5:-1]:
        values_str = input_line.split()
        cities.append(
            (int(values_str[0]), float(values_str[1]), float(values_str[2]))
        )

    # Set attributes.
    self._cutoff_time = cutoff_time
    self._N = len(cities)

    # Build the graph by calculating Euclidean distances between cities (rounded to the nearest
    # integer).
    self._G = [[None for j in range(self._N)] for i in range(self._N)]
    for city_a in range(self._N):
      self._G[city_a][city_a] = 0
      for city_b in range(city_a + 1, self._N):
        self._G[city_a][city_b] = self._G[city_b][city_a] = int(decimal.Decimal(math.sqrt((
            cities[city_a][1] - cities[city_b][1]) ** 2 +
            (cities[city_a][2] - cities[city_b][2]) ** 2
        )).quantize(decimal.Decimal(1), rounding=decimal.ROUND_HALF_UP))
    print("[%s] Built the graph.\n\tdimension = %s" % (time.time(), self._N))

  def solve(self):
    raise NotImplementedError
