# Traveling Salesman Solvers
Implementation of 6 solvers for the TSP: branch-and-bound, MST-based approximation, nearest-neighbors approximation, all-nearest-neighbors approximation, simulated annealing, and genetic algorithm.

# Files and Folders
## Data: Input files.
## code:
  * requirements.txt: Python dependency modules.
  * __init__.py: Empty file.
  * tsp_main.py: Main program that calls solvers.
  * util.py: Utilities.
  * branch_and_bound.py: Branch-and-bound algorithm implementation.
  * MST.py: MST-approximation algorithm implementation.
  * simulated_annealing.py: Simulated annealing algorithm implementation.
  * annealing_restart.py: Utility for the simulated annealing algorithm implementation.
  * GA.py: Genetic algorithm implementation.
  * sa_experiment1b.py: Code to run the simulated annealing experiment set 1.
  * sa_experiment1.py: Code to run the simulated annealing experiment set 2.

## output: Output files.

# Execution
## Command-line arguments
  * -inst [string]: Filepath of a single input instance.
  * -alg [string]: Method to use.
  * -time [integer]: Cut-off time in seconds.
  * -seed [int]: Seed for random generator (optional).

## Example

python3 code/tsp_main.py -inst Data/UKansasState.tsp -alg BnB -time 5

python3 code/tsp_main.py -inst Data/UKansasState.tsp -alg Approx -time 5

python3 code/tsp_main.py -inst Data/UKansasState.tsp -alg LS1 -time 5 -seed 0

python3 code/tsp_main.py -inst Data/UKansasState.tsp -alg LS2 -time 5 -seed 0
