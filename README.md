# CSE-6140-Project
Implementation of 4 solvers for the TSP: branch-and-bound, MST-based approximation, simulated
annealing, and genetic algorithm.

# Files
\_ Data: input files.
  \_ ...
\_ code
  \_ requirements.txt: Python dependency modules.
  \_ __init__.py: Empty file.
  \_ tsp_main.py: Main program that calls solvers.
  \_ util.py: Utilities.
  \_ branch_and_bound.py: Branch-and-bound algorithm implementation.
  \_ MST.py: MST-approximation algorithm implementation.
  \_ simulated_annealing.py: Simulated annealing algorithm implementation.
  \_ annealing_restart.py: Utility for the simulated annealing algorithm implementation.
  \_ GA.py: Genetic algorithm implementation.
  \_ sa_experiment1b.py: Code to run the simulated annealing experiment set 1.
  \_ sa_experiment1.py: Code to run the simulated annealing experiment set 2.
\_ output: output files.
  \_ ...

# Execution
## Command-line arguments
  * -inst [string]: Filepath of a single input instance.
  * -alg [string]: Method to use.
  * -time [integer]: Cut-off time in seconds.
  * -seed [int]: Seed for random generator (optional).

## Example
python3 code/tsp_main.py -inst Data/UKansasState.tsp -alg BnB -time 600
python3 code/tsp_main.py -inst Data/UKansasState.tsp -alg Approx -time 600
python3 src/tsp_main.py -inst Data/UKansasState.tsp -alg LS1 -time 5 -seed 0
python3 src/tsp_main.py -inst Data/UKansasState.tsp -alg LS2 -time 5 -seed 0
