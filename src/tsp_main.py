####################################################################################################
# CSE 6140 - Fall 2019
#   Rodrigo Alves Lima
#   Shichao Liang
#   Jeevanjot Singh
#   Kevin Tynes
####################################################################################################

"""
Command-line arguments:
  -inst [string]: Filepath of a single input instance.
  -alg [string]: Method to use.
  -time [integer]: Cut-off time in seconds.
  -seed [int]: Seed for random generator (optional).
"""

import os
import sys
import time

from branch_and_bound import BranchAndBoundSolver
from MST import MSTSolver
import simulated_annealing as sa



def main():
  time_init = time.time()
  # Parse command line arguments.
  cli_args = dict((sys.argv[1 + i], sys.argv[2 + i]) for i in range(0, len(sys.argv[1:]), 2))
  inst_arg = cli_args.get("-inst", None)
  alg_arg = cli_args.get("-alg", None)
  time_arg = cli_args.get("-time", None)
  seed_arg = cli_args.get("-seed", None)
  if not inst_arg or not alg_arg or not time_arg or \
      (not seed_arg and alg_arg in ("LS1", "LS2")):
    print("Invalid arguments.")
    sys.exit(1)

  # Solve the problem instance.
  print("[%s] Started execution.\n\tinst = %s\n\talg = %s\n\ttime = %s\n\tseed = %s" %
      (time_init, inst_arg, alg_arg, time_arg, seed_arg))
  if (alg_arg == "BnB"):
    solver = BranchAndBoundSolver(inst_arg, int(time_arg))
    cost, tour, trace = solver.solve()
  elif (alg_arg == "Approx"):
    cost, tour, trace = MSTSolver(inst_arg, int(time_arg)) #Not using solver class
  elif (alg_arg == "LS1"):
    # Simulated Annealing
    cost, tour, trace = sa.simulated_annealing_single(inst_arg, int(seed_arg), time_init, float(time_arg))
  elif (alg_arg == "LS2"):
    # Genetic
    cost, tour, trace = ga.ga_single(inst_arg, float(time_arg), int(seed_arg))
  
  inst = os.path.basename(inst_arg).split('.')[0]
  with open("%s_%s_%s.sol" % (inst, alg_arg, time_arg) if seed_arg is None else \
      "%s_%s_%s_%s.sol" % (inst, alg_arg, time_arg, seed_arg), 'w') as solution_file:
    solution_file.write("%s\n" % (cost))
    solution_file.write(",".join([str(city - 1) for city in tour]))
    solution_file.write("\n")
  with open("%s_%s_%s.trace" % (inst, alg_arg, time_arg) if seed_arg is None else \
      "%s_%s_%s_%s.trace" % (inst, alg_arg, time_arg, seed_arg), 'w') as trace_file:
    trace_file.write(
        "\n".join(["%.2f, %s" % (trace_record[0], trace_record[1]) for trace_record in trace]))
    trace_file.write("\n")


if __name__ == "__main__":
  main()
