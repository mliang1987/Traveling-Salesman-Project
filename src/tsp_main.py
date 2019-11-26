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

import sys
import time

from branch_and_bound import BranchAndBoundSolver



def main():
  # Parse command line arguments.
  cli_args = dict((sys.argv[1 + i], sys.argv[2 + i]) for i in range(0, len(sys.argv[1:]), 2))
  inst_arg = cli_args.get("-inst", None)
  alg_arg = cli_args.get("-alg", None)
  time_arg = cli_args.get("-time", None)
  seed_arg = cli_args.get("-seed", None)
  if not inst_arg or not alg_arg or not time_arg or \
      (not seed_arg and alg_arg in ("Approx", "LS1", "LS2")):
    print("Invalid arguments.")
    sys.exit(1)

  # Solve the problem instance.
  print("[%s] Started execution.\n\tinst = %s\n\talg = %s\n\ttime = %s\n\tseed = %s" %
      (time.time(), inst_arg, alg_arg, time_arg, seed_arg))
  if (alg_arg == "BnB"):
    solver = BranchAndBoundSolver(inst_arg, time_arg)
  elif (alg_arg == "Approx"):
    # TODO
    raise NotImplementedError
  elif (alg_arg == "LS1"):
    # TODO
    raise NotImplementedError
  elif (alg_arg == "LS2"):
    # TODO
    raise NotImplementedError
  solver.solve()


if __name__ == "__main__":
  main()
