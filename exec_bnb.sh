#!/bin/bash

INSTANCES="Atlanta Berlin Boston Champaign Cincinnati Denver NYC Philadelphia Roanoke SanFrancisco Toronto UKansasState UMissouri"

for INSTANCE in $INSTANCES
do
  python3 src/tsp_main.py -inst Data/$INSTANCE.tsp -alg BnB -time 21600
done
