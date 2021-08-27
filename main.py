#!/usr/bin/env python

# TODO geral
# - conseguir rodar robot.begin_exploration() de cada robo em paralelo
# - incluir script do occupancy grid
# - juntar os mapas feitos por cada robo


import sim

import analisador_de_fronteira
import rrt
import robot

import time


# TODO: opcao de passar pelo argparse?
TIME_LIMIT_MINUTES = 5  # minutes

def main():
    start_time = time.time
    while time.time - start_time <= TIME_LIMIT_MINUTES * 60:
        pass


if __name__ == "__main__":
    main()