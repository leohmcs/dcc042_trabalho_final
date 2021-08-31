#!/usr/bin/env python
# coding: utf-8

# TODO geral
# - conseguir rodar robot.begin_exploration() de cada robo em paralelo
# - incluir script do occupancy grid
# - juntar os mapas feitos por cada robo

# Notas
# - use o nx.compose() para juntar as arvores de varios robos e envie o resultado como parametros para rrt


import sim
import numpy as np

import analisador_de_fronteira
import rrt
import robot

import time
from multiprocessing import Process


# TODO: opcao de passar pelo argparse?
TIME_LIMIT_MINUTES = 5  # minutes

# Funções Auxiliares
## Retorna uma matriz com duas colunas: [angle, dist]
def format_laser_data(range_data, scan_range, step_size, max_sensor_range):
    laser_data = []

    range_data = np.asarray(range_data)
    pts = np.floor(scan_range/step_size)
    
    angle =- scan_range*0.5
    for i in range(pts):
        
        dist = range_data[i]        
        if dist <= 0:
            dist = max_sensor_range
        
        laser_data.append([angle, dist])
            
        angle=angle+step_size
        
    return np.array(laser_data)


def main():
    # results = [False] * num_robots
    start_time = time.time()
    while time.time() - start_time <= TIME_LIMIT_MINUTES * 60:

        # iniciar exploracao de cada robo em paralelo
        r1 = robot.Robot('Pioneer_p3dx#2', '127.0.0.1', 19999)
        r2 = robot.Robot('Pioneer_p3dx#0', '127.0.0.1', 20000)
        r3 = robot.Robot('Pioneer_p3dx#1', '127.0.0.1', 20001)
        
        finished = r3.begin_exploration(None, 5)

        # # TODO: passar as arvores dos outros robos como argumento
        # p1 = Process(target=r1.begin_exploration, args=(None,))
        # p1.start()

        # p2 = Process(target=r2.begin_exploration, args=(None,))
        # p2.start()

        # p3 = Process(target=r3.begin_exploration, args=(None,))
        # p3.start()

        # p1.join()
        # p2.join()
        # p3.join()


if __name__ == "__main__":
    main()