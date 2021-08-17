#!/usr/bin/env python

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from skimage.draw import line


# This implementation assumes a Turn and Go controller with constant speed
class RRT:
    def __init__(self, robot_pose, goal, m, map_size, time_step=1, speed=1):
        self.robot_pose = robot_pose
        self.goal = goal
        self.m = m
        self.map_size = map_size

        # TODO: ver como fazer a arvore com o NetworkX
        # self.tree = nx.tree ou forest?

        # control
        self.time_step = time_step  # seconds
        self.robot_speed = speed  # meters/second

    def position_to_cell(self, pos, map_size):
        return np.floor(pos / map_size)

    def generate_random_sample(self, m):
        rr = np.random.choice(m.shape[0])
        cc = np.random.choice(m.shape[1])

        return np.array([rr, cc])

    # every cell with occupancy probability greater than thresh is considered occupied
    def validate_sample(self, robot_cell, sample_cell, m, thresh, distance):
        ang = (sample_cell[1] - robot_cell[1]) / (sample_cell[0] - robot_cell[0])

        stop_pos = np.array([distance * np.cos(ang), distance * np.sin(ang)])
        stop_cell = self.position_to_cell(stop_pos, self.map_size)

        rr, cc = line(robot_cell[1], robot_cell[0], stop_cell[1], stop_cell[0])
        
        if np.all(m[rr, cc] < thresh):
            return True

        return False
