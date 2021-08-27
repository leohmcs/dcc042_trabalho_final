#!/usr/bin/env python

import sim
import numpy as np

class TurnAndGo:
    def __init__(self, robot, points):
        self.robot = robot
        self.path = points

        self.current_goal = 0

        self.linear_vel = 1  # m/s
        self.angular_vel = np.deg2rad(45)  # rad/s

    # vai girar ate que sua orientacao seja igual a orientation +/- err
    def turn(self, orientation, err):
        pass

    # vai seguir ate o ponto point +/- err
    def go(self, point, err):
        pass
    
    def update_robot_pose(self, pose):
        self.robot_pose = pose
    
    # retorna a entrada de controle que deve ser usada no instante
    def start(self, err):
        goal = self.path[self.current_goal]

        while np.linalg.norm(self.robot_pose[:2] - goal) > err:
            if np.abs(self.robot_pose[2]):
                pass
            
            if self.current_goal >= len(self.path):
                break  # alcancou o objetivo final
