#!/usr/bin/env python

# TODO (ideia geral)
# - terminar o controlador
# - incluir a parte de detectar fronteira e definir o objetivo em begin_exploration()

import sim
import numpy as np

class Robot:
    def __init__(self, name, rrt, m, port, ip):
        self.name = name
        self.pose = None
        self.rrt = rrt
        self.m = m

        self.robot_handle = None
        self.clientID = sim.simxStart(ip, port, True, True, 5000, 5)
        if self.clientID != -1:
            return_code, self.robot_handle = sim.simxGetObjectHandle(self.clientID, self.name, sim.simx_opmode_oneshot_wait)
        else:
            print('Falhou ao tentar conectar o robo {}'.format(name))
    
    def update_map(self, m):
        self.m = m

    def update_pose(self):
        position = sim.simxGetObjectPosition(self.clientID, self.robot_handle, -1, sim.simx_opmode_oneshot_wait)
        orientation = sim.simxGetObjectOrientation(self.clientID, self.robot_handle, -1, sim.simx_opmode_oneshot_wait)
        self.pose = np.array(position[:2], orientation[2])

    # path e uma lista de Nodes (classe em rrt.py)
    def controller(self, path, ang_err=0.2, dist_err=0.5):
        if self.pose is None:
            print("Pose do robo e desconhecida. Impossivel trilhar caminho.")
            return False

        current_goal_ind = 1
        while np.linalg.norm(self.pose - path[-1]) > dist_err:
            self.update_pose()
            current_goal = path[current_goal_ind]
            ang = (current_goal[1] - self.pose[1]) / (current_goal[0] - self.pose[0])
            if np.abs(self.pose[2] - ang) >= ang_err:
                # TODO: rodar o robo
                continue
    
            dist = np.linalg.norm(self.pose[:2] - current_goal)
            if dist >= dist_err:
                # TODO: andar para frente
                pass

    def begin_exploration(self):
        path = self.rrt.generate_path(np.array([1.5695, 1.2750, 0]), np.array([8.1305, 8.4250]), np.zeros((50, 50)), [50, 50])
        if path is None:
            self.rrt.reset_tree()
            self.begin_exploration()

        reached_goal = self.controller(path)
        if not reached_goal:
            # TODO: fazer o que nesse caso?
            pass
    