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

        # dados do robo
        self.L = 0.381
        self.r = 0.0975
        self.linear_vel = 0.2
        self.ang_vel = np.deg2rad(15)

        # handles da simulacao
        self.robot_handle = None
        self.r_motor_handle = None
        self.l_motor_handle = None

        self.clientID = sim.simxStart(ip, port, True, True, 5000, 5)
        if self.clientID != -1:
            # obter os handles
            return_code, self.robot_handle = sim.simxGetObjectHandle(self.clientID, self.name, sim.simx_opmode_oneshot_wait)
            r_motor_name = self.name[:-2] + "_rightMotor" + self.name[-2:]
            return_code, self.r_motor_handle = sim.simxGetObjectHandle(self.clientID, r_motor_name, sim.simx_opmode_oneshot_wait)
            l_motor_name = self.name[:-2] + "_leftMotor" + self.name[-2:]
            return_code, self.l_motor_handle = sim.simxGetObjectHandle(self.clientID, l_motor_name, sim.simx_opmode_oneshot_wait)

            # obter pose inicial
            self.update_pose()
        else:
            print('Falhou ao tentar conectar o robo {}'.format(name))
    
    def kinematic_model(self, v, w):
        vr = ((2*v) + (w * self.L)) / (2 * self.r)
        vl = ((2*v) - (w * self.L)) / (2 * self.r)

        return vr, vl

    def update_map(self, m):
        self.m = m

    def update_pose(self):
        return_code, position = sim.simxGetObjectPosition(self.clientID, self.robot_handle, -1, sim.simx_opmode_oneshot_wait)
        return_code, orientation = sim.simxGetObjectOrientation(self.clientID, self.robot_handle, -1, sim.simx_opmode_oneshot_wait)
        self.pose = np.array([position[0], position[1], orientation[2]])

    # path e uma lista de Nodes (classe em rrt.py)
    def controller(self, path, ang_err=0.2, dist_err=0.5):
        if self.pose is None:
            print("Pose do robo e desconhecida. Impossivel trilhar caminho.")
            return False

        current_goal_ind = 1
        while np.linalg.norm(self.pose[:2] - path[-1].position) >= dist_err:
            self.update_pose()
            current_goal = path[current_goal_ind].position

            dy = current_goal[1] - self.pose[1]
            dx = current_goal[0] - self.pose[0]
            ang = dy / dx

            ang_error = ang - self.pose[2]
            if np.abs(ang_error) >= ang_err:
                print("Angulo atual: {} \nAngulo desejado: {} \nErro: {} \n".format(np.rad2deg(self.pose[2]), np.rad2deg(ang), ang_error))
                vr, vl = self.kinematic_model(0, self.ang_vel * np.copysign(1, ang_error))
                sim.simxSetJointTargetVelocity(self.clientID, self.r_motor_handle, vr, sim.simx_opmode_oneshot_wait)
                sim.simxSetJointTargetVelocity(self.clientID, self.l_motor_handle, vl, sim.simx_opmode_oneshot_wait)
                continue
    
            dist = np.linalg.norm(self.pose[:2] - current_goal)
            if dist >= dist_err:
                print("Posicao atual: {} \nPosicao desejada: {} \nErro: {} \n".format(self.pose[:2], path[current_goal_ind].position, dist))
                vr, vl = self.kinematic_model(self.linear_vel, 0)
                sim.simxSetJointTargetVelocity(self.clientID, self.r_motor_handle, vr, sim.simx_opmode_oneshot_wait)
                sim.simxSetJointTargetVelocity(self.clientID, self.l_motor_handle, vl, sim.simx_opmode_oneshot_wait)
            else:
                current_goal_ind += 1

            print("Indo para o {}º destino: {}".format(current_goal_ind, path[current_goal_ind].position))

        sim.simxSetJointTargetVelocity(self.clientID, self.r_motor_handle, 0, sim.simx_opmode_oneshot_wait)
        sim.simxSetJointTargetVelocity(self.clientID, self.l_motor_handle, 0, sim.simx_opmode_oneshot_wait)

        print("Chegou ao destino final: {}".format(path[-1].position))

        return True

    def begin_exploration(self):
        goal_teste = np.array([8.1305, 8.4250])
        path = self.rrt.generate_path(self.pose, goal_teste, np.zeros((50, 50)), [50, 50])
        if path is None:
            self.rrt.reset_tree()
            self.begin_exploration()

        reached_goal = self.controller(path)

        if not reached_goal:
            # TODO: fazer o que nesse caso?
            pass
    