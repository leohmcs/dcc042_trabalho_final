#!/usr/bin/env python
# coding: utf-8

import sim
import numpy as np
import matplotlib.pyplot as plt

import analisador_de_fronteira
import occupancy_grid
import object
import rrt

import time

class Robot:
    def __init__(self, name, ip, port):
         # dados do robô
        self.L = 0.381
        self.r = 0.0975
        self.linear_vel = 0.7
        self.ang_vel = np.deg2rad(30)

        self.name = name
        self.pose = None
        self.rrt = rrt.RRT(10000, self.linear_vel)
        self.occ_grid = occupancy_grid.OccupancyGrid()
        self.afront = analisador_de_fronteira.AnalisadorFronteira()

        self.friends = None

        # handles da simulação
        self.robot = object.Object(self.name) 
        self.robot.handle = None
        self.r_motor_handle = None
        self.l_motor_handle = None
        self.laser = object.Object('Hokuyo_URG_04LX_UG01_ROS' + self.name[-2:])
        self.laser.handle = None

        self.laser_data_name = "hokuyo_range_data" + self.name[-2:]

        self.clientID = sim.simxStart(ip, port, True, True, 5000, 5)
        if self.clientID != -1:
            self.log_message("Conexão bem sucedida. clientID: {}".format(self.clientID))
            # obter os handles
            return_code, self.robot.handle = sim.simxGetObjectHandle(self.clientID, self.name, sim.simx_opmode_oneshot_wait)
            r_motor_name = self.name[:-2] + "_rightMotor" + self.name[-2:]
            return_code, self.r_motor_handle = sim.simxGetObjectHandle(self.clientID, r_motor_name, sim.simx_opmode_oneshot_wait)
            l_motor_name = self.name[:-2] + "_leftMotor" + self.name[-2:]
            return_code, self.l_motor_handle = sim.simxGetObjectHandle(self.clientID, l_motor_name, sim.simx_opmode_oneshot_wait)

            returnCode, self.laser.handle = sim.simxGetObjectHandle(self.clientID, self.laser.name, sim.simx_opmode_oneshot_wait)

            # obter pose inicial
            self.update_pose()
        else:
            self.log_message('Falhou ao tentar conectar com o simulador.')
            exit()
    
    def kinematic_model(self, v, w):
        vr = ((2*v) + (w * self.L)) / (2 * self.r)
        vl = ((2*v) - (w * self.L)) / (2 * self.r)

        return vr, vl

    def move(self, vr, vl):
        sim.simxSetJointTargetVelocity(self.clientID, self.r_motor_handle, vr, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(self.clientID, self.l_motor_handle, vl, sim.simx_opmode_oneshot)

    def update_pose(self):
        return_code, position = sim.simxGetObjectPosition(self.clientID, self.robot.handle, -1, sim.simx_opmode_oneshot_wait)
        return_code, orientation = sim.simxGetObjectOrientation(self.clientID, self.robot.handle, -1, sim.simx_opmode_oneshot_wait)
        self.pose = np.array([position[0], position[1], orientation[2]])

    # path é uma lista de Nodes (classe em rrt.py)
    def controller(self, path, start, duration, ang_err=0.2, dist_err=0.5):
        if self.pose is None:
            self.log_message("Pose do robô é desconhecida. Impossível trilhar caminho.")
            return False

        current_goal_ind = 1
        while np.linalg.norm(self.pose[:2] - path[-1].position) >= dist_err and current_goal_ind < len(path):
            # self.log_message("Indo para o {}º destino: {}".format(current_goal_ind, path[current_goal_ind].position))

            if time.time() - start > 60 * duration:
                self.log_message("Limite de tempo atingido.")
                break

            self.update_pose()
            current_goal = path[current_goal_ind].position

            self.mapping()


            dy = current_goal[1] - self.pose[1]
            dx = current_goal[0] - self.pose[0]
            ang = np.arctan2(dy, dx)

            ang_error = ang - self.pose[2]
            if np.abs(ang_error) >= ang_err:
                # self.log_message("Ângulo atual: {} / Ângulo desejado: {} / Erro: {} \n".format(np.rad2deg(self.pose[2]), np.rad2deg(ang), ang_error))
                vr, vl = self.kinematic_model(0, self.ang_vel * np.copysign(1, ang_error))
                self.move(vr, vl)
                continue
    
            dist = np.linalg.norm(self.pose[:2] - current_goal)
            if dist >= dist_err:
                # self.log_message("Posição atual: {} / Posição desejada: {} / Erro: {} \n".format(self.pose[:2], path[current_goal_ind].position, dist))
                vr, vl = self.kinematic_model(self.linear_vel, 0)
                self.move(vr, vl)
            else:
                current_goal_ind += 1

        self.move(0, 0)

        return True

    def begin_exploration(self, trees, duration_min, friends):
        self.log_message('Iniciou exploração.')

        self.friends = friends
        
        count = 0
        num_iter = []
        area_explored = []
        exploration_start_time = time.time()
        while time.time() - exploration_start_time <= (60 * duration_min):
            # primeira etapa: encotrar as fronteiras
            ## girar para detectar os arredores
            vr, vl = self.kinematic_model(0, self.ang_vel)
            self.move(vr, vl)

            duration = (2 * np.pi) / self.ang_vel  # tempo para completar uma volta completa
            start_time = time.time()
            while time.time() - start_time <= duration:
                self.mapping()


            self.move(0, 0)

            occ_map = np.copy(self.occ_grid.m)
            ind_unknown = np.where(occ_map == None)
            occ_map[ind_unknown] = 0
            occ_map = self.occ_grid.logodds_to_prob(occ_map)
            occ_map[ind_unknown] = -1

            self.log_message('Detectando fronteiras')
            # segunda etapa: obter caminho para fronteira
            ## escolher ponto aleatório da fronteira como goal
            frontiers = (self.afront.get_frontier_pixels(occ_map, -1)).astype('uint8')

            # não há mais fronteiras para explorar
            if np.all(frontiers == 0):
                self.log_message("Nenhuma fronteira restante.")
                return True

            rng = np.random.default_rng() 
            goal = rng.choice(np.transpose(np.where(frontiers == 1)))
            goal = goal * self.occ_grid.CELL_SIZE
            goal = goal[::-1]

            self.log_message("Indo para ponto {} da fronteira".format(goal))

            path = self.rrt.generate_path(self.pose, goal, occ_map, -1, self.occ_grid.MAP_SIZE, trees, animate=False, ani_title=self.name)
            if path is None:
                self.log_message('Caminho não encontrado.')
                self.rrt.reset_tree()
                self.begin_exploration()

            self.log_message('Caminho encontrado.')

            # terceira etapa: ir até a fronteira pelo caminho encontrado
            self.controller(path, exploration_start_time, duration_min)

            self.log_message("Chegou à fronteira: {}".format(path[-1].position))
            self.rrt.reset_tree()

            count += 1
            num_iter.append(count)
            explored = self.count_explored(occ_map, -1)
            area_explored.append(explored)

        area_explored = np.array(area_explored)
        area_explored = area_explored * self.occ_grid.CELL_SIZE**2

        plt.plot(num_iter, area_explored)
        plt.show()
        

    def count_explored(self, m, unknown_flag):
        count = 0
        it = np.nditer(m, flags=['multi_index'])
        for p in it:
            if p != unknown_flag:
                    count += 1

        return count

    def mapping(self):        
        # Garantindo que as leituras são válidas
        returnCode = 1
        while returnCode != 0:
            returnCode, range_data = sim.simxGetStringSignal(self.clientID, self.laser_data_name, sim.simx_opmode_streaming + 10)

        # lendo a posição e orientação do laser em relação ao robô
        returnCode, self.laser.position = sim.simxGetObjectPosition(self.clientID, self.laser.handle, self.robot.handle, sim.simx_opmode_oneshot_wait)        
        returnCode, self.laser.orientation = sim.simxGetObjectOrientation(self.clientID, self.laser.handle, self.robot.handle, sim.simx_opmode_oneshot_wait)

        self.update_pose()

        # recebendo as leituras do laser
        returnCode, string_range_data = sim.simxGetStringSignal(self.clientID, self.laser_data_name, sim.simx_opmode_buffer)
        raw_range_data = sim.simxUnpackFloats(string_range_data)

        # calcula a transformação do robô para o mundo e do laser para o robô
        self.robot.transforms = object.Transforms(self.pose[2], self.pose)
        self.laser.transforms = object.Transforms(self.laser.orientation[2], self.laser.position)
        
        # obtém a transformação do laser para o mundo TW_L
        TW_L = self.robot.transforms.homogeneous_tf @ self.laser.transforms.homogeneous_tf
        
        self.occ_grid.update_map(self.laser.position[:2], TW_L, raw_range_data)
        self.occ_grid.plot_map(self.name)

    def log_message(self, msg):
        print('[{}] {}'.format(self.name, msg))
