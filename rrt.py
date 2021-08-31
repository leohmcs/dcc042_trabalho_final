#!/usr/bin/env python
# coding: utf-8

# TODO:
# - considerar amostras de diferentes arvores invalidas quando estao proximas (falta testar)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.draw import line

import cv2

import time


# This implementation assumes a Turn and Go controller with constant speed
class RRT:
    class Node:
        def __init__(self, position):
            self.position = position

    def __init__(self, max_nodes, occupancy_thresh=0.8, time_step=1, speed=1):
        self.OCCUPANCY_THRESH = occupancy_thresh
        
        # RRT
        self.tree = nx.Graph()
        self.max_nodes = max_nodes
        self.robot_node = None
        self.goal_node = None

        # RRTs dos outros robos
        self.friends = None
        
        # control
        self.time_step = time_step  # seconds
        self.robot_speed = speed  # meters/second

    def control_input(self):
        return self.robot_speed * self.time_step

    def position_to_cell(self, pos, cell_size):
        return np.floor(pos / cell_size).astype('int')

    def generate_random_sample(self, m, map_size, unknow_flag):
        # random position
        rng = np.random.default_rng()
        pos = rng.choice(np.transpose(np.where(m != unknow_flag)))

        cc = np.random.choice(map_size[1])
        rr = np.random.choice(map_size[0])


        return np.array([rr, cc])
    
    # every cell with occupancy probability greater than or equal to thresh is considered occupied
    def validate_sample(self, nn, node, m, thresh):
        if node[0] >= m.shape[0] or node[1] >= m.shape[1]:
            return False

        rr, cc = line(node[1], node[0], nn[1], nn[0])

        # m_img = np.copy(m)
        # ind_unknow = np.where(m_img == -1)
        # m_img[ind_unknow] = 0.5
        # m_img = ((1 - m_img)*255).astype('uint8')

        # m_img = cv2.cvtColor(m_img, cv2.COLOR_GRAY2RGB)
        # m_img[rr, cc] = (0, 0, 255)
        # m_img[node[1], node[0]] = (0, 255, 0)

        # scale = 5
        # new_dim = (m_img.shape[1] * scale, m_img.shape[0] * scale)
        # m_resized = cv2.resize(m_img, new_dim, interpolation=cv2.INTER_AREA)

        # cv2.imshow("Validar R", m_resized)
        # cv2.imshow("Validar", m_img)
        # cv2.waitKey(10) & 0xff

        if not np.all((0 <= m[rr, cc]) & (m[rr, cc] < thresh)):
            # print(m[rr, cc])
            return False

        return True and self.compare_trees(node, 2)

    def compare_trees(self, node, thresh):
        if self.friends is None:
            return True

        trees = []
        for friend in self.friends:
            trees.append(friend.rrt)
        
        trees = nx.compose(trees)

        for n in trees:
            if np.linalg.norm(n - node) <= thresh:
                return False

        return True

    # returns the nearest node in the tree to a given node
    def nearest_neighbour(self, node):
        min_dist = np.inf
        nn = None
        for n in self.tree:
            dist = np.linalg.norm(node.position - n.position)
            if dist < min_dist:
                nn = n
                min_dist = dist

        return nn

    # returns the new node to be added to the tree. node is the 'x_rand'; nn is nodes's nearest neighbour
    def new_node(self, node, nn, control_input):
        # vetor na direcao da amostra do tamanho da distancia que o robo vai andar
        u = (node.position - nn.position) / np.linalg.norm(node.position - nn.position)
        u = u * control_input

        # print(u)
        # soma com o vizinho mais proximo para fazer a translacao
        new_node_pos = u + nn.position
        new_node = self.Node(new_node_pos)

        return new_node

    # checks if the new_node added to the tree is close enough to the goal
    def goal_reached(self, new_node, goal, err):
        dist =  np.linalg.norm(new_node - goal)
        if dist <= err:
            return True
        
        return False

    def generate_path(self, robot_pose, goal, m, unknown_flag, map_size, friends, animate=False, ani_title='RRT'):
        # animacao da geracao da arvore
        if animate:
            # TODO: botar nome do robo no titulo do plot
            plt.tight_layout()
            plt.ion()
            plt.show()

        # arvores dos outros robos
        self.friends = friends

        path_found = False
        pos = {}

        # o primeiro no e sempre a posicao do robo
        self.robot_node = self.Node(robot_pose[:2])
        self.tree.add_node(self.robot_node)
        pos[self.robot_node] = self.robot_node.position.astype("int")

        cell_size = map_size[1] / m.shape[0]
        travel_dist = self.robot_speed * self.time_step
        while self.tree.number_of_nodes() <= self.max_nodes:
            # gerar nova amostra aleatoria
            rand_pos = (self.generate_random_sample(m, map_size, unknown_flag) * cell_size).astype('int')
            x_rand = self.Node(rand_pos)

            ## TODO: DEBUGANDO
            # rand_cell = self.position_to_cell(rand_pos, cell_size)
            # m_img = np.copy(m)
            # ind_unknow = np.where(m_img == -1)
            # m_img[ind_unknow] = 0.5
            # m_img = ((1 - m_img)*255).astype('uint8')
            # m_img = cv2.cvtColor(m_img, cv2.COLOR_GRAY2RGB)
            # m_img[rand_cell[1], rand_cell[0]] = (255, 0, 0)

            # scale = 5
            # new_dim = (m_img.shape[1] * scale, m_img.shape[0] * scale)
            # m_resized = cv2.resize(m_img, new_dim, interpolation=cv2.INTER_AREA)

            # cv2.imshow("Validar R", m_resized)
            # cv2.imshow("Validar", m_img)
            # cv2.waitKey(10) & 0xff
            ## DEBUGANDO

            # obter o no mais proximo da amostra, ao qual ela sera ligada
            nn = self.nearest_neighbour(x_rand)
            if nn is None:
                continue
            
            # usar a entrada de controle para gerar o no que de fato sera adicionado na arvore
            x_new = self.new_node(x_rand, nn, travel_dist)
            if x_new is None:
                continue

            # validar o caminho entre o novo no e seu vizinho mais proximo
            nn_cell = self.position_to_cell(nn.position, cell_size)
            x_new_cell = self.position_to_cell(x_new.position, cell_size)
            # print("Positions: {}, {}".format(nn.position, x_new.position))
            # print("Cell: {}, {}".format(x_new_cell, nn_cell))
            if self.validate_sample(nn_cell, x_new_cell, m, self.OCCUPANCY_THRESH):
                self.tree.add_node(x_new)
                self.tree.add_edge(x_new, nn)
                pos[x_new] = x_new.position.astype("int")

                # print("Adicionou nó na posição: {}".format(x_new.position))

                if self.goal_reached(x_new.position, goal, 1):
                    self.goal_node = x_new
                    path_found = True
                    break
            # else:
                # print('Amostra inválida')

            if animate:
                self.animate(pos, goal)
        
        if path_found:
            path = nx.shortest_path(self.tree, self.robot_node, self.goal_node)
            self.draw_tree(path, pos)
            return path
        else:
            return None

    def reset_tree(self):
        self.tree = nx.Graph()

    def animate(self, pos, goal):
        plt.cla()
        nx.draw(self.tree, pos)
        
        plt.plot(goal[0], goal[1], 'rx')

        plt.axis('equal')
        plt.grid(which='major', axis='both', linestyle='-', color='r', linewidth=1)
        plt.plot()
        plt.draw()
        plt.pause(0.001)

    def draw_tree(self, path, pos):
        num_nodes = self.tree.number_of_nodes()
        color_map = ["green"] + ["blue"] * (num_nodes - 2) + ["red"]

        nx.draw(self.tree, pos, node_color=color_map)
        plt.axis('equal')
        
        path_edges = list(zip(path, path[1:]))
        color_map = ["green"] + ["blue"] * (len(path) - 2) + ["red"]
        nx.draw_networkx_nodes(self.tree, pos, nodelist=path, node_color=color_map)
        nx.draw_networkx_edges(self.tree, pos, edgelist=path_edges, edge_color='r')

        plt.axis('equal')
        plt.show()
