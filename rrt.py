#!/usr/bin/env python

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.draw import line

import time


# This implementation assumes a Turn and Go controller with constant speed
class RRT:
    class Node:
        def __init__(self, position):
            self.position = position


    def __init__(self, max_nodes, occupancy_thresh=80, time_step=1, speed=1):
        self.OCCUPANCY_THRESH = occupancy_thresh
        
        # RRT
        self.tree = nx.Graph()
        self.max_nodes = max_nodes
        self.robot_node = None
        self.goal_node = None

        # control
        self.time_step = time_step  # seconds
        self.robot_speed = speed  # meters/second

    def control_input(self):
        return self.robot_speed * self.time_step

    def position_to_cell(self, pos, cell_size):
        return np.floor(pos / cell_size).astype('int')

    def generate_random_sample(self, map_size):
        # random position
        x = np.random.choice(map_size[0])
        y = np.random.choice(map_size[1])

        print("Posicao aleatoria gerada: {}, {}".format(x, y))
        return np.array([x, y])

    # every cell with occupancy probability greater than or equal to thresh is considered occupied
    def validate_sample(self, nn, node, m, thresh):
        if node[0] >= m.shape[0] or node[1] >= m.shape[1]:
            return False

        rr, cc = line(nn[1], nn[0], node[1], node[0])

        if np.all((0 <= m[rr, cc]) & (m[rr, cc] < thresh)):
            return True

        return False

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
        u = (node.position - nn.position) / np.linalg.norm(node.position - nn.position) * control_input

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

    def generate_path(self, robot_pose, goal, m, map_size, animate=True, ani_interval=1000):
        # TODO: fazer animacao
        # ani = FuncAnimation(plt.gcf(), self.animate, interval=ani_interval)

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
            rand_pos = self.generate_random_sample(map_size)
            x_rand = self.Node(rand_pos)

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
            if self.validate_sample(nn_cell, x_new_cell, m, self.OCCUPANCY_THRESH):
                self.tree.add_node(x_new)
                self.tree.add_edge(x_new, nn)
                pos[x_new] = x_new.position.astype("int")

                if self.goal_reached(x_new.position, goal, 1):
                    self.goal_node = x_new
                    path_found = True
                    break
        
        if path_found:
            print('Encontrou o caminho apos gerar {} nos.'.format(self.tree.number_of_nodes() - 1))
            path = nx.shortest_path(self.tree, self.robot_node, self.goal_node)
            self.draw_tree(path, pos)
            return path
        else:
            print('Caminho nao encontrado.')
            return None

    def reset_tree(self):
        self.tree = nx.Graph()

    def animate(self, i):
        nx.draw(self.tree)
        plt.show()

    def draw_tree(self, path, pos):
        num_nodes = self.tree.number_of_nodes()
        color_map = ["green"] + ["blue"] * (num_nodes - 2) + ["red"]

        nx.draw(self.tree, pos, node_color=color_map)
        plt.axis('equal')
        # plt.show()

        path_edges = list(zip(path, path[1:]))
        color_map = ["green"] + ["blue"] * (len(path) - 2) + ["red"]
        nx.draw_networkx_nodes(self.tree, pos, nodelist=path, node_color=color_map)
        nx.draw_networkx_edges(self.tree, pos, edgelist=path_edges, edge_color='r')

        plt.axis('equal')
        plt.show()
