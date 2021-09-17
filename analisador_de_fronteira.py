# coding: utf-8

import numpy as np
import cv2

import math
import time

class AnalisadorFronteira:
    def __init__(self):
        # Map resolution
        self.height = 0
        self.width = 0

    # retorna os indices dos pixels vizinhos
    def get_neighbours(self, map_shape, element_ind):
        rr, cc = element_ind
        row_ind = np.array([[rr - 1]*3, [rr]*3, [rr + 1]*3]).reshape(9,)
        col_ind = np.array([cc - 1, cc, cc + 1]*3)

        ind = np.vstack((row_ind, col_ind))

        # remove o proprio pixel
        ind = np.delete(ind, 4, axis=1)

        # remove os indices que nao existem
        rows2delete = np.where((ind[0] < 0) | (ind[0] >= map_shape[0]))
        ind = np.delete(ind, rows2delete, axis=1)

        cols2delete = np.where((ind[1] < 0) | (ind[1] >= map_shape[1]))
        ind = np.delete(ind, cols2delete, axis=1)

        return (ind[0], ind[1])

    # recebe o mapa de ocupacao, 0 <= p_occ <= 100
    # retorna uma matriz booleana com 1 nos pixels de fronteiras
    def get_frontier_pixels(self, m, unknown_flag, threshold = 49):
        frontier_map = np.zeros(m.shape)
        it = np.nditer(m, flags=['multi_index'])
        for p in it:
            neighbours = self.get_neighbours(m.shape, it.multi_index)
            if 0 <= p < threshold and np.any(m[neighbours] == unknown_flag):
                    # e considerado fronteira todo ponto que esta provavelmente livre e 
                    # e vizinho de um ponto desconhecido
                    frontier_map[it.multi_index] = 1
                
        return frontier_map
        