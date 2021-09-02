#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

from skimage.draw import line


class OccupancyGrid():
    def __init__(self):
        # Dados do LIDAR
        self.sensor_range = 4 # Metros
        self.scan_range = 180*np.pi/180
        self.step_size = 2*np.pi/1024

        # Constantes relacionadas ao mapeamento
        self.MAP_SIZE = np.array([80, 80])
        self.CELL_SIZE = 0.5
        self.INITIAL_PROB_VALUE = 0.5

        rows, cols = (self.MAP_SIZE / self.CELL_SIZE).astype(int)
        self.m = np.full((rows, cols), None)


    ## Funções Auxiliares
    # Retorna uma matriz com duas colunas: [angle, dist]
    def format_laser_data(self, range_data, scan_range, step_size, max_sensor_range):

        laser_data = []

        range_data = np.asarray(range_data)
        pts = math.floor(scan_range/step_size)
        
        angle =- scan_range*0.5
        for i in range(pts):
            
            dist = range_data[i]        
            if dist <= 0:
                dist = max_sensor_range
            
            laser_data.append([angle, dist])
                
            angle=angle+step_size
            
        return np.array(laser_data)

    # Convertendo relacao indice do grid para Posição no mundo
    def coord2index(self, coords, cell_size):
        indexes = np.floor([coords[0] / cell_size, coords[1] / cell_size]).astype('int')
        return indexes

    def laser_noise(self, laser_data):
        noise = np.random.normal(loc=0.0, scale=0.1, size=(laser_data.shape[0],))
        laser_data[:, 1] += noise  # adiciona ruido apenas na distancia, nao no angulo de leitura

        return laser_data

    # Algoritmo de Mapeamento
    def logodds_to_prob(self, m):
        m = m.astype('float64')
        return 1 - (1 / (1 + np.exp(m)))
        
    def update_cells_logodds(self, current_value, prob_occupied, initial_value):
        return current_value + np.log(prob_occupied / (1 - prob_occupied)) - np.log(initial_value/(1 - initial_value))

    # TW_L e a matriz de transformacao homogenea do laser para o frame global
    # laser_position esta sempre no referencial do robo
    def update_map(self, laser_pos, TW_L, raw_range_data, max_sensor_range=5):
        laser_data = self.format_laser_data(raw_range_data, self.scan_range, self.step_size, max_sensor_range)
        laser_data = self.laser_noise(laser_data)

        for data in laser_data:
            ang, dist = data
            x, y = dist * np.cos(ang), dist * np.sin(ang)

            scan_global_pos = np.reshape(TW_L @ np.vstack((x, y, 0, 1)), (4,))
            scan_global_pos = self.coord2index(scan_global_pos[:2], self.CELL_SIZE)

            laser_global_pos = np.reshape(TW_L @ np.vstack((laser_pos[0], laser_pos[1], 0, 1)), (4,))
            laser_global_pos = self.coord2index(laser_global_pos[:2], self.CELL_SIZE)
            
            # Pegar as celulas para atualizar usando o algoritmo de bresenham
            rr, cc = line(laser_global_pos[1], laser_global_pos[0], scan_global_pos[1], scan_global_pos[0])

            m_aux = self.m[rr, cc]
            ind_first_scan = np.where(m_aux == None)
            m_aux[ind_first_scan] = self.INITIAL_PROB_VALUE
            if dist < max_sensor_range:
                m_aux[:-1] = self.update_cells_logodds(m_aux[:-1], 0.49, self.INITIAL_PROB_VALUE)
                m_aux[-1] = self.update_cells_logodds(m_aux[-1], 0.51, self.INITIAL_PROB_VALUE)
            else:
                m_aux = self.update_cells_logodds(m_aux, 0.49, self.INITIAL_PROB_VALUE)

            self.m[rr, cc] = m_aux
            
    def plot_map(self, name, scale=5):
        m = np.copy(self.m)
        ind_unknow = np.where(m == None)
        m[ind_unknow] = 0
        m = self.logodds_to_prob(m.astype('float64'))
        m_img = ((1 - m)*255).astype('uint8')
            
        ## Redimensionar a imagem para melhorar a visualizacao
        new_dim = (m_img.shape[1] * scale, m_img.shape[0] * scale)
        m_resized = cv2.resize(m_img, new_dim, interpolation=cv2.INTER_AREA)

        cv2.imshow(name, m_resized)
        cv2.waitKey(10) & 0xff
