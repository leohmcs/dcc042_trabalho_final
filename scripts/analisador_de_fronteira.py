import numpy as np
import cv2

import math
import time

class AnalisadorFronteira:
    def __init__(self):
        # Map resolution
        self.height = 0
        self.width = 0

    # returns indexes of neighbours of pixel.
    def get_neighbours(self, map_shape, element_ind):
        rr, cc = element_ind
        row_ind = np.array([[rr - 1]*3, [rr]*3, [rr + 1]*3]).reshape(9,)
        col_ind = np.array([cc - 1, cc, cc + 1]*3)

        ind = np.vstack((row_ind, col_ind))

        # remove the element itself
        ind = np.delete(ind, 4, axis=1)

        # remove indexes out of bound
        rows2delete = np.where((ind[0] < 0) | (ind[0] >= map_shape[0]))
        ind = np.delete(ind, rows2delete, axis=1)

        cols2delete = np.where((ind[1] < 0) | (ind[1] >= map_shape[1]))
        ind = np.delete(ind, cols2delete, axis=1)

        return (ind[0], ind[1])

    # radius is the distance (in pixels) from pixel_coord to the neighbour pixels which 
    # will be assigned the value color, i.e. the result is a square with length (2*radius + 1) pixels
    def neighbour_pixels(self, pixel_coord, radius, m, color):
        # error check
        if color < 0 or color > 255:
            raise ValueError("Color value must be between 0 and 255.")
        
        if pixel_coord[0] > m.shape[0] or pixel_coord[1] > m.shape[1]:
            raise ValueError("Pixel doesn't exist.")
        elif pixel_coord[0] < 0 or pixel_coord[1] < 1:
            raise ValueError("Pixel coordinate must be non-negative.")

        if pixel_coord[0] - radius < 0 or pixel_coord[0] + radius > m.shape[0]:
            raise ValueError("Invalid radius for x coordinate.")
        elif pixel_coord[1] - radius < 0 or pixel_coord[1] + radius > m.shape[1]:
            raise ValueError("Invalid radius for y coordinate.")

        x_indexes = range(pixel_coord[0] - radius, pixel_coord[0] + radius + 1, 1)
        y_indexes = range(pixel_coord[1] - radius, pixel_coord[1] + radius + 1, 1)

        m[x_indexes, y_indexes] = color

        return m

    # Define new point to wall expansion: p1 = [x1, y1], p2 = [x2, y2], return p = [x1, x2]
    def define_new_point(self, p1, p2):
        p = np.array([0, 0])

        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        theta = math.atan2(dy, dx)

        theta_deg = theta * 180 / math.pi
        if -40 < theta_deg < 40:
            theta = 0
        elif 50 < theta_deg < 130:
            theta = math.pi / 2
        elif -130 < theta_deg < -50:
            theta = math.pi / 2
        
        distance = 100
        p[0] = distance + np.linalg.norm(p1 - p2) * math.cos(theta) + p1[0]
        p[1] = distance + np.linalg.norm(p1 - p2) * math.sin(theta) + p1[1]

        p[p < 0] = 0
        
        if p[0] > self.height:
            p[0] = self.height - 1

        if p[1] > self.width:
            p[1] = self.width - 1

        return p

    # p1 = [x1, y1], p2 = [x2, y2], sign = increment (default True) or decrement (False) theta
    def define_point_rect(self, p1, p2, sign_pos = True):
        p = np.array([0, 0])

        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        theta = math.atan2(dy, dx)

        theta_deg = theta * 180 / math.pi
        if -40 < theta_deg < 40:
            theta = 0
        elif 50 < theta_deg < 130:
            theta = math.pi / 2
        elif -130 < theta_deg < -50:
            theta = math.pi / 2

        if sign_pos:
            theta += math.pi / 2
        else:
            theta -= math.pi / 2

        distance = 100
        p[0] = distance + np.linalg.norm(p1 - p2) * math.cos(theta) + p1[0]
        p[1] = distance + np.linalg.norm(p1 - p2) * math.sin(theta) + p1[1]

        p[p < 0] = 0
        
        if p[0] > self.height:
            p[0] = self.height - 1

        if p[1] > self.width:
            p[1] = self.width - 1

        return p
    
    def verify_expansion(self):
        pass
    
    # returns the number of free, not free and total pixels
    def count_free_pixels(self, window):
        free = np.count_nonzero(window == 255)
        non_free = window.size
        total = free + non_free
        return free, non_free, total

    # win_start is the coordinate of the top-left pixel of the window to apply the algorithm. win_len and win_height are the 
    # length and height of window, respectively
    # limit = np.array([x, y, length, height])
    # frontier = np.array([x1, y1]. ..., [xn, yn])
    def expand_white_pixels(self, m, frontier, win_start, win_len, win_height):
        color = 255
        color_step = 0.08  # as we move beyond the frontier, uncertainty increases, thus the value of the pixel will decrease

        # turn into lists of columns and rows
        frontier = np.transpose(frontier)
        
        # remove pixels beyond limit
        ## remove pixels which column is out of the window
        ind_2_remove = np.where(frontier[0] < win_start[0] or frontier[0] > win_start[0] + win_len)  
        frontier = np.delete(frontier, ind_2_remove, axis=1)

        ## remove pixels which row is out of the window
        ind_2_remove = np.where(frontier[1] < win_start[1] or frontier[1] > win_start[1] + win_height)  
        frontier = np.delete(frontier, ind_2_remove, axis=1)

        # TODO: escrever isso em codigo
        # x = 255 - 0.08*n, n = a menor distancia, em pixels, entre o pixel x da janela e a fronteira
        
        window_ind_cols = np.array(range(win_start[0], win_start[0] + win_len, 1))
        window_ind_rows = np.array(range(win_start[1], win_start[1] + win_height, 1))

        # indexes of the whole window is just the Cartesian Product of both arrays
        window_ind = np.transpose([np.tile(window_ind_rows, len(window_ind_cols)), np.repeat(window_ind_cols, len(window_ind_rows))])
        
        dists = np.min(np.linalg.norm(abs(window_ind[:, None] - frontier), axis=2), axis=1)
        dists = np.reshape(dists, (win_height, win_len))

        m[win_start[0]:win_start[0] + win_height, win_start[1]:win_start[1] + win_len] = color - dists*color_step

        return m

    # Calculate window entropy; window is a grasycale image
    def entropy(self, window):
        # probability pixel is occupied (prob free = 1 - prob occupied)
        p_occ = window.copy().astype('float64')
        p_occ[p_occ == 0] = 1       # black pixel means its occupied
        p_occ[p_occ == 255] = 0     # white pixel means its free
        p_occ[p_occ == 155] = 0.5   # unknown (-1 in occupancy grid)
        
        indexes = np.array(np.where(p_occ > 155))
        if indexes.size > 0:
            p_occ[indexes[0], indexes[1]] = ((255 - p_occ[indexes[0], indexes[1]]) / 200).astype('float64')

        # remaining pixels
        indexes = np.array(np.where(p_occ > 1))
        if indexes.size > 0:
            p_occ[indexes[0], indexes[1]] =((200 - p_occ[indexes[0], indexes[1]]) / 200).astype('float64')  # remaining pixel
        
        # now calculate total entropy
        entropy = np.ones(p_occ.shape)
        entropy[p_occ == 0] = 0     # log2(1)
        entropy[p_occ == 1] = 0     # log2(1)
        aux = p_occ[entropy != 0]   # avoid log2(0), which happens when p_occ == 0 or p_occ == 1
        entropy[entropy != 0] = (aux * np.log2(aux)) + (1 - aux) * np.log2(1 - aux)

        return -np.sum(np.sum(entropy))

    # receives the occupancy map, not the map image
    # treshold is the minimum probability to consider a pixel free. left it as a parameter because it has a 
    # great impact on frontier detection
    def get_frontier_pixels(self, m, threshold = 49):
        # frontier_map is a boolean array with 1's in frontier positions
        frontier_map = np.zeros(m.shape)
        it = np.nditer(m, flags=['multi_index'])
        for p in it:
            neighbours = self.get_neighbours(m.shape, it.multi_index)
            if 0 <= p < threshold and np.any(m[neighbours] == -1):
                    # in other words, if the cell is probably free and any of its neighbours is unknowm, it is a frontier
                    frontier_map[it.multi_index] = 1

        return frontier_map

    def get_frontiers_goal(self, frontiers, robot_pos):
        pass
        
        