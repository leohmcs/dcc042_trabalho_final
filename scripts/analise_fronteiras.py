#!/usr/bin/env python

import rospy

from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import numpy as np
import cv2

import time
import random

import analisador_de_fronteira
# from near_frontier.src import rrt


# Constants
IMAGES_DIR_PATH = "/home/leozin/verlab_ws/src/ros_mestrado_jhielson/"

class AnalisadorFronteirasNode:
    def __init__(self):
        self.analisador = analisador_de_fronteira.AnalisadorFronteira()

        self.map = None
        self.bin_map = None
        self.map_image = None
        self.map_resolution = 0
        self.map_size = np.array([0, 0], dtype=np.int64)  # height/width form (row/column)
        self.map_entropy = -1
        
        self.map_sub = rospy.Subscriber("map", OccupancyGrid, self.map_callback)
        self.pose_sub = rospy.Subscriber("odom", Odometry, self.pose_callback)
        self.finish_sub = rospy.Subscriber("fim", Twist, self.finish_callback)

        self.biggest_frontier_pub = rospy.Publisher("biggest_frontier", Twist, queue_size=100)
        self.high_gain_frontier_pub = rospy.Publisher("highest_gain_frontier", Twist, queue_size=100)
        self.near_frontier_pub = rospy.Publisher("near_frontier", Twist, queue_size=100)
        self.no_frontier_pub = rospy.Publisher("no_frontier", Twist, queue_size=100)

    # ROS callback functions
    def map_callback(self, msg):
        self.map_resolution = msg.info.resolution
        self.map_size[0] = msg.info.height
        self.map_size[1] = msg.info.width

        self.map = np.reshape(msg.data, self.map_size)
        self.bin_map = self.binarize_map(self.map, 51)
        self.map_image = self.occupancygrid_to_image(self.bin_map)

    def pose_callback(self, msg):
        pass

    def finish_callback(self, msg):
        pass

    # Auxiliar functions
    ## Turn occupancy map into an image
    def occupancygrid_to_image(self, occ_map):
        m = np.copy(occ_map)
        image = np.reshape(m, self.map_size)

        image[image >= 0] = (100 - (image[image >= 0])) * 255/100
        image[image == -1] = 155
        
        return image.astype('uint8')

    ## binarize occupancy map
    def binarize_map(self, m, thresh):
        m = np.array(m)
        m[m > thresh] = 100
        m[(0 < m) & (m < thresh)] = 0
        # m[(free_thresh <= m) & (m <= occ_thresh)] = -1

        return m

rospy.init_node('analiseFronteiras')

node = AnalisadorFronteirasNode()

rate = rospy.Rate(0.2)
while not rospy.is_shutdown():
    while node.map_size[0] == 0 or node.map_size[1] == 0:
        rospy.loginfo("Waiting for map.")
        time.sleep(1)

    # print('Map size (width, height): {}'.format(node.map_size))
    cv2.imshow("Binary Map", node.map_image)
    cv2.waitKey(10)

    # entropy of the whole map
    map_entropy = node.analisador.entropy(node.map_image)
    print("Map entropy: {}".format(map_entropy))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), anchor=(1, 1))
    map_eroded = cv2.erode(node.map_image, kernel)
    cv2.imshow("Eroded", map_eroded)

    frontiers_map = node.analisador.get_frontier_pixels(node.bin_map).astype('uint8')
    cv2.imshow("Antes", frontiers_map * 255)
    frontiers_map = cv2.morphologyEx(frontiers_map, cv2.MORPH_DILATE, kernel, iterations=3)
    cv2.imshow("Frontiers", frontiers_map * 255)

    frontiers = cv2.Canny(frontiers_map, 0, 2)
    cv2.imshow("Edges", frontiers)

    #############################
    # get random pixel of frontier
    # rng = np.random.default_rng() 
    # p = rng.choice(np.transpose(np.where(frontiers > 0)))

    # f = np.where(frontiers > 0)
    # i = int(np.floor(random.random() * len(f[0])))
    # goal = [f[0, i], f[1, i]]
    
    # teste = cv2.cvtColor(frontiers, cv2.COLOR_GRAY2RGB)
    # teste[f[0][i], f[1][i]] = (0, 255, 0)

    # n_size = (8 * teste.shape[0], 8 * teste.shape[1])
    # teste = cv2.resize(teste, n_size, interpolation=cv2.INTER_AREA)
    # cv2.imshow("Point", teste)
    
    #############################

    rate.sleep()