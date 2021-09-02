#!/usr/bin/env python
# coding: utf-8


import cv2
import numpy as np

import robot

from threading import Thread


TIME_LIMIT_MINUTES = 20  # minutes


def save_map_image(robot, scale=5):
        m = np.copy(robot.occ_grid.m)
        ind_unknow = np.where(m == None)
        m[ind_unknow] = 0
        m = robot.logodds_to_prob(m.astype('float64'))
        m_img = ((1 - m)*255).astype('uint8')
            
        ## Redimensionar a imagem para melhorar a visualizacao
        new_dim = (m_img.shape[1] * scale, m_img.shape[0] * scale)
        m_resized = cv2.resize(m_img, new_dim, interpolation=cv2.INTER_AREA)

        title = robot.name
        cv2.imwrite(title + " map", m_resized)


if __name__ == "__main__":
    # iniciar exploracao de cada robo em paralelo
    r0 = robot.Robot('Pioneer_p3dx#0', '127.0.0.1', 19999)
    ## exemplo de como adicionar novos robôs ##
    # r1 = robot.Robot('Pioneer_p3dx#1', '127.0.0.1', 20000)
    # r2 = robot.Robot('Pioneer_p3dx#2', '127.0.0.1', 20001)
    
    # para um único robô
    r0.begin_exploration(None, TIME_LIMIT_MINUTES, None)

    ## exemplo de como usar para vários robôs ##
    # t0 = Thread(target=r0.begin_exploration, args=(None, 5, (r1,)))
    # t0.start()

    # t1 = Thread(target=r1.begin_exploration, args=(None, 5, (r0,)))
    # t1.start()

    # t2 = Thread(target=r2.begin_exploration, args=(None, 5, (r1, r3)))
    # t2.start()

    # time.sleep(1)

    # t0.join()
    # t1.join()

    # start_time = time.time()
    # while time.time() - start_time <= 60 * TIME_LIMIT_MINUTES:
    #     r0.mapping()
    #     r1.mapping()

    #     r0.occ_grid.plot_map(r0.name)
    #     r1.occ_grid.plot_map(r1.name)

    # save_map_image(r0)
    # save_map_image(r1)