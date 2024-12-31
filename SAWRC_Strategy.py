# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 16:04:26 2024

@author: ericl
"""

import numpy as np
# import matplotlib.pyplot as plt 
# import copy 
# import pandas as pd 
# from time import perf_counter
# import random 
# from tqdm import tqdm 
# import multiprocessing 
# import time 
# from collections import deque 

import Utility 
from SAWRC_World import World, Robot 


# In[Robot deployment]
def get_deployment_positions(xc, h0, num_robot, method='uniform'): 
    """
    Deployment methods: 
        'uniform': uniformly distributed over xc
        'over_untraversable_area': uniformly distributed over untraversable areas with a focus on wider area 
    """
    # get some basic parameters 
    world_test = World(width=np.amax(xc)*2) 
    robot_test = Robot(world_test) 
    dx = world_test.dx 
    travelSteepnessMax = robot_test.travelSteepnessMax 
    xl = world_test.x_limited 
    limit = world_test.limit 
    
    # get deployment location array
    dep_x_arr = None
    if method == 'uniform': 
        dep_x_arr = (np.round(np.linspace(np.amin(xl), np.amax(xl), num_robot+2)/dx)*dx).astype(int)[1:num_robot+1] 
    elif method == 'over_untraversable_area': 
        # extract list of connected untraversable areas 
        dh0 = np.gradient(h0, xc) 
        x_untraversable=xc[np.where(((dh0>travelSteepnessMax)|(dh0<-travelSteepnessMax))&(xc>-limit)&(xc<limit))] 
        if x_untraversable.size > 0: 
            xug_list = Utility.extract_connected_regions(x_untraversable, dx) 
            # rank list based on width of each area (high to low)
            sortfunc = lambda ee: np.amax(ee)-np.amin(ee) 
            xug_list.sort(key=sortfunc, reverse=True) 
            # deploy robot, place more robots on the wider area 
            num_xug = len(xug_list)
            residual_arr = np.zeros(num_xug)
            residual_arr[:num_robot%num_xug] = 1 
            nr_arr = (np.ones(num_xug) * (num_robot//num_xug) + residual_arr).astype(int) 
            pos_x_list = [] 
            for kk, xug in enumerate(xug_list): 
                nr = nr_arr[kk] 
                pos_x_list_sub = list((np.round(np.linspace(np.amin(xug), np.amax(xug), nr+2)/dx)*dx).astype(int)[1:nr+1])
                pos_x_list += pos_x_list_sub 
            dep_x_arr = np.array(pos_x_list) 
        else: 
            dep_x_arr = (np.round(np.linspace(np.amin(xl), np.amax(xl), num_robot+2)/dx)*dx).astype(int)[1:num_robot+1] 
        
    return dep_x_arr 









