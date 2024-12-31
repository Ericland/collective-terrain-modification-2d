# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:13:32 2024

@author: ericl
"""

import numpy as np
import matplotlib.pyplot as plt 
# import copy 
# import pandas as pd 
# from time import perf_counter
# import random
# from tqdm import tqdm 
# import multiprocessing 
# import time 
from collections import deque 

# import Utility 
from SAWRC_World import World, Robot
import SAWRC_Strategy 


# In[]
class Simulator: 
    """
    Basic multi-robot simulator that coordinates interactions of World and Robot
    
    action: (action_name, position, indr)
        action_name: name of the action including: 'move', 'pile_right', 'pile_left', 'dig'
        position: travel destination or the position where the action will be taken 
        indr: index of the execution robot 
    
    The robot can: 
        - Move to a given location 
        - Modify the structure at a given location. 
            If the given location is not the current location, the robot needs to move to the given location and modify. 
            If the robot cannot move to the given location, the robot does nothing. 
    """
    def __init__(self, 
                 apply_steepness_constraint = True, 
                 apply_modification_noise = True, 
                 log_actions = False, 
                 log_progress = False, 
                 ): 
        self.apply_steepness_constraint = apply_steepness_constraint 
        self.apply_modification_noise = apply_modification_noise 
        
        # logging 
        self.log_actions = log_actions 
        self.action_history = deque() 
        self.log_progress = log_progress 
        self.progress_history = deque() 
        
        
    # ==============================
    # Core functions  
    # ==============================
    def set_world(self): 
        self.world = World(width=np.amax(self.xc)*2, 
                           structure_initial=(self.xc, self.h0), 
                           structure_goal=(self.xc, self.h1)) 
        self.world.apply_modification_noise = self.apply_modification_noise 
        
        
    def set_robots(self): 
        if np.all(self.pos_x_list) == None: 
            self.pos_x_list = SAWRC_Strategy.get_deployment_positions(self.xc, self.h0, self.num_robot, 'uniform') 
        self.robots = []
        for ii in range(self.num_robot): 
            pos_x = self.pos_x_list[ii]
            robot = Robot(self.world, pos_x=pos_x, name=ii)
            robot.apply_steepness_constraint = self.apply_steepness_constraint
            self.robots.append(robot)
            
            
    def set_task(self, xc, h0, h1, num_robot=1, pos_x_list=None): 
        self.xc = np.copy(xc) 
        self.h0 = np.copy(h0) 
        self.h1 = np.copy(h1) 
        self.num_robot = num_robot # number of robots
        self.pos_x_list = pos_x_list # initial x-coordinate of robots 
        self.set_world()
        self.set_robots()
        
        
    def reset(self): 
        self.set_world()
        self.set_robots()
            
            
    def take_action(self, action): 
        action_name, position, indr = action 
        agent = self.robots[indr]
        # move to a given location 
        if action_name == 'move': 
            # we assume that moving will not affect the structure
            agent.move(position) 
        # modify the structure at a given location 
        # if the given location is not the current location, the robot needs to move to the given location and modify 
        # If the robot cannot move to the given location, the robot does nothing 
        elif action_name in self.world.cfc.action_list: 
            res = agent.move(position) 
            if res == 'success': 
                agent.modify(action_name) 
                # we assume that the robot action will only change y-coordinate of robots 
                for robot in self.robots: 
                    robot.update_pos_y()
                    
        # logging 
        if self.log_actions: 
            self.action_history.append(action)
        if self.log_progress: 
            self.progress_history.append(self.get_progress())
                    
        # update timer 
        self.world.update_timer()
            
            
    # ==============================
    # Utility functions  
    # ==============================
    def plot_state(self): 
        fig, ax = self.world.plot_initial_goal_current_structures()
        for robot in self.robots: 
            ax.scatter(robot.pos_x, robot.pos_y, label='robot #'+str(robot.name), marker='s', zorder=10)
            # ax.scatter(robot.pos_x, robot.pos_y, label='robot #'+str(robot.name), marker="$"+str(robot.name)+"$", zorder=10, color='black', s=50)
        ax.legend()
        
        return fig, ax 
    
    
    def plot_progress_history(self): 
        if len(self.progress_history) > 0: 
            fig, ax = plt.subplots(dpi=150) 
            ax.plot(self.progress_history) 
            ax.set_title('progress history')
        
            return fig, ax 
        
        
    def get_progress(self): 
        progress = self.world.get_progress() 
        
        return progress


    def get_robot_xs(self):
        xrs = [robot.pos_x for robot in self.robots]

        return xrs


    def get_robot_positions(self): 
        prs = [(robot.pos_x, robot.pos_y) for robot in self.robots]
        
        return prs
    
    
    def compute_travel_distance(self): 
        if self.log_actions: 
            xts_list = [[pos_x] for pos_x in self.pos_x_list] 
            for action in self.action_history: 
                xts_list[action[2]].append(action[1]) 
            td_list = [] 
            for xts in xts_list: 
                td_list.append(np.sum(np.abs(np.diff(xts)))) 
            travel_distance = np.sum(td_list) 
            
            return travel_distance
    
    

    
    

    
    
        
            









