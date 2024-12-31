# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:13:32 2024

@author: ericl
"""

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch
# import copy
# import pandas as pd
# from time import perf_counter
# import random
# from tqdm import tqdm
# import multiprocessing
# import time
# from collections import deque

import Utility
# from SAWRC_World import gauss, xc
# from SAWRC_Simulation import Simulator
# import SAWRC_Strategy
from SAWRC_Planning import Planner


# In[]
class Wasserstein_Multi_Robot_Distributed_Planner_Ver1(Planner):
    """
    Basic distributed multi-robot planning algorithm based on Wasserstein geodesics
    Robot only has access to local shape
    Robot steepness constraint is applied 
    Default sensing/action/avoidance range is approximately the same as robot length 
    """
    def __init__(self, xc, h0, h1, num_robot=1, pos_x_list=None,
                 obs_radius=200, action_radius=200, avoidance_distance=200, dW2=1/4, rng_seed=None,
                 apply_steepness_constraint=True, apply_modification_noise=True,
                 log_actions=False, log_progress=False, log_construction_state=False):
        super().__init__(xc, h0, h1, num_robot, pos_x_list,
                         apply_steepness_constraint, apply_modification_noise,
                         log_actions, log_progress, log_construction_state)
        self.obs_radius = obs_radius
        self.action_radius = action_radius
        self.avoidance_distance = avoidance_distance
        self.dW2 = dW2
        self.rng = np.random.default_rng(seed=rng_seed)
        # initialize the action plan
        for robot in self.sim.robots:
            robot.action_plan = []
        # debugging 
        global debugging_info
        debugging_info = {}


    def solve(self, debug=False, make_plot=False, timeout=10000):
        # debugging only 
        if debug:
            debugging_info['W2cur_obs'] = []
            debugging_info['W2t'] = []
            debugging_info['action_plan_size'] = []

        # save the 1st frame 
        if self.make_video:
            self.save_frame()

        # save initial construction state
        if self.log_construction_state:
            self.save_construction_state()

        # solve the construction problem
        while True:
            ind_list = np.arange(self.sim.num_robot)
            self.rng.shuffle(ind_list)
            for indr in ind_list:
                # choose robot 
                robot = self.sim.robots[indr]

                # check if there is an existing action plan
                if len(robot.action_plan) == 0:
                    xr_list = self.sim.get_robot_xs()
                    xr_list.pop(indr)

                    # check if there is any robot to avoid
                    if self.sim.num_robot == 1 or np.amin(np.abs(np.array(xr_list)-robot.pos_x)) > self.avoidance_distance:
                        # get observation 
                        indx = np.argwhere(self.sim.world.x==robot.pos_x)[0][0]
                        ind_obs_lb = max(0, indx - int(self.obs_radius/10))
                        ind_obs_ub = min(self.sim.world.x.size, indx + int(self.obs_radius/10))
                        x_obs = self.sim.world.x[ind_obs_lb:ind_obs_ub]
                        y_obs_true = np.copy(self.sim.world.y[ind_obs_lb:ind_obs_ub])
                        dy_obs = np.gradient(y_obs_true, x_obs)
                        yg_obs = np.copy(self.sim.world.yg[ind_obs_lb:ind_obs_ub])

                        # reconstruct structure from slope/derivative information 
                        y_obs = Utility.compute_1d_integral_function(x_obs, dy_obs)
                        y_obs -= np.trapz(y_obs, x_obs)/(x_obs[-1]-x_obs[0])
                        y_obs += np.trapz(yg_obs, x_obs)/(x_obs[-1]-x_obs[0])

                        # set intermediate goal 
                        W2cur_obs = Utility.compute_Wp_distance_of_structures(2, yg_obs, y_obs, x_obs)
                        t = min(1, self.dW2 / W2cur_obs)
                        ht = Utility.get_W2_geodesics(x_obs, y_obs, yg_obs, [t])[0]
                        if debug:
                            debugging_info['W2cur_obs'].append(W2cur_obs)
                            W2t = Utility.compute_Wp_distance_of_structures(2, y_obs, ht, x_obs)
                            debugging_info['W2t'].append(W2t)

                        # check if the intermediate goal satisfies a certain condition
                        condition_ig = True
                        if condition_ig:
                            """
                            Make an action plan 
                            """
                            # get area of actions
                            ind_action_lb = max(0, indx - int(self.action_radius/10))
                            ind_action_ub = min(self.sim.world.x.size, indx + int(self.action_radius/10))
                            x_action = self.sim.world.x[ind_action_lb:ind_action_ub]
                            x_action = x_action[np.where((x_action<self.sim.world.limit)&(x_action>-self.sim.world.limit))]

                            # create the matrix of all permitted construction functions
                            sn_arr = x_action / self.sim.world.dx
                            sn_arr = sn_arr.astype(int)
                            ypr0 = self.sim.world.cfc.consFunc('pile_right', 0, self.sim.world.x)
                            ypl0 = self.sim.world.cfc.consFunc('pile_left', 0, self.sim.world.x)
                            ypr_arr = np.zeros((sn_arr.size, x_obs.size))
                            ypl_arr = np.zeros((sn_arr.size, x_obs.size))
                            for ii, sn in enumerate(sn_arr):
                                ypr_arr[ii] = Utility.shift_elements(ypr0, sn)[ind_obs_lb:ind_obs_ub]
                                ypl_arr[ii] = Utility.shift_elements(ypl0, sn)[ind_obs_lb:ind_obs_ub]

                            # debugging only
                            if debug and make_plot:
                                print('index of observation:', ind_obs_lb, ind_obs_ub)
                                print('index of action:', ind_action_lb, ind_action_ub)
                                print('y_obs vol:', np.trapz(y_obs-self.sim.world.height_base, x_obs))
                                print('yg_obs vol:', np.trapz(yg_obs-self.sim.world.height_base, x_obs))
                                print('x_obs:', x_obs[0], '~', x_obs[-1])
                                print('x_action:', x_action[0], '~', x_action[-1])
                                fig, ax = plt.subplots()
                                ax.plot(x_obs, y_obs, label='y_obs')
                                ax.plot(x_obs, yg_obs, label='yg_obs')
                                ax.plot(x_obs, ht, label='ht')
                                ax.plot(x_obs, y_obs_true, label='y_obs_true')
                                ax.legend()

                            # Use greedy approach to find the action sequence that leads to ht
                            fnc = ht - y_obs
                            yc_sum = 0
                            yerror = np.inf
                            # find the optimal construction function
                            while True:
                                dpr = np.linalg.norm(ypr_arr + yc_sum - fnc, axis=1)
                                dpl = np.linalg.norm(ypl_arr + yc_sum - fnc, axis=1)
                                if np.amin(dpr) <= np.amin(dpl):
                                    action_name = 'pile_right'
                                    ind = np.argmin(dpr)
                                    position = x_action[ind]
                                    yc_opt = ypr_arr[ind]
                                else:
                                    action_name = 'pile_left'
                                    ind = np.argmin(dpl)
                                    position = x_action[ind]
                                    yc_opt = ypl_arr[ind]
                                # check if L2 distance is reduced
                                yerror_new = np.linalg.norm(yc_opt + yc_sum - fnc)
                                if yerror_new < yerror:
                                    yerror = yerror_new
                                    robot.action_plan.append((action_name, position))
                                    yc_sum += yc_opt
                                else:
                                    if debug:
                                        debugging_info['action_plan_size'].append(len(robot.action_plan))
                                    break
                        else:
                            """
                            Intermediate goal condition is not satisfied. Move elsewhere
                            """
                            self.sim.take_action(('move', self.rng.choice(self.sim.world.x_limited), indr)) 
                            if self.make_video:
                                self.save_frame()
                            if self.log_construction_state:
                                self.save_construction_state()
                    else:
                        """
                        Too close to another robot. Move elsewhere
                        """
                        self.sim.take_action(('move', self.rng.choice(self.sim.world.x_limited), indr))
                        if self.make_video:
                            self.save_frame()
                        if self.log_construction_state:
                            self.save_construction_state()
                else:
                    """
                    Execute one action from the action plan
                    """
                    app = robot.action_plan.pop(0)
                    action = (*app, indr)
                    self.sim.take_action(action)
                    if self.make_video:
                        self.save_frame()
                    if self.log_construction_state:
                        self.save_construction_state()

            # time out 
            if self.sim.world.stepCounter >= timeout:
                # print('Time out!')
                break

        # save the last frame and make video
        if self.make_video:
            self.save_frame(end=True)

        # merge saved construction state data
        if self.log_construction_state:
            self.save_construction_state(merge_data=True)
        
        


















