# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:13:32 2024

@author: ericl
"""
import os
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
import time 
from collections import deque 

import Utility 
# from SAWRC_World import Random_Structure_Generator
from SAWRC_Simulation import Simulator
import SAWRC_Strategy


# In[]
class Planner: 
    """
    Basic multi-robot planner 
    """
    def __init__(self, xc, h0, h1, num_robot=1, pos_x_list=None, 
                 apply_steepness_constraint=True, apply_modification_noise=True, 
                 log_actions=False, log_progress=False, log_construction_state=False):
        self.xc = np.copy(xc) 
        self.h0 = np.copy(h0) 
        self.h1 = np.copy(h1) 
        self.num_robot = num_robot
        self.pos_x_list = pos_x_list 
        self.apply_steepness_constraint = apply_steepness_constraint 
        self.apply_modification_noise = apply_modification_noise 
        self.log_actions = log_actions 
        self.log_progress = log_progress
        self.log_construction_state = log_construction_state

        # logging construction state
        self.data_autosave_size = 100 # batch size of autosaved data
        self.construction_state_list = [] # list of construction stats
        self.autosaved_data_dir_list = [] # list of directories of autosaved data
        self.merged_data_dir = None # directory of merged data

        # make video
        self.make_video = False 
        self.frame_index = 0 # index of frame for making videos 
        
        self.reset()
        
        
    def reset(self): 
        self.sim = Simulator(self.apply_steepness_constraint, self.apply_modification_noise, 
                             self.log_actions, self.log_progress) 
        self.sim.set_task(self.xc, self.h0, self.h1, self.num_robot, self.pos_x_list)
        self.solution = deque()
        self.progress = 0


    def solve(self):
        pass
        
        
    def load(self, planner): 
        self.xc = np.copy(planner.xc) 
        self.h0 = np.copy(planner.h0) 
        self.h1 = np.copy(planner.h1) 
        self.num_robot = planner.num_robot 
        self.sim = planner.sim 
        # redeploy robots
        self.pos_x_list = (np.round(np.linspace(-self.sim.world.limit, self.sim.world.limit, self.num_robot+2)/10)*10).astype(int)[1:self.num_robot+1]
        xrs = np.array([robot.pos_x for robot in self.sim.robots])
        ind_list = np.argsort(xrs) 
        for ii, indr in enumerate(ind_list): 
            action = ('move', self.pos_x_list[ii], indr) 
            self.sim.take_action(action) 
            if self.make_video: 
                self.save_frame()
            if self.log_construction_state:
                self.save_construction_state()
        xrs = np.array([robot.pos_x for robot in self.sim.robots])
        if not np.all(np.sort(xrs) == self.pos_x_list): 
            print('redeployment fails')
            print('desired:', self.pos_x_list)
            print('result:', xrs)
        # video setup 
        self.make_video = planner.make_video 
        if planner.make_video: 
            self.frame_sampling_period = planner.frame_sampling_period 
            self.image_loc = planner.image_loc 
            self.delete_images_afterwards = planner.delete_images_afterwards 
            self.frame_index = planner.frame_index
    
    
    def measure_computation_time(self): 
        tic = time.perf_counter() 
        self.solve()
        toc = time.perf_counter()
        duration = toc - tic 
        efficiency = duration / self.sim.world.stepCounter
        print(f"Duration: {duration:0.4f} seconds")
        print(f"Efficiency: {efficiency:0.4f} seconds/action")
        
        
    def set_video(self, frame_sampling_period=10, image_loc='video/video_image/', delete_images_afterwards=True): 
        self.make_video = True 
        self.frame_sampling_period = frame_sampling_period 
        self.image_loc = image_loc 
        self.delete_images_afterwards = delete_images_afterwards 
    
    
    def save_frame(self, end=False): 
        if (self.sim.world.stepCounter%self.frame_sampling_period == 0 or 
            self.sim.world.stepCounter == 0 or end): 
            fig, ax = self.plot_state() 
            ax.legend(loc='upper right')
            file_name = self.image_loc + 'img_' + str(self.frame_index) + '.png'
            plt.savefig(file_name, dpi=200)
            plt.close()
            self.frame_index += 1 
            print('.', end='')
            
        if end: 
            Utility.make_video(delete_images_afterwards=self.delete_images_afterwards)


    def save_construction_state(self, merge_data=False):
        """
        Construction state = (h, [xr0, xr1, ...])
        where h is the structure and xri is the x-coordinate of robot i 
        This is the minimum needed to recover the construction processs 
        """
        # if data size reaches data_autosave_size or merging data, save the data
        if (len(self.construction_state_list) == self.data_autosave_size
                or (merge_data and len(self.construction_state_list) > 0)):
            autosaved_data_dir = Utility.save_data(
                self.construction_state_list,
                'data/autosave/',
                'cs_' + str(len(self.autosaved_data_dir_list)) + '.pkl',
                False)
            self.autosaved_data_dir_list.append(autosaved_data_dir)
            # clear list after autosaving
            self.construction_state_list = []

        if not merge_data:
            # log construction state 
            # save h as float32 to save storage space
            construction_state = (self.sim.world.y.astype(np.float32), self.sim.get_robot_xs())
            self.construction_state_list.append(construction_state)
        else:
            # merge all data
            all_construction_states = []
            for dir in self.autosaved_data_dir_list:
                construction_state_list = Utility.load_data(*dir)
                all_construction_states += construction_state_list
            construction_data = (
                np.copy(self.sim.world.x),
                np.copy(self.sim.world.yi),
                np.copy(self.sim.world.yg),
                all_construction_states,
                )
            self.merged_data_dir = Utility.save_data(
                construction_data,
                'data/autosave/',
                'default',
                False)
            # delete autosaved data batches
            for dir in self.autosaved_data_dir_list:
                save_loc, file_name = dir
                os.remove(save_loc + file_name)
            self.autosaved_data_dir_list = []
            
            
    def plot_state(self): 
        fig, ax = self.sim.plot_state() 
        # traversable area 
        xc_traversable_list = [robot.get_traversable_area() for robot in self.sim.robots]
        xc_traversable_union = Utility.get_union(xc_traversable_list) 
        yc_traversable_union = np.interp(xc_traversable_union, self.sim.world.x, self.sim.world.y) 
        ax.scatter(xc_traversable_union, yc_traversable_union, color='yellow', label='reachable area', marker='.') 
        # movable area 
        xc_movable_list = [robot.get_traversable_area(climb_down_allowed=False) for robot in self.sim.robots]
        xc_movable_union = Utility.get_union(xc_movable_list) 
        yc_movable_union = np.interp(xc_movable_union, self.sim.world.x, self.sim.world.y) 
        ax.scatter(xc_movable_union, yc_movable_union, color='green', label='traversable area', marker='.') 
        # legend 
        ax.legend(loc='upper right') 
        
        return fig, ax 
    
    
    def plot_progress_history(self): 
        fig, ax = self.sim.plot_progress_history() 
        
        return fig, ax 
        
        
# In[]
class Wasserstein_Multi_Robot_Planner(Planner): 
    """
    Basic multi-robot planning algorithm based on Wasserstein geodesics 
    Robot has access to global knowledge 
    Robot steepness constraint can be applied 
    
    Each subtask is solved by a greedy approach. 
    Then the action is executed by the robot that is closest to the action position 
    and can move to that location. If such robot does not exist, skip this action. 
    """
    def __init__(self, xc, h0, h1, num_robot=1, pos_x_list=None, dW2=1, 
                 apply_steepness_constraint=True, apply_modification_noise=True, 
                 log_actions=False, log_progress=False, log_construction_state=False):
        super().__init__(xc, h0, h1, num_robot, pos_x_list, 
                         apply_steepness_constraint, apply_modification_noise, 
                         log_actions, log_progress, log_construction_state)
        self.dW2 = dW2 
        
        
    def update_solution(self): 
        pass 
        
        
    def update_progress(self): 
        pass 
            
            
    def get_construction_function_matrix(self, action_area):
        # create the matrix of all construction functions in the action area 
        sn_arr = action_area / self.sim.world.dx 
        sn_arr = sn_arr.astype(int)
        ypr0 = self.sim.world.cfc.consFunc('pile_right', 0, self.sim.world.x)
        ypl0 = self.sim.world.cfc.consFunc('pile_left', 0, self.sim.world.x)
        ypr_arr = np.zeros((sn_arr.size, ypr0.size)) 
        ypl_arr = np.zeros((sn_arr.size, ypl0.size)) 
        for ii, sn in enumerate(sn_arr): 
            ypr_arr[ii] = Utility.shift_elements(ypr0, sn)  
            ypl_arr[ii] = Utility.shift_elements(ypl0, sn) 
            
        return ypr_arr, ypl_arr 
    
    
    def rank_execution_robot(self, action_name, action_position, method): 
        # return the ranked robot index list based on the action to be executed 
        # ranking method: 
        #   min distance: distance to the action position (low to high)
        #   max std: standard deviation of robot positions (high to low)
        rpx_arr = np.array([robot.pos_x for robot in self.sim.robots])
        ind_list = None
        if method == 'min distance': 
            ind_list = np.argsort(np.abs(rpx_arr-action_position)) 
        elif method == 'max std': 
            rpx_matrix = np.zeros((self.sim.num_robot, self.sim.num_robot)) 
            rpx_matrix += rpx_arr 
            for rr in range(self.sim.num_robot): 
                rpx_matrix[rr, rr] = action_position 
            std_arr = np.std(rpx_matrix, axis=1) 
            ind_list = np.flip(np.argsort(std_arr)) 
            
        return ind_list 
        
        
    def solve(self, debug=False, timeout=10000, method='min distance'): 
        # save the 1st frame 
        if self.make_video: 
            self.save_frame()

        # save initial construction state
        if self.log_construction_state:
            self.save_construction_state()
        
        # create the matrix of all permitted construction functions 
        ypr_arr, ypl_arr = self.get_construction_function_matrix(self.sim.world.x_limited)
            
        # solve the construction problem
        W2init = self.sim.world.W2init  
        W2cur = W2init
        W2opt = W2cur 
        while True:             
            # set intermediate goal 
            t = min(1, self.dW2 / W2cur) 
            ht = Utility.get_W2_geodesics(self.sim.world.x, self.sim.world.y, self.sim.world.yg, [t])[0] 
            
            # Use greedy approach to find the action sequence that leads to ht 
            action_sequence_sub = []
            fnc = ht - self.sim.world.y 
            yc_sum = 0 
            yerror = np.inf 
            # find the optimal construction function 
            while True: 
                dpr = np.linalg.norm(ypr_arr + yc_sum - fnc, axis=1) 
                dpl = np.linalg.norm(ypl_arr + yc_sum - fnc, axis=1) 
                if np.amin(dpr) <= np.amin(dpl): 
                    action = 'pile_right' 
                    ind = np.argmin(dpr)
                    position = self.sim.world.x_limited[ind] 
                    yc_opt = ypr_arr[ind] 
                else: 
                    action = 'pile_left' 
                    ind = np.argmin(dpl)
                    position = self.sim.world.x_limited[ind]
                    yc_opt = ypl_arr[ind]
                # check if L2 distance is reduced 
                yerror_new = np.linalg.norm(yc_opt + yc_sum - fnc) 
                if yerror_new < yerror: 
                    yerror = yerror_new  
                    action_sequence_sub.append((action, position)) 
                    yc_sum += yc_opt 
                else: 
                    break 
                
            # for debugging 
            if debug: 
                fig, ax = plt.subplots()
                ax.plot(self.sim.world.x, fnc, label='fnc') 
                ax.plot(self.sim.world.x, yc_sum, label='yc_sum') 
                ax.legend()
                plt.show()
                
            # execute action sequence
            # The execution robot is the one that is closest to the action positon 
            # and can move to that location 
            # If such robot does not exist then skip this action 
            for app in action_sequence_sub: 
                action_name, position = app 
                ind_list = self.rank_execution_robot(action_name, position, method) 
                xc_traversable_list = [robot.get_traversable_area(climb_down_allowed=True) for robot in self.sim.robots]
                for ind in ind_list: 
                    if position in xc_traversable_list[ind]: 
                        action = (action_name, position, ind) 
                        self.sim.take_action(action) 
                        if self.make_video: 
                            self.save_frame()
                        if self.log_construction_state:
                            self.save_construction_state()
                        break 

            # check if W2 distance is reduced
            W2cur = Utility.compute_Wp_distance_of_structures(2, self.sim.world.yg, self.sim.world.y, self.sim.world.x)
            if W2cur < W2opt: 
                W2opt = W2cur 
                self.update_solution() 
                self.update_progress()
            else:
                break 
            
            # time out 
            if self.sim.world.stepCounter > timeout: 
                print('Time out!')
                break 
        
        # save the last frame and make video  
        if self.make_video: 
            self.save_frame(end=True)

        # merge saved construction state data
        if self.log_construction_state:
            self.save_construction_state(merge_data=True)
            
            
# In[]
class Flatten_Planner(Planner): 
    def __init__(self, xc, h0, h1, num_robot=1, pos_x_list=None, rng_seed=None, steepness_desired=np.array([-0.3, 0.3]), 
                 apply_steepness_constraint=True, apply_modification_noise=True, 
                 log_actions=False, log_progress=False, log_construction_state=False):
        if np.all(pos_x_list) == None: 
            pos_x_list = SAWRC_Strategy.get_deployment_positions(xc, h0, num_robot, 'over_untraversable_area') 
        super().__init__(xc, h0, h1, num_robot, pos_x_list, 
                         apply_steepness_constraint, apply_modification_noise, 
                         log_actions, log_progress, log_construction_state)
        self.rng = np.random.default_rng(seed=rng_seed) 
        self.steepness_desired = steepness_desired
        self.steepness_limit = np.copy(self.steepness_desired)
            
            
    def solve(self, timeout=10000):
        # save the 1st frame 
        if self.make_video: 
            self.save_frame()

        # save initial construction state
        if self.log_construction_state:
            self.save_construction_state()
            
        # debugging 
        global info 
        info = [[] for ii in range(self.sim.num_robot)] 
        
        tt = 0 
        ind_list = np.arange(self.sim.num_robot)
        while True: 
            # iterate through robots in random order 
            action_name = 'do_nothing' 
            self.rng.shuffle(ind_list) 
            for indr in ind_list: 
                robot = self.sim.robots[indr] 
                xc_movable = robot.get_traversable_area(climb_down_allowed=False) 
                
                # debugging 
                # info[indr].append((np.amin(xc_movable), np.amax(xc_movable), self.sim.world.stepCounter))
                
                dy = np.gradient(self.sim.world.y, self.sim.world.x) 
                position_candidates = self.sim.world.x[np.where((self.sim.world.x>=np.amin(xc_movable))
                                                                &(self.sim.world.x<=np.amax(xc_movable))
                                                                &((dy<self.steepness_limit[0])|(dy>self.steepness_limit[1])))]
                if position_candidates.size > 0: 
                    position = self.rng.choice(position_candidates) 
                    dy_position = dy[np.where(self.sim.world.x==position)][0] 
                    if dy_position < self.steepness_limit[0]: 
                        action_name = 'pile_right' 
                    elif dy_position > self.steepness_limit[1]: 
                        action_name = 'pile_left' 
                    action = (action_name, position, indr) 
                    self.sim.take_action(action) 
                    if self.make_video: 
                        self.save_frame()
                    if self.log_construction_state:
                        self.save_construction_state()
                tt += 1 
                
            # if robots did nothing, decrease the range of steepness limit 
            if action_name == 'do_nothing': 
                self.steepness_limit *= 0.9 
                            
            # check if the steepness of the structure falls into the desired limit 
            if self.check_completion(): 
                break 
                            
            # time out 
            if tt > timeout: 
                print('Time out!')
                break 
            
        # save the last frame and make video  
        if self.make_video: 
            self.save_frame(end=True)

        # merge saved construction state data
        if self.log_construction_state:
            self.save_construction_state(merge_data=True)
            
            
    def check_completion(self): 
        dh = np.gradient(self.sim.world.y, self.sim.world.x) 
        completion = (np.amin(dh)>=self.steepness_desired[0] and np.amax(dh)<=self.steepness_desired[1])
            
        return completion 
    
    
# In[] 
class Flatten_Wasserstein_Planner: 
    def __init__(self, xc, h0, h1, num_robot=1, steepness_desired=np.array([-0.3, 0.3]), dW2=1,
                 apply_steepness_constraint=True, apply_modification_noise=True, 
                 log_actions=False, log_progress=False, log_construction_state=False):
        self.planner1 = Flatten_Planner(xc, h0, h1, num_robot, None, None, steepness_desired, 
                                        apply_steepness_constraint, apply_modification_noise, 
                                        log_actions, log_progress, log_construction_state)
        self.planner2 = Wasserstein_Multi_Robot_Planner(xc, h0, h1, num_robot, None, dW2,
                                                        apply_steepness_constraint, apply_modification_noise, 
                                                        log_actions, log_progress, log_construction_state)
        
        
    def set_video(self, frame_sampling_period=10, image_loc='video/video_image/', delete_images_afterwards=False): 
        self.planner1.set_video(frame_sampling_period, image_loc, delete_images_afterwards) 
        
        
    def solve(self, debug=False, timeout=10000):
        if debug: 
            self.planner1.plot_state() 
            
        self.planner1.solve(timeout=timeout) 
        
        if debug: 
            fig, ax = self.planner1.plot_state() 
            ax.legend([]) 
            
        # continue construction with Wasserstein planner 
        # redeploy robots 
        self.planner2.load(self.planner1) 
        
        if self.planner2.make_video: 
            self.planner2.delete_images_afterwards = True 
        
        if debug: 
            self.planner2.plot_state() 
            
        timeout += self.planner2.sim.world.stepCounter
        self.planner2.solve(timeout=timeout) 
        
        if debug: 
            fig, ax = self.planner2.plot_state()
            ax.legend([])

        # combine saved construction data
        cd1 = Utility.load_data(*self.planner1.merged_data_dir)
        cd2 = Utility.load_data(*self.planner2.merged_data_dir)
        cd_all = (cd1[0], cd1[1], cd1[2], cd1[3] + cd2[3][1:])
        # remove old data
        os.remove(self.planner1.merged_data_dir[0] + self.planner1.merged_data_dir[1])
        os.remove(self.planner2.merged_data_dir[0] + self.planner2.merged_data_dir[1])
        # save merged data
        self.merged_data_dir = Utility.save_data(
            cd_all,
            'data/autosave/',
            'default',
            False)


    

    
    
        
            









