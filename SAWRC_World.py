# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:31:42 2024

@author: ericl
"""

import numpy as np
import matplotlib.pyplot as plt 
# import copy 
# import pandas as pd 
from time import perf_counter
# import random
from tqdm import tqdm 

import Utility

# In[notes]
"""
Length unit: 1mm
Time unit: 1s
structure: a tuple (x, y)
"""


# In[]
gauss = lambda a, w, phi, x: a * np.exp(-0.5 * (x - phi)**2 / w**2)
xc = np.arange(-2500, 2500+10, 10)


# In[constructors of functions]
class Construction_Function_Constructor: 
    def __init__(self, 
                 ap = 5, 
                 wp = 100, 
                 ad = 0.02, 
                 wd = 160, 
                 rng_seed = None, 
                 ): 
        self.ap = ap 
        self.wp = wp 
        self.ad = ad 
        self.wd = wd 
        self.action_list = ['pile_left', 'pile_right', 'dig'] # list of available action functions
        self.rng = np.random.default_rng(seed=rng_seed) 
        
        
    def gaussd1(self, a, w, phi, x):
        """
        modified 1st gaussian derivative
        """
        y = (a / np.sqrt(2 * np.pi * w**2)) * (x - phi) * np.exp(-0.5 * (x - phi)**2 / w**2) 
            
        return y
    
    
    def gaussd2(self, a, w, phi, x):
        """
        modified 2nd gaussian derivative
        """
        y = (a / np.sqrt(2 * np.pi * w**2)) * ((x - phi)**2 - w**2) * np.exp(-0.5 * (x - phi)**2 / w**2) 
        
        return y
    
    
    def consFunc(self, action, position, x, noise=False):  
        """
        Compute construction function 
        The construction function models the effect of taking an action for 1s
        """
        assert action in self.action_list, "action name not found"

        # noise parameters
        nf_a = 0.1 # noise fraction of a
        nf_w = 0.1 # noise fraction of w
        noise_position = 10 # noise of position

        y = None
        if action == 'pile_right': 
            if noise: 
                y = self.gaussd1(
                    self.rng.normal(self.ap, self.ap*nf_a), 
                    self.rng.normal(self.wp, self.wp*nf_w), 
                    self.rng.normal(position, noise_position), 
                    x) 
            else: 
                y = self.gaussd1(self.ap, self.wp, position, x) 
        elif action == 'pile_left': 
            if noise: 
                y = self.gaussd1(
                    -self.rng.normal(self.ap, self.ap*nf_a), 
                    self.rng.normal(self.wp, self.wp*nf_w), 
                    self.rng.normal(position, noise_position), 
                    x) 
            else: 
                y = self.gaussd1(-self.ap, self.wp, position, x) 
        elif action == 'dig': 
            if noise: 
                y = self.gaussd1(
                    -self.rng.normal(self.ad, self.ad*nf_a), 
                    self.rng.normal(self.wd, self.wd*nf_w), 
                    self.rng.normal(position, noise_position), 
                    x) 
            else: 
                y = self.gaussd2(self.ad, self.wd, position, x)
            
        return y 
    
    
# In[]
class Random_Structure_Generator: 
    def __init__(self,
                 width = 5000, 
                 dx = 10,  
                 rng_seed = None, 
                 ):
        self.steepnessMax = 0.42
        self.travelSteepnessMax = 0.3 
        self.width = width 
        self.dx = dx 
        self.x = np.arange(-int(self.width/2), int(self.width/2)+self.dx, self.dx)
        self.rng = np.random.default_rng(seed=rng_seed) 
        self.a2w_max = self.steepnessMax * np.exp(0.5) 
        
        
    def sample_gaussians(self): 
        w = self.rng.uniform(self.width/10 * 0.25, self.width/10 * 0.75)
        a = w * self.rng.uniform(-1, 1) * self.a2w_max 
        limit = self.width/2 - 5*w
        phi = self.rng.uniform(-limit, limit)
        paras = np.array([a, w, phi])
        
        return paras 
    
    
    def evaluate_h(self, paras_arr, x): 
        m, n = paras_arr.shape 
        h = 0
        for ii in range(m): 
            h += gauss(*paras_arr[ii,:], x) 
        
        return h 
    
    
    def plot_h(self, paras_arr, x): 
        m, n = paras_arr.shape 
        h = 0 
        fig, ax = plt.subplots()
        for ii in range(m): 
            h += gauss(*paras_arr[ii,:], x) 
            ax.plot(x, gauss(*paras_arr[ii,:], x), '--', alpha=0.5) 
        ax.plot(x, h)
        ax.set_aspect(5)
        ax.set_ylim([-500,500])
        
        return fig, ax 
    
    
    def get_parameters(self, make_plot=False): 
        # sample Gaussian function parameters 
        num_func = self.rng.integers(1, 8)
        paras_arr = np.zeros((num_func, 3))
        for ii in range(num_func): 
            paras_arr[ii,:] = self.sample_gaussians()
            
        # scale the structure if the max steepness of the structure exceeds steepnessMax
        # If yes, scale it s.t. that max steepness is 0.99 * steepnessMax
        h = self.evaluate_h(paras_arr, self.x)
        dh_abs_max = np.amax(np.abs(np.gradient(h, self.x)))
        if dh_abs_max > self.steepnessMax:
            paras_arr[:,0] *= self.steepnessMax*0.99/dh_abs_max
        h = self.evaluate_h(paras_arr, self.x) 
        dh_abs_max = np.amax(np.abs(np.gradient(h, self.x))) 
        
        # check max steepness
        if dh_abs_max > self.steepnessMax: 
            paras_arr = None 
            print('Random structure formation fails') 
            
        # plot gaussians and the summation of gaussians 
        if make_plot: 
            self.plot_h(paras_arr, self.x)
        
        return paras_arr 
    
    
    def get_h(self):
        h = None
        paras_arr = self.get_parameters() 
        if paras_arr.any() != None: 
            h = self.evaluate_h(paras_arr, self.x) 
        
        return h
    
    
    def get_h_pair(self): 
        h0 = self.get_h()
        h1 = self.get_h()
        # equalize the volume
        v0 = np.trapz(h0, self.x) 
        v1 = np.trapz(h1, self.x) 
        if abs(v0) > abs(v1): 
            h0 *= (v1/v0) 
        else:
            h1 *= (v0/v1) 
        hmin = min(np.amin(h0), np.amin(h1))
        # hmin cannot be lower than the default base height (-1000)
        # if yes set hmin to 0
        if hmin < -750:
            h0 -= hmin 
            h1 -= hmin 
        
        return h0, h1 
    
    
    def get_h_pairs(self, num_pair=16): 
        hp_list = []
        for pp in tqdm(range(num_pair)): 
            hp = self.get_h_pair()
            hp_list.append(hp) 
        
        return hp_list


    def get_h_pair_categorized(self, category='easy'):
        """
        generate pair of structures based on the given category of difficulty
        List of categories:
            easy: traversable to traversable
            medium: untraversable to traversable or vice versa
            hard: untraversable to untraversable
        """
        # keep generating random structures until finding one that fits the given category
        while True:
            h0 = self.get_h()
            h1 = self.get_h()
            # equalize the volume
            v0 = np.trapz(h0, self.x)
            v1 = np.trapz(h1, self.x)
            if abs(v0) > abs(v1):
                h0 *= (v1 / v0)
            else:
                h1 *= (v0 / v1)
            # identify the current category
            dh0maxabs = np.amax(np.abs(np.gradient(h0, self.x)))
            dh1maxabs = np.amax(np.abs(np.gradient(h1, self.x)))
            dhma = np.array([dh0maxabs, dh1maxabs])
            if np.all(dhma > self.travelSteepnessMax):
                category_cur = 'hard'
            elif np.all(dhma <= self.travelSteepnessMax):
                category_cur = 'easy'
            else:
                category_cur = 'medium'
            # stop if the structure fits the category
            if category_cur == category:
                break
        # hmin cannot be lower than the default base height (-1000)
        # if yes set hmin to 0
        hmin = min(np.amin(h0), np.amin(h1))
        if hmin < -750:
            h0 -= hmin
            h1 -= hmin

        return h0, h1


    def get_h_pairs_categorized(self, category_list):
        hp_list = []
        for pp in tqdm(range(len(category_list))):
            hp = self.get_h_pair_categorized(category=category_list[pp])
            hp_list.append(hp)

        return hp_list
    
    
# In[]
class World: 
    def __init__(self,
                 height_base = 1000, # base height of the construction area (default 1m)
                 width = 5000, # width of the construction site (default 5m)
                 dx = 10, # resolution/mesh size (default 10mm) 
                 structure_initial = None, 
                 structure_goal = None, 
                 apply_modification_noise = True, 
                 ): 
        # structure parameters
        self.steepnessMax = 0.42
        self.height_base = height_base 
        self.width = width 
        self.dx = dx 
        self.x = np.arange(-int(self.width/2), int(self.width/2)+self.dx, self.dx)
        self.structure_initial = structure_initial 
        self.structure_goal = structure_goal 
        self.apply_modification_noise = apply_modification_noise 
        
        self.cfc = Construction_Function_Constructor()
        self.set_structures() 
        
        # time 
        self.dt = 1
        self.reset_timer()
        
        
    def reset_timer(self): 
        self.stepCounter = 0
        self.simTime = 0
        self.runTime = 1e-6
        self.runTimeStart = perf_counter()
        
        
    def update_timer(self): 
        self.stepCounter += 1
        self.simTime = self.stepCounter * self.dt
        self.runTime = perf_counter() - self.runTimeStart
        
        
    def check_max_steepness(self, y): 
        dh_abs = np.abs(np.gradient(y, self.x))
        if np.amax(dh_abs) > self.steepnessMax: 
            results = ('violated', np.amax(dh_abs))
        else: 
            results = ('pass', None)
            
        return results
    
    
    def preserve_volume(self, y): 
        yp = np.copy(y)
        yp += (self.volume - np.trapz(yp, self.x)) / self.width 
        
        return yp
    
    
    def get_W2(self): 
        W2cur = Utility.compute_Wp_distance_of_structures(2, self.yg, self.y, self.x) 
        
        return W2cur 
    
    
    def get_progress(self): 
        W2cur = self.get_W2() 
        progress = 1 - W2cur / self.W2init 
        
        return progress 
        
        
    def set_structures(self): 
        # set initial structure 
        if self.structure_initial == None: 
            self.yi = self.x * 0.0 + self.height_base 
        else: 
            self.yi = np.interp(self.x, *self.structure_initial) + self.height_base
            
        # set goal structure 
        if self.structure_goal == None: 
            self.yg = self.x * 0.0 + self.height_base 
        else: 
            self.yg = np.interp(self.x, *self.structure_goal) + self.height_base 
            
        # get distance between hi and hg 
        self.W2init = Utility.compute_Wp_distance_of_structures(2, self.yg, self.yi, self.x) 
            
        # equalize the total volume of initial and goal structures 
        self.volume = np.trapz(self.yg, self.x) # use the volume of the goal structure as the fixed volume 
        self.yi = self.preserve_volume(self.yi)
            
        # set the current structure to the initial structure 
        self.y = np.copy(self.yi) 
        
        # check max steepness 
        res = self.check_max_steepness(self.yi) 
        assert res[0] == 'pass', 'invalid initial structure (max steepness violated)' 
        res = self.check_max_steepness(self.yg) 
        assert res[0] == 'pass', 'invalid goal structure (max steepness violated)'
        
        # find the limited region where robot can travel or modify the structure 
        self.limit = self.width/2 - 5 * self.cfc.wp 
        self.x_limited = self.x[np.where((self.x<self.limit)&(self.x>-self.limit))] 
        
        # get ylim for plotting based on the initial and goal structure 
        y_range = np.array([min(np.amin(self.yi), np.amin(self.yg)), max(np.amax(self.yi), np.amax(self.yg))])
        self.ylim_for_plot = (y_range[1]-y_range[0])*0.1*np.array([-1,1]) + y_range 
    
    
    def modify(self, action, position):
        """
        Modify the structure.
        """
        # get construction function 
        yc = self.cfc.consFunc(action, position, self.x, self.apply_modification_noise) 
        
        # compute the attenuation factor alpha 
        ym = self.y + yc
        res = self.check_max_steepness(ym)
        if res[0] == 'pass': 
            alpha = 1 
        else: 
            dym = np.gradient(ym, self.x) 
            dym_abs = np.abs(dym)
            ind = np.argmax(dym_abs)
            dh = np.gradient(self.y, self.x) 
            df = np.gradient(yc, self.x) 
            alpha = (np.sign(dym[ind]) * self.steepnessMax - dh[ind]) / df[ind]
            
        # compute modified height function 
        res = self.check_max_steepness(self.y + alpha * yc ) # validate modified height function again
        if res[0] == 'pass': 
            pass 
        else:
            alpha = 0 
        self.y += alpha * yc # modify the structure 
        
        
    def plot_initial_and_goal_structures(self): 
        fig, ax = plt.subplots(dpi=150) 
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.plot(self.x, self.yi, label='initial structure') 
        ax.plot(self.x, self.yg, label='goal structure')
        ax.legend()
        ax.set_ylim(self.ylim_for_plot) 
        
        return fig, ax
    
    
    def plot_structure(self): 
        fig, ax = plt.subplots(dpi=150)
        ax.plot(self.x, self.y)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title('step: ' + str(self.stepCounter))
        ax.set_ylim(self.ylim_for_plot) 
        
        return fig, ax 
    
    
    def plot_initial_goal_current_structures(self): 
        fig, ax = plt.subplots(dpi=150) 
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.plot(self.x, self.yi, '--', label='initial structure', alpha=0.5) 
        ax.plot(self.x, self.yg, '--', label='goal structure', alpha=0.5)
        ax.plot(self.x, self.y, label='current structure', color='black')
        ax.set_title('step: ' + str(self.stepCounter))
        ax.legend()
        ax.set_ylim(self.ylim_for_plot) 
        
        return fig, ax
    
    
# In[]
class Robot: 
    def __init__(self,
                 world, 
                 pos_x = 0, 
                 name = 'robot', 
                 ): 
        self.world = world 
        self.pos_x = pos_x 
        self.pos_y = self.world.y[np.where(self.world.x==self.pos_x)][0]
        self.name = name 
        
        self.travelSteepnessMax = 0.3 # max steepness the robot can travel (from SAW paper)
        self.apply_steepness_constraint = False 
        self.width = 383 # width/length of the robot in mm 
        
        
    def update_pos_y(self): 
        self.pos_y = self.world.y[np.where(self.world.x==self.pos_x)][0]
        
        
    def get_traversable_area(self, h=None, pos_x=None, climb_down_allowed=True): 
        if np.all(h) == None: 
            h = self.world.y 
        if pos_x == None: 
            pos_x = self.pos_x 
        
        if self.apply_steepness_constraint: 
            dy = np.gradient(h, self.world.x) 
            indr = np.where(self.world.x == pos_x)[0][0]
            if climb_down_allowed: 
                dy[:indr] *= -1 
                dy[indr] = 0 
                x_traversable_candidate = self.world.x[np.where((self.world.x<self.world.limit)
                                                                &(self.world.x>-self.world.limit)
                                                                &(dy<self.travelSteepnessMax))] 
            else: 
                dy[indr] = 0 
                dyabs = np.abs(dy) 
                x_traversable_candidate = self.world.x[np.where((self.world.x<self.world.limit)
                                                                &(self.world.x>-self.world.limit)
                                                                &(dyabs<self.travelSteepnessMax))] 
            if pos_x in x_traversable_candidate: 
                indr = np.where(x_traversable_candidate==pos_x)[0][0]
                inds = np.arange(x_traversable_candidate.size)
                xf = x_traversable_candidate - pos_x - (inds - indr) * self.world.dx 
                x_traversable = x_traversable_candidate[np.where(xf==0)]
            else: 
                x_traversable = np.array([pos_x])
        else: 
            x_traversable = np.copy(self.world.x_limited) 
            
        # debugging only 
        # fig, ax = plt.subplots()
        # ax.plot(self.world.x, dy) 
        # ax.scatter(x_traversable_candidate, x_traversable_candidate*0-0.1) 
        # ax.scatter(x_traversable, x_traversable*0-0.2)
        # ax.plot(self.world.x, self.world.x*0 + self.travelSteepnessMax)
        # print(np.amin(x_traversable), np.amax(x_traversable))
            
        return x_traversable 
        
        
    def move(self, pos_x_goal): 
        # check whether the robot can move to the new location 
        if self.apply_steepness_constraint: 
            if pos_x_goal == self.pos_x: 
                res = 'success' 
            else:     
                if pos_x_goal in self.get_traversable_area(): 
                    res = 'success' 
                else: 
                    res = 'failure'
        else: 
            res = 'success' 
        
        # update robot position 
        if res == 'success': 
            self.pos_x = pos_x_goal 
            self.update_pos_y()
        
        return res 
    
    
    def modify(self, action): 
        self.world.modify(action, self.pos_x) 
    
    









