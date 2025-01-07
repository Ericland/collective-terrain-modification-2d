# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:25:22 2024

@author: ericl
"""
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from time import perf_counter

import Utility
from SAWRC_World import World, Robot


# In[] 
def plot_h_pair(x, hp): 
    h0, h1 = hp 
    fig, ax = plt.subplots() 
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.plot(x, h0, label='h0') 
    ax.plot(x, h1, label='h1')
    ax.legend()
    
    return fig, ax 


def plot_h_pairs(x, hp_list, m=4, n=4):  
    """
    Plot first mxn pairs
    """
    fig, axs = plt.subplots(m, n, constrained_layout=True)
    fig.dpi = 150
    ind = 0
    for ii in range(m): 
        for jj in range(n): 
            ax = axs[ii, jj] 
            h0, h1 = hp_list[ind] 
            ax.plot(x, h0, label='h0') 
            ax.plot(x, h1, label='h1') 
            ax.set_title(str(ind))
            ax.set_xticks([])
            ax.set_yticks([])
            ind += 1
            
    return fig, axs


# In[]
class Construction_Data_Analyzer:
    def __init__(self, construction_data):
        self.x, self.hi, self.hg, construction_state_list = construction_data
        self.step_num = len(construction_state_list)
        self.h_list = []
        self.xrs_list = []
        self.yrs_list = []
        for h, xrs in construction_state_list:
            self.h_list.append(h)
            self.xrs_list.append(xrs)
            self.yrs_list.append([h[np.where(self.x==xr)][0] for xr in xrs])
        self.hf = self.h_list[-1]
        self.num_robot = len(self.xrs_list[0])

        # construction progress
        self.w2i = Utility.compute_Wp_distance_of_structures(2, self.hi, self.hg, self.x)
        self.w2f = Utility.compute_Wp_distance_of_structures(2, self.hf, self.hg, self.x)
        self.progress_history = []
        for h in self.h_list:
            w2 = Utility.compute_Wp_distance_of_structures(2, h, self.hg, self.x)
            self.progress_history.append(1 - w2/self.w2i)
        self.final_progress = self.progress_history[-1] 
        
        # volume of moved material
        self.vdiff = np.trapz(np.abs(self.hi - self.hg), self.x)
        self.vomm_net = np.trapz(np.abs(self.hi - self.hf), self.x) 

        # parameters for plotting
        self.y_range = np.array([min(np.amin(self.hi), np.amin(self.hg)), max(np.amax(self.hi), np.amax(self.hg))])
        self.x_range = np.array([self.x[0], self.x[-1]])
        self.ylim_for_plot = (self.y_range[1] - self.y_range[0]) * 0.1 * np.array([-1, 1]) + self.y_range

        # tool
        self.world_tool = World(width=np.amax(self.x)*2)
        self.robot_tool = Robot(self.world_tool)


    def plot_task(self):
        fig, ax = plot_h_pair(self.x, (self.hi, self.hg))

        return fig, ax


    def get_settling_time(self, tolerance=0.01, make_plot=False):
        # smooth the progress history data
        ph_arr = np.array(self.progress_history)
        t_arr = np.arange(self.step_num)
        filter_length = int(self.step_num / 10)
        tfil, phfil = Utility.moving_average(t_arr, ph_arr, filter_length) # filter data
        dphfil = np.gradient(phfil, tfil) # derivative of smooth data

        # find a window over which the average derivative is less than the steady_state_derivative
        # the steady state value is the mean of the window of filtered progress history
        steady_state_derivative = 1e-06
        steady_state_window_size = int(len(dphfil) * 0.1)
        steady_state_progress = phfil[-1]  # default steady state value
        while steady_state_window_size > 1:
            if np.abs(np.mean(dphfil[-steady_state_window_size:])) < steady_state_derivative:
                steady_state_progress = np.mean(phfil[-steady_state_window_size:])
                break
            else:
                steady_state_window_size -= 1

        # find the settling time at which progress reaches the steady state value with 1% error
        tolerance_band = tolerance * steady_state_progress
        settling_time = 0
        for ii in range(self.step_num):
            if abs(self.progress_history[ii] - steady_state_progress) > tolerance_band:
                settling_time = ii

        # plot the analysis
        if make_plot:
            fig, ax = plt.subplots()
            ax.plot(t_arr, ph_arr, label='progress')
            ax.plot(tfil, phfil, label='filtered data')
            ax.plot(t_arr, t_arr * 0 + steady_state_progress, label='steady state progress')
            ax.scatter(settling_time, ph_arr[settling_time], color='red', label='settling point')
            ax.legend()

        return settling_time, steady_state_progress


    def get_construction_time(self):
        ph_arr = np.array(self.progress_history)
        t_arr = np.where(ph_arr >= 0.9)[0]
        if t_arr.size > 0:
            construction_time = t_arr[0]
        else:
            construction_time = None # task is not complete

        return construction_time


    def get_traveling_distance_history(self):
        traveling_distance_history = []
        traveling_distance = 0
        traveling_distance_history.append(traveling_distance)
        for ii in range(self.step_num - 1):
            h = self.h_list[ii+1]
            xrs_old = np.array(self.xrs_list[ii])
            xrs = np.array(self.xrs_list[ii+1])
            if np.any(xrs_old != xrs):
                xr_old = xrs_old[np.where(xrs_old != xrs)][0]
                xr = xrs[np.where(xrs_old != xrs)][0]
                xrl = [xr_old, xr]
                xrl.sort()
                xs = self.x[np.where((self.x>=xrl[0])&(self.x<=xrl[1]))]
                ys = h[np.where((self.x>=xrl[0])&(self.x<=xrl[1]))]
                traveling_distance += Utility.compute_arc_length_piecewise(xs, ys)
            traveling_distance_history.append(traveling_distance)

        return traveling_distance_history


    def get_volume_of_moved_material_history(self):
        vomm_history = []
        vomm = 0
        vomm_history.append(vomm)
        for ii in range(self.step_num - 1):
            vomm += np.trapz(np.abs(self.h_list[ii+1] - self.h_list[ii]), self.x)
            vomm_history.append(vomm)

        return vomm_history


    def get_mistake_timestamp(self):
        """
        Get the timestamp of actions that lead to decrease in progress
        """
        timestamp_list = []
        for ii in range(self.step_num - 1):
            if self.progress_history[ii + 1] < self.progress_history[ii]:
                timestamp_list.append(ii + 1)

        return timestamp_list


    def get_untraversable_structure_timestamp(self):
        """
        Get the timestamp when structure is untraversable
        """
        timestamp_list = []
        for ii in range(self.step_num):
            h = self.h_list[ii]
            if np.amax(np.abs(np.gradient(h, self.x))) > self.robot_tool.travelSteepnessMax:
                timestamp_list.append(ii)

        return timestamp_list


    def plot_construction(self, time, 
                          show_reachable_area=True,
                          show_traversable_area=True,
                          show_untraversable_area=True, 
                          fill_between=True, 
                          subplots = None, 
                          ):
        """
        Plot the construction state at given time
        """
        if subplots == None: 
            figsize = np.array([np.ptp(self.x_range), np.ptp(self.y_range)]) * np.array([18, 3]) / np.array([10000, 100])
            fig, ax = plt.subplots(dpi=150, figsize=figsize)
        else: 
            fig, ax = subplots 
        # initial, goal and current structures
        ax.plot(self.x, self.hi, '--', label='initial structure', alpha=0.5)
        ax.plot(self.x, self.hg, '--', label='goal structure', alpha=0.5)
        ax.plot(self.x, self.h_list[time], label='current structure', color='black')
        if fill_between: 
            ax.fill_between(self.x, self.h_list[time], 0, color='silver')
        # robot positions, traversable area, movable area
        xc_reachable_list = []
        xc_traversable_list = []
        for rr, (xr, yr) in enumerate(zip(self.xrs_list[time], self.yrs_list[time])):
            ax.scatter(xr, yr, label='robot #' + str(rr), marker='s', zorder=10)
            xc_reachable_list.append(self.robot_tool.get_traversable_area(self.h_list[time], xr))
            xc_traversable_list.append(self.robot_tool.get_traversable_area(self.h_list[time], xr, climb_down_allowed=False))
        if show_reachable_area:
            xc_union = Utility.get_union(xc_reachable_list)
            yc_union = np.interp(xc_union, self.x, self.h_list[time])
            ax.scatter(xc_union, yc_union, color='yellow', label='reachable area', marker='.')
        if show_traversable_area:
            xc_union = Utility.get_union(xc_traversable_list)
            yc_union = np.interp(xc_union, self.x, self.h_list[time])
            ax.scatter(xc_union, yc_union, color='green', label='traversable area', marker='.')
        if show_untraversable_area:
            dhabs = np.abs(np.gradient(self.h_list[time], self.x))
            xc_union = self.x[dhabs > self.robot_tool.travelSteepnessMax]
            yc_union = np.interp(xc_union, self.x, self.h_list[time])
            ax.scatter(xc_union, yc_union, color='crimson', label='untraversable area', marker='x')

        # axis
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title('step: ' + str(time))
        ax.legend(loc='upper right')
        ax.set_ylim(self.ylim_for_plot)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        return fig, ax


    def make_video_from_construction_data(self, frame_sampling_period=10, 
                                          show_reachable_area=True,
                                          show_traversable_area=True,
                                          show_untraversable_area=True, 
                                          fill_between=True, 
                                          video_format='mp4', 
                                          ):
        image_loc = 'video/video_image/'
        delete_images_afterwards = True
        frame_index = 0
        for tt in tqdm(range(self.step_num)):
            if tt%frame_sampling_period == 0 or tt == self.step_num-1:
                self.plot_construction(tt, show_reachable_area, show_traversable_area, show_untraversable_area, fill_between) 
                file_name = image_loc + 'img_' + str(frame_index) + '.png'
                plt.savefig(file_name, dpi=200)
                plt.close()
                frame_index += 1 
        Utility.make_video(video_format=video_format, delete_images_afterwards=delete_images_afterwards)


# In[]
class Multi_Task_Construction_Data_Analyzer:
    def __init__(self, data_loc, file_name, save_loc='data/analysis/'):
        self.data_loc = data_loc
        self.file_name = file_name
        self.save_loc = save_loc

        # load data
        tic = perf_counter()
        print('loading data...')
        self.construction_data_list = Utility.load_data(data_loc, file_name)
        toc = perf_counter()
        print('time consumed (s):', toc - tic)

        # get basic simulation info
        self.num_sim = len(self.construction_data_list)
        str_list = file_name[:-4].split('_')
        self.alg = str_list[0]
        self.task = str_list[1].split('=')[1]
        self.num_task = self.task.split('x')[1]
        self.num_trial_per_task = int(str_list[2].split('=')[1])
        self.num_robot = int(str_list[3].split('=')[1])
        if 'dW2' in str_list[4]:
            self.dw2 = float(str_list[4].split('=')[1])
        else:
            self.dw2 = None


    def analyze(self, make_plot=False):
        w2i_list = [] # initial w2 distance
        pf_list = [] # final progress
        pss_list = [] # steady state progress
        tc_norm_list = [] # normalized construction time (construction time / w2i)
        vomm_tc_norm_list = [] # normalized volume of moved material up to construction time (vomm_tc / vdiff)
        td_tc_norm_list = [] # normalized traveling distance up to construction time (td_tc / w2i)
        freq_mistake_tc_list = [] # frequency of mistakes up to construction time
        num_untraversable = 0 # number of trials when robots create untraversable structures
        num_untraversable_complete = 0 # number of trials when robots create untraversable structures and construction is complete
        print('analyzing data...')
        for ii in tqdm(range(self.num_sim)):
            construction_data = self.construction_data_list[ii]
            cda = Construction_Data_Analyzer(construction_data)
            tss, pss = cda.get_settling_time()
            tc = cda.get_construction_time()
            tdh = cda.get_traveling_distance_history()
            vommh = cda.get_volume_of_moved_material_history()
            vdiff = cda.vdiff
            w2i = cda.w2i
            tmistake_arr = np.array(cda.get_mistake_timestamp())
            tuntraversable = cda.get_untraversable_structure_timestamp()
            # collect info
            if ii % self.num_trial_per_task == 0:
                w2i_list.append(w2i)
            pf_list.append(cda.final_progress)
            pss_list.append(pss)
            if len(tuntraversable) > 0:
                num_untraversable += 1
                if tc != None:
                    num_untraversable_complete += 1
            if tc != None:
                tc_norm_list.append(tc / w2i)
                vomm_tc_norm_list.append(vommh[tc] / vdiff)
                td_tc_norm_list.append(tdh[tc] / w2i)
                freq_mistake_tc_list.append(np.where(tmistake_arr <= tc)[0].size /tc)
            else:
                tc_norm_list.append(None)
                vomm_tc_norm_list.append(None)
                td_tc_norm_list.append(None)
                freq_mistake_tc_list.append(None) 
        pf_arr = np.array(pf_list)
        success_rate = np.sum(pf_arr >= 0.9) / pf_arr.size # success rate (percentage of simulations with progress > 90%), float

        analysis_results = {
            'simulation info': (self.num_sim, self.alg, self.task, self.num_task, self.num_trial_per_task, self.num_robot, self.dw2),
            'construction success rate': success_rate,
            'initial w2 distance': w2i_list,
            'final progress': pf_list,
            'steady state progress': pss_list,
            'normalized construction time': tc_norm_list,
            'normalized volume of moved material up to construction time': vomm_tc_norm_list,
            'normalized robot traveling distance up to construction time': td_tc_norm_list,
            'frequency of mistakes up to construction time': freq_mistake_tc_list,
            'number of trials when robots create untraversable structures': num_untraversable,
            'number of trials when robots create untraversable structures and construction is complete': num_untraversable_complete,
        }

        if make_plot:
            for data_name, data in analysis_results.items():
                if data_name in [
                    'initial w2 distance',
                    'final progress',
                    'steady state progress',
                    'normalized construction time',
                    'normalized volume of moved material up to construction time',
                    'normalized robot traveling distance up to construction time',
                    'frequency of mistakes up to construction time',
                ]:
                    if None in data:
                        data_arr = np.array(data)
                        data = data_arr[np.where(data_arr != None)]
                    fig, ax = plt.subplots(dpi=150)
                    ax.boxplot(data, showfliers=False, showmeans=True, meanline=True)
                    ax.set_title(data_name)
                    ax.set_xlabel(self.file_name)
                    plt.show()
                else:
                    print(data_name + ':', data)

        file_name = self.file_name[:-4] + '_analysis.pkl'
        Utility.save_data(analysis_results, self.save_loc, file_name)
        print()

        return analysis_results







