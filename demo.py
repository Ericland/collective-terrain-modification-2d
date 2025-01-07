# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:01:16 2024

@author: ericl
"""
import numpy as np
from matplotlib import pyplot as plt

from SAWRC_World import gauss, xc 
import Utility 
from SAWRC_Planning import Wasserstein_Multi_Robot_Planner, Flatten_Wasserstein_Planner 
from SAWRC_Distributed_Planning import Wasserstein_Multi_Robot_Distributed_Planner_Ver1 
from SAWRC_Analysis import Construction_Data_Analyzer#, Multi_Task_Construction_Data_Analyzer


# In[customized tasks]
run = 1 
if run: 
    # traversable structure to traversable structure
    run = 0
    if run: 
        h0 = gauss(103, 300, -1000, xc)
        h1 = gauss(103, 300, 1000, xc) 
    run = 1
    if run: 
        h0 = gauss(70, 300, -1000, xc) + gauss(70, 300, 1000, xc)
        h1 = gauss(140, 300, 0, xc)
    run = 0
    if run:
        xc = np.arange(-5000, 5000+10, 10)
        h0 = gauss(103, 300, -1000-2500, xc)
        h1 = gauss(103, 300, 1000+2500, xc)
    # traversable structure to untraversavle structure
    run = 0
    if run: 
        h0 = gauss(103, 300, -1000, xc) + gauss(103, 300, 1000, xc)
        h1 = gauss(206, 300, 0, xc)
    # untraversable structure to traversavle structure
    run = 0
    if run: 
        h0 = gauss(206, 300, 0, xc)
        h1 = gauss(103, 300, -1000, xc) + gauss(103, 300, 1000, xc)
    # untraversable structure to untraversable structure 
    run = 0 
    if run: 
        h0 = (gauss(206, 300, -500, xc))
        h1 = (gauss(206, 300, 500, xc))
    run = 0 
    if run: 
        h0 = (gauss(206, 300, -1000, xc))
        h1 = (gauss(206, 300, 1000, xc))
        
    print('max dh0:', np.amax(np.abs(np.gradient(h0, xc))))
    print('max dh1:', np.amax(np.abs(np.gradient(h1, xc))))
        
        
# In[randomly generated tasks]
run = 0 
if run: 
    hp_list, rsg = Utility.load_data('data/simulation/', 'hps.pkl') 
    h0, h1 = hp_list[20] 
    xc = np.copy(rsg.x)
    
    
# In[Flatten_Wasserstein_Planner]
run = 0
if run: 
    timeout = 10000
    planner = Flatten_Wasserstein_Planner(xc, h0, h1, num_robot=10, steepness_desired=np.array([-0.3, 0.3]), dW2=1, 
                                          apply_steepness_constraint=True, apply_modification_noise=True, 
                                          log_actions=False, log_progress=False, log_construction_state=True)
    # planner.set_video(frame_sampling_period=10)
    planner.solve(debug=True, timeout=timeout)
    cd = Utility.load_data(*planner.merged_data_dir)
    analyzer = Construction_Data_Analyzer(cd)
    fig, ax = analyzer.plot_construction(len(analyzer.h_list)-1, show_reachable_area=False, show_traversable_area=False, show_untraversable_area=True)
    fig.set_size_inches([9, 3])
    # analyzer.make_video_from_construction_data(frame_sampling_period=10,
    #                                            show_reachable_area=False,
    #                                            show_traversable_area=True,
    #                                            show_untraversable_area=True,
    #                                            )
    
    
# In[Wasserstein_Multi_Robot_Distributed_Planner_Ver1] 
run = 1
if run: 
    timeout = 10000
    planner = Wasserstein_Multi_Robot_Distributed_Planner_Ver1(xc, h0, h1, num_robot=5, dW2=1/4, 
                                                               apply_steepness_constraint=True, apply_modification_noise=True, 
                                                               log_actions=False, log_progress=False, log_construction_state=True)
    print('w2i:', planner.sim.world.W2init)
    planner.solve(timeout=timeout)
    # planner.plot_state()
    cd = Utility.load_data(*planner.merged_data_dir) 
    analyzer = Construction_Data_Analyzer(cd)
    fig, ax = analyzer.plot_construction(timeout, show_reachable_area=False, show_traversable_area=False, show_untraversable_area=True)
    analyzer.make_video_from_construction_data(
        frame_sampling_period=100, 
        show_reachable_area=False, 
        show_traversable_area=False, 
        show_untraversable_area=True, 
        video_format='gif', 
        )


# In[]
plt.show()





