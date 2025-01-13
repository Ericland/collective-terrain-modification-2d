# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:01:16 2024

@author: ericl
"""
import numpy as np
from matplotlib import pyplot as plt
import sys 
sys.path.insert(0, 'src')

from SAWRC_World import gauss
import Utility 
from SAWRC_Planning import Flatten_Wasserstein_Planner#, Wasserstein_Multi_Robot_Planner
from SAWRC_Distributed_Planning import Wasserstein_Multi_Robot_Distributed_Planner_Ver1
from SAWRC_Analysis import Construction_Data_Analyzer#, Multi_Task_Construction_Data_Analyzer


# In[define tasks]
xc = np.arange(-5000, 5000+10, 10)
h0 = gauss(103, 300, -3500, xc)
h1 = gauss(103, 300, 3500, xc)

# print some structure info
print('max dh0:', np.amax(np.abs(np.gradient(h0, xc))))
print('max dh1:', np.amax(np.abs(np.gradient(h1, xc))))
print('w2 distance:', Utility.compute_Wp_distance_of_structures(2, h0, h1, xc))
    
    
# In[centralized planner]
run = 0
if run: 
    timeout = 10000
    planner = Flatten_Wasserstein_Planner(
        xc, h0, h1, num_robot=5, steepness_desired=np.array([-0.3, 0.3]), dW2=10,
        log_actions=False, log_progress=False, log_construction_state=True,
    )
    # planner.set_video(frame_sampling_period=10)
    planner.solve(debug=True, timeout=timeout)
    cd = Utility.load_data(*planner.merged_data_dir)
    analyzer = Construction_Data_Analyzer(cd)
    fig, ax = analyzer.plot_construction(
        len(analyzer.h_list)-1,
        show_reachable_area=False,
        show_traversable_area=False,
        show_untraversable_area=True,
    )
    
    
# In[distributed coordination]
run = 1
if run: 
    timeout = 50000
    planner = Wasserstein_Multi_Robot_Distributed_Planner_Ver1(
        xc, h0, h1, num_robot=5, dW2=1/4,
        log_actions=False, log_progress=False, log_construction_state=True,
    )
    print('w2i:', planner.sim.world.W2init)
    planner.solve(timeout=timeout)
    # planner.plot_state()
    cd = Utility.load_data(*planner.merged_data_dir) 
    analyzer = Construction_Data_Analyzer(cd)
    fig, ax = analyzer.plot_construction(
        timeout,
        show_reachable_area=False,
        show_traversable_area=False,
        show_untraversable_area=True,
    )


# In[make video]
run = 1
if run:
    analyzer.make_video_from_construction_data(
        frame_sampling_period=100,
        show_reachable_area=False,
        show_traversable_area=True,
        show_untraversable_area=True,
        video_format='gif',
    )


# In[]
plt.show()





