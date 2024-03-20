# import threading
import data_analysis_cycle as dac
import symmetry_transformation_v4_3.list_code_analysis as lca
import time
import computeTime as ct
import symmetry_transformation_v4_3.analysis_controller as ac
import workflow_analysis as wa
tm1 = time.localtime(time.time())


simu_index = 342  # hc: 132-3-4
seed = 0
frame_ini = 100
frame_final = 1000
prefix_write = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/"
output_file_gsd = prefix_write+'trajectory_auto' + \
    str(int(simu_index))+'_'+str(int(seed))+'.gsd'

str_index = str(simu_index)+'_'+str(seed)
dir0 = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/record_trajectories/'+str_index+'/'
trap_filename = dir0+'traps.txt'
preproceed = True  # False
if preproceed:
    # get 'txyz_stable.npy'
    gdf = ac.get_data_from_a_trajectory(simu_index, seed, gsd_filename=output_file_gsd, stable=True)
    # get 'list_sum_id_nb_stable.csv'
    sbt = wa.show_bonds_transition_from_hex_to_honeycomb()
    sbt.get_bond_plot(dir0, None, trap_filename, 1.0)  # monitor_neighbor_change_event()

daw = dac.data_analysis_workflow()
daw.get_defect_motion(dir0, frame_ini, frame_final, str_index, trap_filename, 1.0)
tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)

"""
kg
342_0_
kp
622_0_58_59
bpm.restrict_axis_limitation([-12.5, 2.5], [-12.5, 2.5])
622_0_59_60
bpm.restrict_axis_limitation([-5, 10], [-10, 5])
hp
6482_0_1021_1022 
bpm.restrict_axis_limitation([2, 17], [-15, 0])
particle_size = 200  # 50
circle_size = particle_size
lw = 4  # 2  # circle linewidths
circle_color = np.array([100,143,255])/255.0
arrow_color = circle_color
back_ratio = 0.5  # 0
arrow_scale = 0.8  # 1
arrow_width = 0.015
hp
6482_0_829_830 
bpm.restrict_axis_limitation([5, 20], [-15, 0])
particle_size = 200  # 50
circle_size = particle_size
lw = 4  # 2  # circle linewidths
circle_color = 'limegreen'  # 'orange'
arrow_color = circle_color
back_ratio = 0.5  # 0
arrow_scale = 0.8  # 1
arrow_width = 0.015

hc
132_0_3-4
bpm.restrict_axis_limitation([5, 20], [5, 20])
particle_size = 200  # 50
circle_size = particle_size
lw = 4  # 2  # circle linewidths
circle_color = np.array([100,143,255])/255.0
arrow_color = circle_color  # 'limegreen'
back_ratio = 0.5  # 0
arrow_scale = 0.8  # 1
arrow_width = 0.013
bond_width = 2#3

hc
132_0_3-4
bpm.restrict_axis_limitation([-10, 5], [-13, 2])
particle_size = 200  # 50
circle_size = particle_size
lw = 4  # 2  # circle linewidths
circle_color = np.array([100,143,255])/255.0
arrow_color = circle_color  # 'limegreen'
back_ratio = 0.5  # 0
arrow_scale = 0.8  # 1
arrow_width = 0.013
bond_width = 2#3
"""
