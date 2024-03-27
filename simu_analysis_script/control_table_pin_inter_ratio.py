# import threading
import data_analysis_cycle as dac
import symmetry_transformation_v4_3.list_code_analysis as lca
import time
import computeTime as ct
import symmetry_transformation_v4_3.analysis_controller as ac
import workflow_analysis as wa
tm1 = time.localtime(time.time())


simu_index = 6482  # hc: 132-3-4
seed = 0
part = True
frame_ini = 500
frame_final = 1000
lcr = 0.81
prefix_write = "/home/remote/xiaotian_file/link_to_HDD/hoomd-examples_1/"
output_file_gsd = prefix_write+'trajectory_auto' + \
    str(int(simu_index))+'_'+str(int(seed))+'.gsd'

str_index = str(simu_index)+'_'+str(seed)
dir0 = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/record_trajectories/'+str_index+'/'
trap_filename = dir0+'traps.txt'

spi = wa.show_pin_interstitial_order_parameter_v43(simu_index, seed, 0.5)
spi.workflow_data_v43(True)  #
spi.workflow_plot_v43(True, part)

tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)
