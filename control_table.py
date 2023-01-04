#import threading
import time


tm1=time.localtime(time.time())

import points_analysis_2D as pa
import numpy
import pandas as pd
directory = "/home/remote/xiaotian_file/data/20221129/video_5/"
file_traj_stable = directory + "txyz_stable.npy"#difference of hex_txyz & txyz_stable?
#particle id should be set as what in txyz_stable!
txyz_stable = numpy.load(file_traj_stable)#um
dpa = pa.dynamic_points_analysis_2d(txyz_stable,mode='exp')
dpa.compute_nearest_neighbor_displacements(unit='um',csv_prefix=directory)
file_ts_id_dxy = directory + 'ts_id_dxy.csv'
ts_id_dxy = pd.read_csv(file_ts_id_dxy)
dpa.monitor_neighbor_change_event(ts_id_dxy=ts_id_dxy,csv_prefix=directory)
file_list_sum_id_nb_stable = directory + 'list_sum_id_nb_stable.csv'
list_sum_id_nb_stable = pd.read_csv(file_list_sum_id_nb_stable)
dpa.bond_plot(nb_change=list_sum_id_nb_stable,data_name='video_5',account='remote')
#a0 = {('A'+unit):result.bond_length_median*lc}
#record_msd_id = pa.compute_atmsd_t_chips(0.9,True)


"""
ts_id_dxy['particle']
#pa.compute_atmsd_t_chips(interval_max=0.9,msd_per_particle=True)
path_to_folder = '/home/remote/xiaotian_file/data/20221129/video5/'
time_log = path_to_folder+'DefaultVideo_5.txt'
png_filename = path_to_folder+'particle_'+str(id)+'_'+'msds_loglog.png'
pa.plot_msd_uniform(m_msd=1,time_log=time_log,png_filename=png_filename)
"""

#time.sleep(1)
tm2=time.localtime(time.time())
#calculate the time cost
import computeTime as ct
ct.getTimeCost(tm1,tm2)

