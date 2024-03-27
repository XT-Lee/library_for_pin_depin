# import threading
import matplotlib.pyplot as plt
import data_analysis_cycle as dac
import symmetry_transformation_v4_3.list_code_analysis as lca
import time
import computeTime as ct
import symmetry_transformation_v4_3.analysis_controller as ac
import workflow_analysis as wa
import numpy as np
import pandas as pd
import data_decorate as dd
import symmetry_transformation_v4_3.simulation_controller as sc
tpn = sc.simulation_controller_type_n_part_traps()
list_lcr0 = tpn.get_type_n_lcr0()
tm1 = time.localtime(time.time())
n_simu = 9
list_simu_index = np.linspace(532, 772, n_simu, dtype=int)
colors = ['brown', 'r', 'orange', 'gold', 'g', 'c', 'b', 'purple', 'k']  # limegreen
lcr = np.linspace(0.83, 0.91, n_simu)
lcr = np.round(lcr, 2)
type_n = 8
unit_area0_trap = list_lcr0[type_n-1]*list_lcr0[type_n-1]
unit_area_trap = np.square(lcr)
rho = unit_area0_trap / unit_area_trap
rho = np.round(rho, 2)
d2 = dd.data_decorator()
fig, ax = plt.subplots()


for i in range(n_simu):
    simu_index = list_simu_index[i]
    seed = 0
    part = True
    prefix_write = "/home/remote/xiaotian_file/link_to_HDD/hoomd-examples_1/"
    str_index = str(simu_index)+'_'+str(seed)
    dir0 = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/record_trajectories/'+str_index+'/'
    csv_filename = dir0+'T_VS_CN_k_cutindex'+str_index+'.csv'
    df = pd.read_csv(csv_filename)
    frames = df['t'].values
    cnks = df['cn4'].values

    n_data = 10
    # list_index,t_trans_ratio_decorated = d2.coarse_grainize_and_average_data_log(t_trans_ratio,n_data)
    frames, cnks = d2.coarse_grainize_and_average_data_log(
        cnks, n_data)
    ax.semilogx(frames+1, cnks, c=colors[i], marker='o', label=str(rho[i]))

ax.legend()
png_filename = 'multi_cn3_kp.png'
plt.savefig(png_filename)
plt.close()

tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)
