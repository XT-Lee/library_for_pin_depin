import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import computeTime as ct
# import symmetry_transformation_v4_3.list_code_analysis as lca
import time
tm1 = time.localtime(time.time())


def draw(x, y, z, png_filename):
    fig, ax = plt.subplots()
    # ax.scatter(pos[:,0],pos[:,1],c='k')
    sc = ax.scatter(x, y, c=z)
    ax.set_xlabel('$rho$')
    ax.set_ylabel('$U_{trap}$')
    fig.colorbar(sc)
    fig.savefig(png_filename)
    plt.close()  # fig


"""agf = lca.analyze_a_series_of_gsd_file_dynamic()
agf.get_transformation_velocity_scan_csv()  # get_cnks_vs_t_from_csv()
"""
prefix_read = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430'
dir_h = '/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'
dir_hp = '/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
dir_kg = '/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv'
dir_kgp = '/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv'
# gsd_reocrd_location: example_1,1,1,1; dir_hp moved to exple_1 and remain only seed1-9
dirs = [dir_h, dir_hp, dir_kg, dir_kgp]
list_simu_index_head = [3, 6373, 243, 513]
# print(dirs)
prefix_ana = prefix_read + '/record_trajectories/'
for i in range(7):
    # csv_filename = prefix_read+dirs[i]
    csv_filename = prefix_ana+'trans_velocity' + '_index_' + str(
        list_simu_index_head
        [0]) + '.csv'
    df = pd.read_csv(csv_filename)
    print(df.columns)
    df_sub = df.iloc[240*i:240*(i+1)]
    print(df_sub.head(5))
    # list_simu_index = df['simu_index'].values
    list_u = -df_sub['U_eq'].values
    list_rho = df_sub['rho_trap_relative'].values
    list_frame_last = df_sub['saturate_value'].values
    # list_v = df['averaged_velocity'].values
    list_v = np.log10(df_sub['averaged_velocity'].values)
    list_isinf = np.isinf(list_v)
    list_v[list_isinf] = list_v.max()  # let -inf not be the min
    list_v[:] -= list_v.min()  # set meaningful min as baseline
    list_v[list_isinf] = 0  # zeroify the -inf again, let -inf the min
    normal_z = list_v/list_v.max()
    normal_zf = list_frame_last/list_frame_last.max()
    # sz = list_simu_index.shape()
    # num = int(sz[0]/10)
    png_filename_v = prefix_ana+'diagram_trans_velocity_index_' + str(
        list_simu_index_head
        [0]) + '_'+str(i) + '.png'
    draw(list_rho, list_u, normal_z, png_filename_v)
    png_filename_f = prefix_ana+'diagram_saturate_frame_index_' + str(
        list_simu_index_head
        [0]) + '_'+str(i) + '.png'
    draw(list_rho, list_u, normal_zf, png_filename_f)

"""import workflow_analysis as aw
atp = aw.archimedean_tilings_polygon_dye()
for i in range(11):
    atp.workflow_type_n_part_points_type_n_polygon(i+1)
#atp.workflow_type_n(62)
#atp.workflow_type_n(72)
import data_analysis_cycle as dac
import gsd.hoomd as hd
import symmetry_transformation_v4_3.analysis_controller as ac
index1 = 2243
prefix_write="/home/tplab/record_cairo/"#/hoomd-examples_0 

while index1<2253:
    input_file_gsd = prefix_write+'trajectory_auto'+str(int(index1))+'.gsd'
    gs = ac.get_a_gsd_from_setup()
    gs.get_gsd_data_from_filename(input_file_gsd)
    last_frame = gs.gsd_data[-1]
    gf = ac.get_data_from_a_gsd_frame(last_frame)
    gf.get_bonds_png_from_a_gsd_frame(str(index1)+'bond.png')
    #dac.save_from_gsd(index1,final_cut=True,bond_plot=True)
    index1+=1"""

"""csvs = ['/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv',
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv',
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv',
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv']
parts = [True,False,True,False]
types = [3,3,8,8]
import getDataAndDiagramCsv as dac
for i in range(4):
    dac.get_diagram_from_csv_type_n(csvs[i],types[i],parts[i])"""


tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)
