import time
tm1=time.localtime(time.time())

import symmetry_transformation_v4_3.list_code_analysis as lca
agf = lca.analyze_a_series_of_gsd_file_dynamic()
agf.get_transformation_velocity_scan_csv()#get_cnks_vs_t_from_csv()
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


tm2=time.localtime(time.time())
#calculate the time cost
import computeTime as ct
ct.getTimeCost(tm1,tm2)