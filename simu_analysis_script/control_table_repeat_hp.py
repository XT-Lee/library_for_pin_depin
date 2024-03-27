# import threading
import symmetry_transformation_v4_3.list_code_simulation as lcs
import time
import computeTime as ct
tm1 = time.localtime(time.time())
prefix_read = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430'
dir_h = '/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'
dir_hp = '/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
dir_kg = '/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv'
dir_kgp = '/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv'
# gsd_reocrd_location: example_1,1,1,1; dir_hp moved to exple_1 and remain only seed1-9
dirs = [dir_h, dir_hp, dir_kg, dir_kgp]
cnks = [3, 3, 8, 8]
# print(dirs)


csv_filename = prefix_read+dirs[1]
lcs.pin_hex_to_hp_fill_workflow()

"""import symmetry_transformation_v4_3.simulation_controller as sc
prefix_write = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/'
index1 = 6013
output_file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_'+str(int(index1))+'.csv'
schp = sc.simulation_controller_honeycomb_part_traps() 
schp.generate_initial_state_hexagonal_particle_honeycomb_part_gaus_eq_harmo_trap_scan_csv(output_file_csv)"""


tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)
