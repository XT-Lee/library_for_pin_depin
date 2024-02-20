#import threading
import time
import computeTime as ct
tm1=time.localtime(time.time())

"""import symmetry_transformation_v4_3.simulation_controller as sc
prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_part_pin/'
index1 = 6013
output_file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_'+str(int(index1))+'.csv'
schp = sc.simulation_controller_honeycomb_part_traps() 
schp.generate_initial_state_hexagonal_particle_honeycomb_part_gaus_eq_harmo_trap_scan_csv(output_file_csv)"""

col = ['t']
n_params=8
for i in range(n_params-1):
    col +=  ['cn'+str(i+2)]
print(col)
tm2=time.localtime(time.time())
#calculate the time cost
ct.getTimeCost(tm1,tm2)

