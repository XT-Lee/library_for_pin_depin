# import threading
import workflow_analysis_achimedean as waa
import time
import computeTime as ct
tm1 = time.localtime(time.time())

type_n = [7,9]#[43, 53, 63, 73, 93]
atp = waa.archimedean_tilings_polygon_dye()
#sdl = waa.show_dual_lattice()
for i in range(2):
    #if i == 2:
    atp.workflow_type_n_complement(type_n[i],xylim=3)#atp.workflow_type_n(type_n[i])
    #sdl.show_dual_type_n_part_special(type_n[i])  # workflow_type_n(type_n[i])
tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)
