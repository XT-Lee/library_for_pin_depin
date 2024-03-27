# import threading

import time
import computeTime as ct
import workflow_analysis as wa
import workflow_analysis_achimedean as waa
import numpy as np
import symmetry_transformation_v4_3.simulation_controller as sc
tpn = sc.simulation_controller_type_n_part_traps()
list_lcr0 = tpn.get_type_n_lcr0()
tm1 = time.localtime(time.time())

at = waa.archimedean_tilings()
# at.generate_type9_superlattice()
# print(at.position)
ad = waa.archimedean_tilings_polygon_dye()
# traps plot cartoon:
"""ad.workflow_type_n_pure_background(1, 2, color_particle='r')
ad.workflow_type_n_pure_background(2, 2, color_particle='r')
ad.workflow_type_n_pure_background(3, 3, color_particle='r')
ad.workflow_type_n_pure_background(8, 3, color_particle='r')
ad.workflow_type_n_pure_background(3, 3, part=True, color_particle='r')
ad.workflow_type_n_pure_background(8, 3, part=True, color_particle='r')
"""
# particles fianl state plot cartoon:
ad.workflow_type_n_pure_background(1, 2)
ad.workflow_type_n_pure_background(2, 2)
ad.workflow_type_n_pure_background(3, 3)
ad.workflow_type_n_pure_background(8, 3)
# workflow_type_n_complement(62)
# ad.workflow_type_n_complement(72)
# ad.workflow_type_n_complement(1)
# sdl = waa.show_dual_lattice()
# sdl.show_dual_type_n_part()
# sdl.show_dual_type_n_part_special(62)
# sdl.show_dual_type_n_part_special(72)
# sdl.show_dual_type_n_part_special(92)
tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)
