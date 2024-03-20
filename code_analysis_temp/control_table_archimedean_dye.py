# import threading

import symmetry_transformation_v4_3.list_code_analysis as lca
import time
import computeTime as ct
import symmetry_transformation_v4_3.analysis_controller as ac
import workflow_analysis as wa
import workflow_analysis_achimedean as waa
import numpy as np
import pandas as pd
import data_decorate as dd
import symmetry_transformation_v4_3.simulation_controller as sc
tpn = sc.simulation_controller_type_n_part_traps()
list_lcr0 = tpn.get_type_n_lcr0()
tm1 = time.localtime(time.time())

at = waa.archimedean_tilings()
# at.generate_type9_superlattice()
# print(at.position)
ad = waa.archimedean_tilings_polygon_dye()
ad.workflow_type_n(3)
# workflow_type_n_complement(62)
# ad.workflow_type_n_complement(72)
ad.workflow_type_n_complement(1)
# sdl = waa.show_dual_lattice()
# sdl.show_dual_type_n_part()
# sdl.show_dual_type_n_part_special(62)
# sdl.show_dual_type_n_part_special(72)
# sdl.show_dual_type_n_part_special(92)
tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)
