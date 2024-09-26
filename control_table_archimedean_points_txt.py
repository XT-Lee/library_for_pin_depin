# import threading
import workflow_analysis_achimedean as waa

atp = waa.show_dual_lattice()
waa.archimedean_tilings()
type_n = 12
# for type_n in [4, 5, 6, 9]:  # range(11):
# atp.show_dual_type_n(type_n*10+3, xylim=5)
atp.show_dual_type_n(type_n, xylim=5)  # workflow_type_n(type_n[i]),*10+3
