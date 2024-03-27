
import getDataAndDiagram as gdd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.spatial import distance
import pandas as pd
import workflow_analysis_achimedean as waa
ad = waa.archimedean_tilings_polygon_dye()

R"""
introduction:
    since origin2021pro can not edit color for each scatter point, 
    matplotlib is chosen as a replacement.
    marker can only be defined a string, not an array
"""
# get data
csv_filename = 'wca_yukawa_depin_from_type3_7933_ana.csv'  # <edit>
df = pd.read_csv(csv_filename)
seeds = df['seed'].values
u_pp = df['u_yukawa_r1'].values
u_trap = df['U_eq'].values
state_id = df['state(0hex1other2honey)'].values

# select single seed
seedi = 0  # for seedi in range(10):
list_id = seeds[:] == seedi
u_ppi = u_pp[list_id]
u_trapi = u_trap[list_id]
state_idi = state_id[list_id]
n_points = len(state_idi)

# colors = ['b', 'g', 'r']
# list_colors = np.zeros((n_points,), dtype=str)
# colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]])/255
colors = [ad.color3, ad.color12, ad.color6]
list_colors = np.zeros((n_points, 3))
markers = ['^', 'o', 'H']
list_markers = np.zeros((n_points,), dtype=str)
for i in range(n_points):
    list_colors[i] = colors[state_idi[i]]
    # list_markers[i] = markers[state_id[i]]

bm = gdd.scatter_plot_module()
# m.scatter_defined_colors([0, 5, 10], [5, 5, 5], np.array(
#    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), '^')
u_pp_rs, u_pp_rs_ticks, u_pp_rs_ticks_label = bm.rescale_data_log2(u_ppi)
u_trap_rs, u_trap_rs_ticks, u_trap_rs_ticks_label = bm.rescale_data_log2(u_trapi)
for i in range(len(markers)):
    # marker can only be defined a string, not an array
    list_id = state_idi[:] == i
    bm.scatter_defined_colors(
        u_pp_rs[list_id],
        u_trap_rs[list_id],
        list_colors[list_id],
        markers[i])
    """bm.scatter_defined_colors(
        u_pp[list_id],
        u_trap[list_id],
        list_colors[list_id],
        markers[i])"""
bm.ax.tick_params(axis='both', which='both', direction='in')
bm.ax.set_aspect('equal', 'box')
"""bm.ax.set_yscale('log', base=2)
bm.ax.set_xscale('log', base=2)
"""  # Set the ticks and labels to represent the original log2 values

# xtick_labels = [f'$2^{i}$' for i in u_pp_rs_ticks]
xtick_labels = [f'{i}' for i in u_pp_rs_ticks_label]
bm.ax.set_xticks(u_pp_rs_ticks)
bm.ax.set_xticklabels(xtick_labels)
# ytick_labels = [f'$2^{i}$' for i in u_trap_rs_ticks]
ytick_labels = [f'{i}' for i in u_trap_rs_ticks_label]
bm.ax.set_yticks(u_trap_rs_ticks)
bm.ax.set_yticklabels(ytick_labels)
bm.ax.set_xlabel('$U_{pp} / k_BT$')
bm.ax.set_ylabel('$U_{trap} / k_BT$')
png_filename = 'wca_yukawa_depin_from_type3_7933_ana_'+str(seedi)+'.pdf'
bm.save_figure(png_filename)
