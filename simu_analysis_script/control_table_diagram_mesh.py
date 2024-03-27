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
csv_filenames = ['diagram_pin_hex_to_type_3', 'diagram_pin_hex_to_type_3_part',
                 'diagram_pin_hex_to_type_8', 'diagram_pin_hex_to_type_8_part']  # <edit>
cn_k = [3, 3, 4, 4]
rows = [8, 8, 9, 9]
for i in range(4):
    csv_filename = csv_filenames[i]+'.csv'
    df = pd.read_csv(csv_filename)  # cn3,U_eq,rho_trap_relative
    rho_trap = df['rho_trap_relative'].values
    u_trap = df['U_eq'].values
    cn_value = df['cn'+str(cn_k[i])].values

    n_points = len(cn_value)

    bm = gdd.scatter_plot_module()

    cmap_name = 'plasma'  # 'autumn_r'#'autumn'#newcmp#'binary'#
    transparency = 1.0  # 0.3

    # make data
    X = rho_trap.reshape(rows[i], 30)
    Y = u_trap.reshape(rows[i], 30)
    # X, Y = np.meshgrid(rho_trap, u_trap)
    Z = cn_value.reshape(rows[i], 30)

    sc = bm.ax.pcolormesh(X, Y, Z, cmap=cmap_name, zorder=-1, alpha=transparency)  # ,zorder=1
    # bm.fig.colorbar(sc, ax=bm.ax)
    bm.ax.tick_params(axis='both', which='both', direction='in')
    bm.ax.set_aspect('auto', 'box')  #
    bm.fig.set_size_inches(6, 6)
    """
        bm.ax.set_yscale('log', base=2)
        bm.ax.set_xscale('log', base=2)
        """  # Set the ticks and labels to represent the original log2 values

    bm.ax.set_xlabel('$\\rho_t/\\rho_p$')
    bm.ax.set_ylabel('$U_{trap} / k_BT$')
    png_filename = csv_filenames[i]+'_mesh_n.pdf'
    bm.save_figure(png_filename)
