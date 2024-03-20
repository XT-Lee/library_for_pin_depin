# import threading

import points_analysis_2D as pa
import matplotlib.pyplot as plt
import pandas as pd
import particle_tracking as pt
import get_a_from_image as gi
import numpy as np

import time
import computeTime as ct

import workflow_analysis_achimedean as waa

import symmetry_transformation_v4_3.simulation_controller as sc
tpn = sc.simulation_controller_type_n_part_traps()
list_lcr0 = tpn.get_type_n_lcr0()
tm1 = time.localtime(time.time())

at = waa.archimedean_tilings()
# at.generate_type9_superlattice()
# print(at.position)
image_filename = 'DefaultVideo_3-00888.jpg'  # 'DefaultVideo_3-03014.jpg'
if False:
    gi.get_a_from_image(image_filename, save_data=True)  #
csv_filename = 'feature_single_frame_3-00888.csv'
trap_filename = 'honeycomb45.txt'  # kagome58.txt
traps = np.loadtxt(trap_filename)
traps = traps[:, :2]
# traps[:, 1] = -traps[:, 1]
theta = 4.6  # rotate image data to match traps and readers.
df = pd.read_csv(csv_filename)
xy = df[['x', 'y']].values


# input points to get rectangle particle arrays.
rcp = gi.show_points_select()
xy = rcp.tune_points(xy)
# rotate
sfe = pt.save_points_from_exp()
sfe.set_worksapce('', '')
sfe.path_to_results = ''
# [36, 43], 0.75,
xy, points_filename = sfe.get_point_positions(None, xy, [31.5, 23], 1,  rotate_adjust=theta)
rcp.show_points(xy, image_filename, traps=traps)


if False:  # check points
    fig, ax = plt.subplots()
    ax.scatter(points[:0], points[:1], c='k')
    ax.scatter()
    plt.show()
    plt.close()

tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)
