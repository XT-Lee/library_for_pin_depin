# import threading

import points_analysis_2D_freud as pa
from PIL import Image
import pandas as pd
import image_analysis.particle_tracking as pt
import image_analysis.get_a_from_image as gi
import numpy as np
import time
import computeTime as ct
tm1 = time.localtime(time.time())

# at.generate_type9_superlattice()
# print(at.position)
image_filename = 'DefaultVideo_3-03014.jpg'  # <edit>
if False:
    gi.get_a_from_image(image_filename, save_data=True)
csv_filename = 'feature_single_frame_3-03014.csv'  # <edit>
trap_filename = 'kagome58.txt'  # #<edit>
traps = np.loadtxt(trap_filename)
traps = traps[:, :2]
df = pd.read_csv(csv_filename)
xy = df[['x', 'y']].values


# input points to get rectangle particle arrays.
rcp = gi.show_points_select()
xy = rcp.tune_pix2um(xy)
# rotate
sfe = pt.save_points_from_exp()
sfe.set_worksapce('', '')
sfe.path_to_results = ''

theta_ratate = 8.2  # <edit>
traps, points_filename = sfe.get_trap_positions(
    None, traps, [-29.5, -38.44], 0.75,  theta_ratate)  # <edit>
# rcp.show_points(xy, image_filename, traps=traps)

# horizontalization
traps, points_filename = sfe.get_trap_positions(None, traps, rotate_adjust=-theta_ratate)
xy, points_filename = sfe.get_point_positions(None, xy, rotate_adjust=-theta_ratate)
region_to_show_um = np.array([[-30, 15], [-25, 20]])  # <edit>
rcp.show_points_finetune(xy, image_filename, traps=traps, restrict=region_to_show_um)
# bond_length=[2,6.5]
# bpm.plot_scale_bar(-2.5, -17)
#
"""
regenerate image through particle decorate, background 0.2.
"""
# direct_cut_image
region_to_show_pix = rcp.tune_um2pix(region_to_show_um, False)

img = Image.open(image_filename)
r_img = img.rotate(-theta_ratate)
rx = region_to_show_pix.flatten()
r_img = r_img.crop((rx[0], rx[3], rx[1], rx[2]))  # reshape(1, -1)
# r_img.show()
bi = pa.bond_plot_module_for_image(r_img)
bi.restrict_axis_property_relative(hide_axis=True)  #
bi.plot_scale_bar(400, 400, 5)
bi.save_figure(image_filename+'_crop_bar.jpg')
# r_img.save(image_filename+'_crop.jpg')
tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)
