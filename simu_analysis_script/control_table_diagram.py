# import threading
import getDataAndDiagramCsv as gdc
import time
import computeTime as ct
tm1 = time.localtime(time.time())
# gdc.get_diagram_from_csv_type3()
# gdc.get_diagram_from_csv_type3_part()
# gdc.get_diagram_from_csv_type8()
# gdc.get_diagram_from_csv_type8_part()
csvs = ['/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv']
type_n = [3, 3, 8, 8]
part = [False, True, False, True]
for i in range(4):
    # gdc.get_diagram_from_csv_type_n(csvs[i], type_n[i], part[i])
    gdc.get_diagram_binary_from_csv_type_n(type_n[i], part[i], save=True)
tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)

"""
rcut = 1.0
cmap_name = 'Reds'  # 'autumn_r'#'autumn'#newcmp#'binary'#
transparency = 0.5  # 0.3

# set traps
max = np.max(traps)
min = np.min(traps)
length = (max - min)
steps = length/(rcut/10.0)
# plt.style.use('_mpl-gallery-nogrid')

# make data
X, Y = np.meshgrid(np.linspace(min, max, steps.astype(int)),
                    np.linspace(min, max, steps.astype(int)))
HarmonicK = 100
# origin = np.zeros((1,2))
sz = np.shape(traps)
i = 0
Z = ((0.50*HarmonicK*rcut*rcut-0.50*HarmonicK*((X-traps[i, 0])**2 + (Y-traps[i, 1])**2))
        * (((X-traps[i, 0])**2 + (Y-traps[i, 1])**2) < rcut*rcut))
i = i+1
while i < sz[0]:  # sz[0]
    Zi = (0.50 * HarmonicK * rcut * rcut - 0.50 * HarmonicK *
            ((X - traps[i, 0]) ** 2 + (Y - traps[i, 1]) ** 2)) * (((X - traps[i, 0]) ** 2 +
                                                                (Y - traps[i, 1]) ** 2) < rcut * rcut)
    Z = Z + Zi
    i = i+1

self.ax.pcolormesh(X, Y, Z, cmap=cmap_name, zorder=-1, alpha=transparency)  # ,zorder=1
"""
