# import threading
import function_plot.example_plot as fp
import getDataAndDiagramCsv as gdc
import time
import computeTime as ct
tm1 = time.localtime(time.time())

fpm = fp.functions_plot_module()
# x, y1 = fpm.generate_dipole(724)
x, y2 = fpm.generate_dipole_coarse(1500)
x, y3 = fpm.generate_yukawa()
fpm.plot_function2(x, y2, y3)

tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)
