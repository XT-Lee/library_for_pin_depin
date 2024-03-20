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
    gdc.get_diagram_from_csv_type_n(csvs[i], type_n[i], part[i])
    # gdc.get_diagram_binary_from_csv_type_n(type_n[i], part[i], save=True)
tm2 = time.localtime(time.time())
# calculate the time cost
ct.getTimeCost(tm1, tm2)
