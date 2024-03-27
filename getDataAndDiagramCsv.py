import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
R"""
#NEW plot standard:
#https://matplotlib.org/3.6.2/tutorials/introductory/quick_start.html#sphx-glr-tutorials-introductory-quick-start-py
    import matplotlib.pyplot as plt
    import numpy as np

    points = np.random.random((100,2))#random.random () ()
    
    fig,ax = plt.subplots()
    ax.plot(points[:,0],points[:,1])
    ax.scatter(points[:,0],points[:,1],c=points[:,1])
    ax.set_xlabel('x label')  # Add an x-label to the axes.
    ax.set_ylabel('y label')  # Add a y-label to the axes.
    ax.set_title("Simple Plot")  # Add a title to the axes
    plt.show()

"""


class csv_data_processor:
    def __init__(self, csv_filename):
        self.record = pd.read_csv(csv_filename)
        # get list_lcr0 for archimedean type_1,2,3...11
        import symmetry_transformation_v4_3.simulation_controller as sc
        tpn = sc.simulation_controller_type_n_part_traps()
        self.list_lcr0 = tpn.get_type_n_lcr0()
        del tpn

    def select_single_seed(self, seed):
        self.sub_record = self.record[self.record['seed'] == seed]
        self.sub_record.sort_values(by='simu_index', ascending=True, inplace=True)

    def get_relative_rho(self, type_n):
        R"""
        rho_trap_relative = ( lcr0_trap / lcr )^2
        """
        unit_area0_trap = self.list_lcr0[type_n-1]*self.list_lcr0[type_n-1]
        unit_area_trap = np.square(self.record['lcr'].values)
        self.record['rho_trap_relative'] = unit_area0_trap / unit_area_trap

    def save_csv(self, output_file_csv):
        # to strip 'unnamed:0' column which contains [1,2,3...]
        list_col = self.record.columns.values[1:]
        pd.DataFrame.to_csv(self.record[list_col], output_file_csv)

    def save_sub_csv(self, output_file_csv):
        # to strip 'unnamed:0' column which contains [1,2,3...]
        list_col = self.sub_record.columns.values[1:]
        pd.DataFrame.to_csv(self.sub_record[list_col], output_file_csv)

    def get_data_diagram(self, cn_k=4):
        self.rho = self.sub_record['rho_trap_relative'].values
        self.u = self.sub_record['U_eq'].values
        self.cnk = self.sub_record['cn'+str(cn_k)].values


class diagram_plot_module:
    def __init__(self, fig=None, ax=None):
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = fig
            self.ax = ax

    def set_parameters(self):
        self.rhos = None
        self.us = None
        self.cnks = None

    def set_figure_elements(self, xlabel_name='$\\rho_t/\\rho_p$', ylabel_name='$U_t/k_{B}T$'):
        self.ax.set_xlabel(xlabel_name)
        self.ax.set_ylabel(ylabel_name)
        self.ax.tick_params(axis='both', which='both', direction='in')

    def draw_diagram_scatter_oop(self):  # ,data,title_name,xlabel_name,ylabel_name,prefix,postfix
        sc = self.ax.scatter(self.rhos, self.us, c=self.cnks, cmap='plasma')
        self.fig.colorbar(sc, ax=self.ax)
        """ax.set_title(title_name)
        ax.set_xlabel(xlabel_name)
        ax.set_ylabel(ylabel_name)
        #ax.set_yscale('log')
        #ax.set_xlim(xlim)
        ax.set_ylim([0,300])
        from matplotlib import cm
        from matplotlib.colors import Normalize as nm
        vmin = np.min(data[:,2])
        vmax = np.max(data[:,2])
        #print(vmin,vmax)
        nmm = nm(vmin=vmin,vmax=vmax)
        #https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar
        fig.colorbar(cm.ScalarMappable(norm=nmm),ax=ax)#.autoscale(data[:,2])
        png_filename=prefix+title_name+postfix
        fig.savefig(png_filename)
        plt.close()"""

    def save_figure(self, png_filename):
        R"""
        parameter:
            png_filename: "prefix/bond_plot_index1513.png"

        if latex is used in matplolib,
        'pip install latex' is necessary for plt.savefig()
        """
        self.fig.savefig(png_filename, bbox_inches='tight')  # plt.savefig(png_filename)
        plt.close('all')  # self.fig,plt.close() # closes the current active figure


def get_diagram_binary_from_csv_type_n(type_n, part, csv_filename=None, save=False):
    """
    output_file_csvs = ["/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv",#0-6
            "/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv",#0-9
            "/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv",#0-9
            "/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv"]#0-9
    seed_lims = [6,9,9,9]
    import symmetry_transformation_v4_3.list_code_analysis as lca
    asg = lca.analyze_a_series_of_gsd_file()
    for i in range(4):
        if i<2:
            add_type_3 = True
        else:
            add_type_3 = False
        asg.get_cnks_from_csv_files_type_n_part(output_file_csvs[i],seed_lims[i],add_type_3)
    """
    """
    csvs = ['/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv']
    type_n = [3, 3, 8, 8]
    part = [False, True, False, True]
    """
    if csv_filename is None:
        if part:
            csv_filename = 'diagram_pin_hex_to_type_'+str(type_n)+'_part'+'.csv'
        else:
            csv_filename = 'diagram_pin_hex_to_type_'+str(type_n)+'.csv'
    cdp = csv_data_processor(csv_filename)
    cdp.select_single_seed(0)
    import workflow_analysis as wa
    at = wa.archimedean_tilings()
    coord_num_k = at.get_coordination_number_k_for_type_n(type_n)
    column_name_to_merge = 'cn'+str(coord_num_k)
    # col_select = [column_name_to_merge,'U_eq','rho_trap_relative']
    cdp.get_data_diagram(coord_num_k)
    fig, ax = plt.subplots()
    dpm = diagram_plot_module(fig, ax)
    dpm.set_parameters()
    dpm.rhos = cdp.rho
    dpm.us = cdp.u
    list_trans = cdp.cnk > 0.80
    list_hex = cdp.cnk < 0.20
    dpm.cnks = np.array(cdp.cnk > 0.80).astype(float)  # .astype(int)
    dpm.cnks[:] = 0.5
    dpm.cnks[list_trans] = 1
    dpm.cnks[list_hex] = 0
    dpm.draw_diagram_scatter_oop()
    dpm.set_figure_elements()

    if part:
        unit_name = 'diagram_bi_pin_hex_to_type_'+str(type_n)+'_part'
        dpm.save_figure(unit_name+'.png')
    else:
        unit_name = 'diagram_bi_pin_hex_to_type_'+str(type_n)
        dpm.save_figure(unit_name+'.png')
    if save:
        df = pd.DataFrame({'rho_trap_relative': dpm.rhos, 'U_eq': dpm.us,
                           column_name_to_merge: dpm.cnks})
        df.to_csv(unit_name+'.csv', index=False)


def get_diagram_from_csv_type_n(csv_filename, type_n, part):
    """
    output_file_csvs = ["/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv",#0-6
            "/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv",#0-9
            "/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv",#0-9
            "/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv"]#0-9
    seed_lims = [6,9,9,9]
    import symmetry_transformation_v4_3.list_code_analysis as lca
    asg = lca.analyze_a_series_of_gsd_file()
    for i in range(4):
        if i<2:
            add_type_3 = True
        else:
            add_type_3 = False
        asg.get_cnks_from_csv_files_type_n_part(output_file_csvs[i],seed_lims[i],add_type_3)
    """
    """
    csvs = ['/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv']
    type_n = [3, 3, 8, 8]
    part = [False, True, False, True]
    """

    # csv_filename ='/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
    cdp = csv_data_processor(csv_filename)
    # cdp.get_relative_rho(3)
    # cdp.save_csv(csv_filename)

    cdp.select_single_seed(0)
    import workflow_analysis as wa
    at = wa.archimedean_tilings()
    coord_num_k = at.get_coordination_number_k_for_type_n(type_n)
    column_name_to_merge = 'cn'+str(coord_num_k)
    cnk_averaged, cnk_std = merge_cnk_std_by_seed(cdp.record, column_name_to_merge, 10)
    cdp.sub_record[column_name_to_merge] = cnk_averaged
    cdp.sub_record[column_name_to_merge+'std'] = cnk_std
    cdp.sub_record['U_eq'] = -cdp.sub_record['U_eq'].values
    if part:
        cdp.save_sub_csv('diagram_pin_hex_to_type_'+str(type_n)+'_part'+'.csv')
    else:
        cdp.save_sub_csv('diagram_pin_hex_to_type_'+str(type_n)+'.csv')
    cdp.get_data_diagram(coord_num_k)
    fig, ax = plt.subplots()
    dpm = diagram_plot_module(fig, ax)
    dpm.set_parameters()
    dpm.rhos = cdp.rho
    dpm.us = cdp.u
    dpm.cnks = cdp.cnk
    dpm.draw_diagram_scatter_oop()
    dpm.set_figure_elements()
    if part:
        dpm.save_figure('diagram_pin_hex_to_type_'+str(type_n)+'_part.png')
    else:
        dpm.save_figure('diagram_pin_hex_to_type_'+str(type_n)+'.png')


def get_diagram_from_csv_type8():
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv'
    csv_filename = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv'
    cdp = csv_data_processor(csv_filename)
    # cdp.get_relative_rho(8)
    # cdp.save_csv(csv_filename)

    cdp.select_single_seed(0)
    import workflow_analysis as wa
    at = wa.archimedean_tilings()
    coord_num_k = at.get_coordination_number_k_for_type_n(8)
    column_name_to_merge = 'cn'+str(coord_num_k)
    cnk_averaged = merge_cnk_by_seed(cdp.record, column_name_to_merge, 10)
    cdp.sub_record['cn4'] = cnk_averaged
    cdp.save_sub_csv('diagram_pin_hex_to_type_8.csv')
    cdp.get_data_diagram(coord_num_k)
    fig, ax = plt.subplots()
    dpm = diagram_plot_module(fig, ax)
    dpm.set_parameters()
    dpm.rhos = cdp.rho
    dpm.us = -cdp.u
    dpm.cnks = cdp.cnk
    dpm.draw_diagram_scatter_oop()
    dpm.set_figure_elements()
    dpm.save_figure('diagram_pin_hex_to_type_8.png')


def get_diagram_from_csv_type8_part():
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv'
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv'

    csv_filename = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv'
    cdp = csv_data_processor(csv_filename)
    # cdp.get_relative_rho(8)
    # cdp.save_csv(csv_filename)

    cdp.select_single_seed(0)
    import workflow_analysis as wa
    at = wa.archimedean_tilings()
    coord_num_k = at.get_coordination_number_k_for_type_n(8)
    column_name_to_merge = 'cn'+str(coord_num_k)
    cnk_averaged = merge_cnk_by_seed(cdp.record, column_name_to_merge, 10)
    cdp.sub_record['cn4'] = cnk_averaged
    cdp.save_sub_csv('diagram_pin_hex_to_type_8_part.csv')
    cdp.get_data_diagram(4)
    fig, ax = plt.subplots()
    dpm = diagram_plot_module(fig, ax)
    dpm.set_parameters()
    dpm.rhos = cdp.rho
    dpm.us = -cdp.u
    dpm.cnks = cdp.cnk
    dpm.draw_diagram_scatter_oop()
    dpm.set_figure_elements()
    dpm.save_figure('diagram_pin_hex_to_type_8_part.png')


def get_diagram_from_csv_type3():
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv'
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv'
    csv_filename = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'

    cdp = csv_data_processor(csv_filename)
    # cdp.get_relative_rho(3)
    # cdp.save_csv(csv_filename)

    cdp.select_single_seed(0)
    import workflow_analysis as wa
    at = wa.archimedean_tilings()
    coord_num_k = at.get_coordination_number_k_for_type_n(3)
    column_name_to_merge = 'cn'+str(coord_num_k)
    cnk_averaged = merge_cnk_by_seed(cdp.record, column_name_to_merge, 10)
    cdp.sub_record[column_name_to_merge] = cnk_averaged
    cdp.save_sub_csv('diagram_pin_hex_to_type_3.csv')
    cdp.get_data_diagram(3)
    fig, ax = plt.subplots()
    dpm = diagram_plot_module(fig, ax)
    dpm.set_parameters()
    dpm.rhos = cdp.rho
    dpm.us = -cdp.u
    dpm.cnks = cdp.cnk
    dpm.draw_diagram_scatter_oop()
    dpm.set_figure_elements()
    dpm.save_figure('diagram_pin_hex_to_type_3.png')


def get_diagram_from_csv_type3_part():
    """
    import getDataAndDiagramCsv as gdc
    #gdc.get_diagram_from_csv_type8()
    #gdc.get_diagram_from_csv_type8_part()
    #gdc.get_diagram_from_csv_type3()
    gdc.get_diagram_from_csv_type3_part()
    """
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv'
    '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv'

    csv_filename = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
    cdp = csv_data_processor(csv_filename)
    # cdp.get_relative_rho(3)
    # cdp.save_csv(csv_filename)

    cdp.select_single_seed(0)
    import workflow_analysis as wa
    at = wa.archimedean_tilings()
    coord_num_k = at.get_coordination_number_k_for_type_n(3)
    column_name_to_merge = 'cn'+str(coord_num_k)
    cnk_averaged, cnk_std = merge_cnk_std_by_seed(cdp.record, column_name_to_merge, 10)
    cdp.sub_record[column_name_to_merge] = cnk_averaged
    cdp.sub_record[column_name_to_merge+'std'] = cnk_std
    cdp.sub_record['U_eq'] = cdp.sub_record['U_eq'].values
    cdp.save_sub_csv('diagram_pin_hex_to_type_3_part.csv')
    cdp.get_data_diagram(coord_num_k)
    fig, ax = plt.subplots()
    dpm = diagram_plot_module(fig, ax)
    dpm.set_parameters()
    dpm.rhos = cdp.rho
    dpm.us = -cdp.u
    dpm.cnks = cdp.cnk
    dpm.draw_diagram_scatter_oop()
    dpm.set_figure_elements()
    dpm.save_figure('diagram_pin_hex_to_type_3_part.png')


def merge_cnk_by_seed(record, column_name_to_merge, n_seeds):
    for i in range(n_seeds):
        sub_record_seedi = record[record['seed'] == i]
        sub_record_seedi.sort_values(by=['simu_index'], ascending=True, inplace=True)
        cnk_seedi = sub_record_seedi[column_name_to_merge].values
        if i == 0:
            cnk_sum = np.array(cnk_seedi)  # avoid copying only pointer
        else:
            cnk_sum += cnk_seedi
    cnk_averaged = cnk_sum/n_seeds
    return cnk_averaged


def merge_cnk_std_by_seed(record, column_name_to_merge, n_seeds):
    for i in range(n_seeds):
        sub_record_seedi = record[record['seed'] == i]
        sub_record_seedi.sort_values(by=['simu_index'], ascending=True, inplace=True)
        cnk_seedi = sub_record_seedi[column_name_to_merge].values
        if i == 0:
            cnk_sum = np.array(cnk_seedi)  # avoid copying only pointer
            n_row = cnk_sum.shape[0]
            list_cnk = np.zeros((n_row, 10))
            list_cnk[:, 0] = cnk_sum
        else:
            list_cnk[:, i] = cnk_seedi
    cnk_averaged = np.average(list_cnk, axis=1)  # multi columns into one column
    cnk_sted = np.std(list_cnk, axis=1)
    return cnk_averaged, cnk_sted
