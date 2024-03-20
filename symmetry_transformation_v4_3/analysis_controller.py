import points_analysis_2D as pa
import gsd.hoomd
import numpy as np


class get_a_gsd_from_setup:
    def __init__(self):
        pass

    def set_file_parameters(self, simu_index, seed, read=True):
        self.prefix_read = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/"
        self.prefix_write = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/"
        self.seed = seed
        self.simu_index = simu_index
        self.input_file_gsd = self.prefix_write+'trajectory_auto' + \
            str(int(self.simu_index))+'_'+str(int(self.seed))+'.gsd'
        # self.input_file_gsd = self.prefix_read+'particle_and_trap.gsd'
        # self.output_file_gsd = self.prefix_write+'trajectory_auto'+str(int(self.simu_index))+'_'+str(int(self.seed))+'.gsd'
        self.snap_period = 1000
        if read:
            self.gsd_data = gsd.hoomd.open(self.input_file_gsd)

    def set_file_parameters0(self, simu_index, seed):
        self.prefix_read = "/media/remote/32E2D4CCE2D49607/file_lxt/record_cairo/"  # hoomd-examples_0
        self.prefix_write = "/media/remote/32E2D4CCE2D49607/file_lxt/record_cairo/"  # /hoomd-examples_0
        self.seed = seed
        self.simu_index = simu_index
        self.input_file_gsd = self.prefix_write+'trajectory_auto' + \
            str(int(self.simu_index))+'_'+str(int(self.seed))+'.gsd'
        # self.input_file_gsd = self.prefix_read+'particle_and_trap.gsd'
        # self.output_file_gsd = self.prefix_write+'trajectory_auto'+str(int(self.simu_index))+'_'+str(int(self.seed))+'.gsd'
        self.snap_period = 1000
        self.gsd_data = gsd.hoomd.open(self.input_file_gsd)

    def get_gsd_data_from_file(self):
        self.gsd_data = gsd.hoomd.open(self.input_file_gsd)

    def get_gsd_data_from_filename(self, input_file_gsd):
        self.gsd_data = gsd.hoomd.open(input_file_gsd)


class get_gsds_from_mysql_or_csv:
    def __init__(self):
        pass

    def get_record_from_sql_by_lcr(self, lcr1=0.81, table_name='pin_hex_to_honeycomb_klt_2m_gauss'):
        R"""
        table_name = 'pin_hex_to_honeycomb_klt_2m_gauss'
        | simu_index | seed | lcr  | trap_gauss_epsilon | temperature |
        list_simu = [colums of table]
        """
        import opertateOnMysql as osql
        # import data_retriever as dr
        # se = dr.search_engine_for_simulation_database()

        # se.search_single_simu_by_lcr_k(table_name,0.81)
        lcr_step = 0.0001
        lcr_min = lcr1 - 0.5*lcr_step
        lcr_max = lcr1 + 0.5*lcr_step
        con = ' where lcr >'+str(lcr_min)+' and lcr <'+str(lcr_max)
        list_simu = osql.getDataFromMysql(table_name=table_name, search_condition=con)
        # simu_index,seed,lcr,k,kT
        return list_simu

    def get_gsds_from_mysql_record(self, record):
        R"""
        table_name = 'pin_hex_to_honeycomb_klt_2m_gauss'
        | simu_index | seed | lcr  | trap_gauss_epsilon | temperature |
        list_simu = [colums of table]
        """
        import symmetry_transformation_v4_3.simulation_core as sc
        n_simu = len(record)
        gsds = []
        for i in range(n_simu):
            sct = sc.simulation_core_traps(record[i][0], record[i][1])
            gsds.append(sct.output_file_gsd)  # record[i].append(sct.output_file_gsd)
            # print(record[i])
        return gsds

    def get_record_from_csv(self, csv_filename):
        R"""
        table_name = 'pin_hex_to_honeycomb_klt_2m_gauss'
        | simu_index | seed | lcr  | trap_gauss_epsilon | temperature |
        list_simu = [colums of table]
        """
        import symmetry_transformation_v4_3.simulation_core as sc
        import pandas as pd
        record = pd.read_csv(csv_filename)
        simu_index = record['simu_index'].values
        seed = record['seed'].values
        n_simu = len(seed)
        gsds = []
        for i in range(n_simu):
            sct = sc.simulation_core_traps(simu_index[i], seed[i])
            gsds.append(sct.output_file_gsd)  # record[i].append(sct.output_file_gsd)
            # print(record[i])
            del sct
        self.record = record
        return gsds

    def get_last_frame_from_gsd(self, gsd_filename):
        gsd_data = gsd.hoomd.open(gsd_filename)
        return gsd_data[-1]


class get_data_from_a_gsd_frame:
    def __init__(self, last_frame=None, points=None, traps=None):
        if not last_frame is None:
            self.last_frame = last_frame
            xy_particles_traps = last_frame.particles.position[:, :2]
            ids = np.array(last_frame.particles.typeid)
            list_p = ids == 0
            list_t = ids == 1
            self.points = xy_particles_traps[list_p]
            self.traps = xy_particles_traps[list_t]
        else:
            self.points = points
            self.traps = traps

    def get_cn_k_from_a_gsd_frame(self, tune_dis=2.4, k=None):
        points = self.points
        traps = self.traps
        obj_of_simu_index = pa.static_points_analysis_2d(points)  # ,hide_figure=False
        # tune_dis = 2.4#lattice_a*lcr?
        xmax = max(traps[:, 0]) - tune_dis
        ymax = max(traps[:, 1]) - tune_dis
        xmin = min(traps[:, 0]) + tune_dis
        ymin = min(traps[:, 1]) + tune_dis
        obj_of_simu_index.cut_edge_of_positions_by_xylimit(xmin, xmax, ymin, ymax)
        obj_of_simu_index.get_coordination_number_conditional(tune_dis)
        ccn = obj_of_simu_index.count_coordination_ratio
        # print(ccn[3])

        """import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.scatter(points[:,0],points[:,1],color='k')#
        ax.scatter(traps[:,0],traps[:,1],color='r',marker = 'x')#
        fence = np.array([[xmin,ymin],[xmin,ymax],[xmax,ymax],[xmax,ymin],[xmin,ymin]])
        ax.plot(fence[:,0],fence[:,1])
        #ax.scatter(dula[:,0],dula[:,1],facecolors='none',edgecolors='k')#,marker = 'x'
        ax.set_xlabel('x label')  # Add an x-label to the axes.
        ax.set_ylabel('y label')  # Add a y-label to the axes.
        ax.set_title("Simple Plot")  # Add a title to the axes
        ax.set_aspect('equal','box')
        plt.show()
        plt.close('all')"""
        if k is None:
            return ccn
        else:
            return ccn[k]

    def get_bonds_from_a_gsd_frame(self, tune_dis=2.4):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        p2d = pa.static_points_analysis_2d(self.points, hide_figure=False)

        p2d.get_first_minima_bond_length_distribution(
            lattice_constant=tune_dis, png_filename='bond_hist.png')
        # draw bonds selected
        bpm = pa.bond_plot_module(fig, ax)
        bpm.restrict_axis_property_relative('(sigma)')
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(
            p2d.bond_length, [2, p2d.bond_first_minima_left])

        # p2d.bond_length[:,:2].astype(int)
        bpm.plot_points_with_given_bonds(
            self.points, list_bond_index, bond_color='k', particle_color='k')
        bpm.plot_traps(self.traps)
        plt.show()
        """traps=self.traps
        #tune_dis = 2.4#lattice_a*lcr?
        xmax = max(traps[:,0]) - tune_dis
        ymax = max(traps[:,1]) - tune_dis
        xmin = min(traps[:,0]) + tune_dis
        ymin = min(traps[:,1]) + tune_dis
        p2d.cut_edge_of_positions_by_xylimit(xmin,xmax,ymin,ymax)"""

    def get_bonds_png_from_a_gsd_frame(self, png_filename, tune_dis=2.4):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        p2d = pa.static_points_analysis_2d(self.points)  # dis_edge_cut=,, hide_figure=False

        p2d.get_first_minima_bond_length_distribution(
            lattice_constant=tune_dis, png_filename='bond_hist.png')
        # draw bonds selected
        bpm = pa.bond_plot_module(fig, ax)
        bpm.restrict_axis_property_relative(hide_axis=True)  # '(sigma)'
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(
            p2d.bond_length, [1.2, p2d.bond_first_minima_left])

        # p2d.bond_length[:,:2].astype(int)
        bpm.draw_points_with_given_bonds(
            self.points, list_bond_index, bond_color='k', particle_color='k')
        bpm.plot_traps(self.traps)
        bpm.restrict_axis_limitation([-10, 10], [-10, 10])  # [-20,0],[-5,15]
        bpm.save_figure(png_filename)
        del bpm

    def get_given_bonds_png_from_a_gsd_frame(self, png_filename, bond_first_minima_left):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        p2d = pa.static_points_analysis_2d(self.points, hide_figure=False)  # dis_edge_cut=

        p2d.get_first_minima_bond_length_distribution(png_filename='bond_hist.png')
        # draw bonds selected
        bpm = pa.bond_plot_module(fig, ax)
        bpm.restrict_axis_property_relative(hide_axis=True)  # '(sigma)'
        bpm.restrict_axis_limitation([-10, 10], [-10, 10])
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(
            p2d.bond_length, [1.2, bond_first_minima_left])

        # p2d.bond_length[:,:2].astype(int),,particle_size=20
        bpm.plot_points_with_given_bonds(
            self.points, list_bond_index, bond_color='k', particle_color='k')
        bpm.plot_traps(self.traps)
        # bpm.restrict_axis_limitation([-10,10],[-10,10])
        bpm.save_figure(png_filename)
        del bpm


class get_data_from_a_trajectory:
    def __init__(self, simu_index, seed, gsd_filename=None, stable=False):  # txy=None,traps=None
        self.__set_file_parameters(simu_index, seed)
        if not gsd_filename is None:
            self.from_gsd_to_data(gsd_filename, stable)
        self.load_data(stable)

    def __set_file_parameters(self, simu_index, seed, work_space=None):
        self.simu_index = simu_index
        self.seed = seed
        if work_space is None:
            self.work_space = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/record_trajectories/'
        else:
            self.work_space = work_space
        self.folder_for_a_trajectory = str(int(self.simu_index))+'_'+str(int(self.seed))
        self.str_simu_index = self.folder_for_a_trajectory
        self.directory_to_trajectory_data = self.work_space+self.folder_for_a_trajectory+'/'
        # self.input_file_gsd = self.prefix_read+'particles.gsd'
        # self.output_file_gsd = self.prefix_write+'trajectory_auto'+str(int(self.simu_index))+'_'+str(int(self.seed))+'.gsd'

    def from_gsd_to_data(self, filename_gsd_seed, stable):
        import proceed_file as pf
        pfile = pf.proceed_file_shell()
        pfile.create_folder(self.work_space, self.folder_for_a_trajectory)
        # save data
        gsd_data = pf.proceed_gsd_file(filename_gsd_seed=filename_gsd_seed)
        gsd_data.get_trajectory_data_with_traps(self.directory_to_trajectory_data)
        if stable:
            gsd_data.get_trajectory_stable_data_with_traps(self.directory_to_trajectory_data)

    def load_data(self, stable=False):
        # load data
        filename_txyz = self.directory_to_trajectory_data+'txyz.npy'
        filename_txyz_stable = self.directory_to_trajectory_data+'txyz_stable.npy'
        filename_traps = self.directory_to_trajectory_data+'traps.txt'
        filename_box = self.directory_to_trajectory_data+'box.txt'
        self.txyz = np.load(filename_txyz)
        if stable:
            self.txyz_stable = np.load(filename_txyz_stable)
        self.traps = np.loadtxt(filename_traps)
        self.box = np.loadtxt(filename_box)

    def get_polygon(self, dis_cut=0):
        import proceed_file as pf
        pfile = pf.proceed_file_shell()

        prefix_write = pfile.create_folder(self.directory_to_trajectory_data, 'polygon_6')
        txyz = self.txyz
        nframes = np.shape(txyz)[0]
        # self.read_data(prefix_write)
        for frame in range(nframes):  # [8,104,1427]:#
            if frame > 141:
                p2d = pa.static_points_analysis_2d(txyz[frame])

                png_filename = None  # prefix_write+"ridge_length_hist"+str(int(frame))+".png"
                p2d.get_first_minima_ridge_length_distribution(
                    hist_cutoff=5, png_filename=png_filename)
                # png_filename = prefix_write+"bond_length_hist"+str(int(frame))+".png"
                # p2d.get_first_minima_bond_length_distribution(png_filename=png_filename)

                png_filename = prefix_write+"bond_vertices_patch"+str(int(frame))+".png"
                import matplotlib.pyplot as plt
                import workflow_analysis as wa
                at = wa.archimedean_tilings_polygon_dye()
                fig, ax = plt.subplots()
                bpm = pa.bond_plot_module(fig, ax)
                bpm.restrict_axis_property_relative(hide_axis=True)

                list_bond_index = bpm.get_bonds_with_conditional_ridge_length(
                    p2d.voronoi.ridge_length[:], p2d.voronoi.ridge_points, p2d.ridge_first_minima_left)
                # list_bond_index = bpm.get_bonds_with_conditional_bond_length(p2d.bond_length,[0.8,p2d.bond_first_minima_left])
                bpm.plot_points_with_given_bonds(
                    self.txyz[frame],
                    list_bond_index, bond_color='k', particle_size=1)
                bpm.plot_traps(self.traps)
                count_polygon_relative = p2d.get_conditional_bonds_and_simplices_vertex_length(
                    p2d.ridge_first_minima_left)
                # count_polygon_relative = p2d.get_conditional_bonds_and_simplices_bond_length(p2d.bond_first_minima_left)
                fig, ax = p2d.draw_polygon_patch_oop(fig, ax, at.color3, 3)
                # fig,ax = p2d.draw_polygon_patch_oop(fig,ax,at.color4,4)#fig,ax =
                # fig,ax = p2d.draw_polygon_patch_oop(fig,ax,at.color6,6)
                xx = self.box[0]/2 - dis_cut
                yy = self.box[1]/2 - dis_cut
                lim = [[-xx, xx], [-yy, yy]]
                bpm.restrict_axis_limitation(lim[0], lim[1])
                bpm.save_figure(prefix_write+str(frame)+'_bond_polygon.png')
                # a=10
                # lim = [[-a,a],[-a,a]]#-4,12,-5,16
                # p2d.draw_bonds_simplex_conditional_oop(png_filename=png_filename,x_unit='($\sigma$)',axis_limit=lim,fig=fig,ax=ax)

    def get_cnks(self, dis_cut=0, csv_filename='cnks.csv', last_frame=False, cn_final_check=6, bench_mark=0.5):  # , return_value=False
        txyz = self.txyz
        nframes = np.shape(txyz)[0]
        n_params = 1+9
        record_cnks = np.zeros((nframes, n_params))  # cn0~8, 9 parameters + 1 time
        record_cnks[:, 0] = np.linspace(1, nframes, nframes)  # time steps

        frame = nframes-1
        gf = get_data_from_a_gsd_frame(points=self.txyz[frame], traps=self.traps)
        ccn = gf.get_cn_k_from_a_gsd_frame(dis_cut, k=None)
        record_cnks[frame, 1:] = np.transpose(ccn[1:n_params])
        if ccn[cn_final_check] > bench_mark:
            go_on = True
        else:
            go_on = False

        for frame in range(nframes):
            if not go_on:
                break
            if last_frame:
                frame = nframes-1
            # if frame>141:
            gf = get_data_from_a_gsd_frame(points=self.txyz[frame], traps=self.traps)
            ccn = gf.get_cn_k_from_a_gsd_frame(dis_cut, k=None)
            record_cnks[frame, 1:] = np.transpose(ccn[1:n_params])
            if last_frame:
                break
        import pandas as pd
        col = ['t']
        for i in range(n_params-1):
            col += ['cn'+str(i+1)]  # +2
        df = pd.DataFrame(record_cnks, columns=col)
        df.to_csv(csv_filename)
        # if return_value:
        #    return df.values

    def get_bicolor_displacement_field(self, frame1, frame2, lim):
        R"""
        import workflow_analysis as wa
        sdf = wa.show_disp_field()
        seed=[0,1,2,8,9]
        frame=[12,10,29,6,8]
        lim=[[[-5,9],[-5,10]],[[11,21],[-13,1]],[[9,21],[-7,7]],[[-9,4],[-12,3]],[[-4,8],[-3,16]]]
        for i in range(5):
            print(seed[i])
            sdf.get_bicolor_disp(seed[i],frame[i],lim[i])
        """
        import proceed_file as pf
        pfile = pf.proceed_file_shell()

        prefix_write = pfile.create_folder(self.directory_to_trajectory_data, 'pin_check')

        txyz_stable = self.txyz_stable

        df = pa.dynamical_facilitation_module()
        df.get_pin_bool(self.traps, txyz_stable, prefix_write, 1.0)
        # plot
        # frame=8
        file_t_pin_bool = self.directory_to_trajectory_data+'/pin_check/t_pin_bool.npy'
        t_pin_bool = np.load(file_t_pin_bool)

        p2d = pa.dynamic_points_analysis_2d(txyz_stable)
        p2d.displacement_field_module()
        # lim=[[-25,25],[-20,20]]
        # _part
        png_filename = prefix_write+'displacement_field_xy_' + \
            str(int(frame1))+'_'+str(int(frame2))+'.png'
        p2d.displacemnt_field.get_bicolor_disp(
            t_pin_bool[frame2],
            frame1, frame2, plot=True, png_filename=png_filename, limit=lim, traps=self.traps)
        # png_filename=prefix_write+'displacement_field_xy_0_2000.png'#'_part.png'
        # p2d.displacemnt_field.get_bicolor_disp(t_pin_bool[2000],0,2000,plot=True,png_filename=png_filename)

    # def get_cnk_vs_t(self):
