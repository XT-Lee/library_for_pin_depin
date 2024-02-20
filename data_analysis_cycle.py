import points_analysis_2D as pa
import numpy as np
import matplotlib.pyplot as plt
import os
import proceed_file as pf


def save_exp_20230113_6(i, stable=True):
    R"""
    import data_analysis_cycle as dac
    dac.save_exp_20230113_6()
    """
    import pandas as pd
    import particle_tracking as pt
    spe = pt.save_points_from_exp()
    path_to_folder = '/home/remote/xiaotian_file/data/20230113'
    video_name = 'DefaultVideo_6'
    spe.set_worksapce(path_to_folder, video_name)
    tsf_filename = 'trap_honeycomb_part.txt'

    # image size
    pixel_size = np.array([1008, 1014])
    pixel_to_um = 3.0/32.0
    um_size = pixel_size*pixel_to_um

    # trap location
    # adjust coordinations to match particle and traps
    # standard_tune, precise_tune
    center = np.array([48, 48])+[10, 10]
    trap_locate = np.array([38.53, -40.92])+np.array([13, -11])
    xy_adjust = center+trap_locate
    pos, trap_filename = spe.get_trap_positions(tsf_filename, xy_adjust, 1, 90)

    directory = spe.path_to_results+'/'
    dataname = video_name
    if stable:
        txyz = np.load(directory+'txyz_stable.npy')
        xyi = txyz[i]

    else:
        txyz = pd.read_csv(directory+'txyz.csv')
        frame_num = txyz['frame'].values.max()+1
        if i < 0:
            i = frame_num+i
        txyz_ith = txyz[txyz['frame'] == i]
        xyi = txyz_ith[['x', 'y', 'z']].values
        xyi = xyi*pixel_to_um
    a_frame = pa.static_points_analysis_2d(xyi)

    data_name = dataname
    prefix = directory
    trap_filename = trap_filename
    bond_cut_off = 8
    trap_lcr = 0.8
    str_index = data_name
    png_filename1 = prefix + 'bond_hist_index'+str_index+'_'+str(int(i))+'.png'
    png_filename2 = prefix + 'bond_plot_1st_minima_index'+str_index+'_'+str(int(i))+'.png'
    a_frame.get_first_minima_bond_length_distribution(
        lattice_constant=1, hist_cutoff=bond_cut_off, png_filename=png_filename1)  # ,png_filename=png_filename1
    a_frame.draw_bonds_conditional_bond_oop(
        check=[0.4, a_frame.bond_first_minima_left],
        png_filename=png_filename2, x_unit='(um)', LinearCompressionRatio=trap_lcr,
        trap_filename=trap_filename, axis_limit=um_size)
    # dpa = pa.dynamic_points_analysis_2d(txyz,'exp')
    # dpa.plot_bond_neighbor_change_oop(data_name=dataname,prefix=directory,final_cut=True,#init_cut=True,#final_cut=True,
    #                    trap_filename=trap_filename,bond_cut_off=10,
    #                    trap_lcr=0.8)


def save_exp_20230113_8(i, stable=True):
    R"""
    import data_analysis_cycle as dac
    dac.save_exp_20230113_6()
    """
    import pandas as pd
    import particle_tracking as pt
    spe = pt.save_points_from_exp()
    path_to_folder = '/home/remote/xiaotian_file/data/20230113'
    video_name = 'DefaultVideo_8'
    spe.set_worksapce(path_to_folder, video_name)
    tsf_filename = 'trap_kagome_part.txt'

    # image size
    pixel_size = np.array([1024, 1024])
    pixel_to_um = 3.0/32.0
    um_size = pixel_size*pixel_to_um

    # trap location
    # adjust coordinations to match particle and traps
    # standard_tune, precise_tune
    center = np.array([48, 48])+[10, 10]
    trap_locate = np.array([30.55, -38.26])+np.array([0.5, -9])
    xy_adjust = center+trap_locate
    pos, trap_filename = spe.get_trap_positions(tsf_filename, xy_adjust, 1, 90)

    directory = spe.path_to_results+'/'
    dataname = video_name
    if stable:
        txyz = np.load(directory+'txyz_stable.npy')
        xyi = txyz[i]

    else:
        txyz = pd.read_csv(directory+'txyz.csv')
        frame_num = txyz['frame'].values.max()+1
        if i < 0:
            i = frame_num+i
        txyz_ith = txyz[txyz['frame'] == i]
        xyi = txyz_ith[['x', 'y', 'z']].values
        xyi = xyi*pixel_to_um
    a_frame = pa.static_points_analysis_2d(xyi)

    data_name = dataname
    prefix = directory
    trap_filename = trap_filename
    bond_cut_off = 8
    trap_lcr = 0.88+0.012
    str_index = data_name
    png_filename1 = prefix + 'bond_hist_index'+str_index+'_'+str(int(i))+'.png'
    png_filename2 = prefix + 'bond_plot_1st_minima_index'+str_index+'_'+str(int(i))+'.png'
    a_frame.get_first_minima_bond_length_distribution(
        lattice_constant=1, hist_cutoff=bond_cut_off, png_filename=png_filename1)  # ,png_filename=png_filename1
    a_frame.draw_bonds_conditional_bond_oop(
        check=[0.4, a_frame.bond_first_minima_left],
        png_filename=png_filename2, x_unit='(um)', LinearCompressionRatio=trap_lcr,
        trap_filename=trap_filename, axis_limit=um_size)


def get_KBT_pressure(file_log):
    data = np.genfromtxt(fname=file_log, skip_header=True)
    average_KBT = np.average(data[300:-1, 2])
    average_pressure = np.average(data[300:-1, 3])

    # check temperature
    return average_KBT, average_pressure


def rearrange_data():
    R"""
    Purpose:
        merge multiple txt file [index psi_k_global psi_k_rate] into one,
        and at last get
        [1     2 3                        4            5         ]
        [index k linear_compression_ratio psi_k_global psi_k_rate]
    """
    lcr_list = np.linspace(0.77, 0.88, 12)
    # control pointers, not data.
    # linear_compression_ratio2=0.80
    fn1 = '/home/tplab/Downloads/91-102'
    # fn2='/home/tplab/Downloads/103-132'
    # fn3='/home/tplab/Downloads/133-192'
    data1 = readdata(fn1)
    # data2=readdata(fn2)
    # data3=readdata(fn3)
    sp1 = np.shape(data1)
    # sp2=np.shape(data2)
    # sp3=np.shape(data3)

    data_ex = np.zeros((sp1[0], sp1[1]+2))  # +sp2[0]+sp3[0]
    data_ex[0:sp1[0], 0] = data1[:, 0]
    data_ex[0:sp1[0], 1] = 800
    data_ex[0:sp1[0], 2] = lcr_list
    data_ex[0:sp1[0], 3:5] = data1[:, 1:3]

    print(data_ex)
    # save_filename=fn1+'kl'
    save_filename = '/home/tplab/Downloads/91-102kl'
    np.savetxt(save_filename, data_ex)
    '''
    #merge two 5-column list
    fn1='/home/tplab/Downloads/103-132kl'
    fn2='/home/tplab/Downloads/133-192kl'
    data1=readdata(fn1)
    data2=readdata(fn2)
    sp1=np.shape(data1)
    sp2=np.shape(data2)
    data_ex=np.zeros((sp1[0]+sp2[0],sp1[1]))
    data_ex[0:sp1[0]]=data1[:]
    data_ex[sp1[0]:(sp1[0]+sp2[0])]=data2[:]
    '''
    '''
    #merge two 3-column list
    data_ex[0:30,1]=k_list
    data_ex[30:60,1]=k_list
    data_ex[0:30,2]=linear_compression_ratio1
    data_ex[30:60,2]=linear_compression_ratio2
    '''
    '''
    #add kl for 3-column list, generate 5-column list
    lcr_list=np.linspace(0.77,0.88,12)
    fn1='/home/tplab/Downloads/91-102'
    data1=readdata(fn1)
    sp1=np.shape(data1)
    data_ex=np.zeros((sp1[0],sp1[1]+2))
    data_ex[0:sp1[0],0]=data1[:,0]
    data_ex[0:sp1[0],1]=800
    data_ex[0:sp1[0],2]=lcr_list
    data_ex[0:sp1[0],3:5]=data1[:,1:3]
    '''


def readdata(fn):
    data = np.loadtxt(fn)
    return data


def plotHarmonicKAndPsi():
    data = readdata("/home/tplab/Downloads/193-205kl")
    data_psi3 = data[:, [1, 3]]
    data_psi6 = data[:, [1, 4]]
    plt.figure()
    p3 = plt.scatter(data_psi3[:, 0], data_psi3[:, 1], c='r')
    p6 = plt.scatter(data_psi6[:, 0], data_psi6[:, 1], c='b')
    plt.legend(handles=[p3, p6], labels=['Psi3', 'Psi6'], loc='best')
    plt.xlabel('Harmonic_K')
    plt.ylabel('Psi_3 & Psi6')
    plt.show()


def phase_diagram(select=True):
    if select:
        tt = 'Psi_3_ratio'  # tt='Psi_3_ratio'
        prefix = '/home/tplab/Downloads/'
        name = '103-192kl'
        filename = prefix+name
        data = readdata(filename)

        k_unit = 0.5*data[:, 1]*np.multiply(1, 1)
        k_unit = k_unit.astype(int)
        plt.figure()
        plt.scatter(data[:, 2]*100, k_unit, c=data[:, 4])
        plt.title(tt+'(>0.9)(E_yukawa=233,substrate=400)')
        plt.colorbar()
        plt.xlabel('linear_compression_ratio/%')
        plt.ylabel('depth of trap/kT')
        png_filename = filename+tt
        plt.savefig(png_filename)
        plt.show()
    else:
        tt = 'Psi_3'  # tt='Psi_3_ratio'
        prefix = '/home/tplab/Downloads/'
        name = '103-192kl'
        filename = prefix+name
        data = readdata(filename)

        k_unit = 0.5*data[:, 1]*np.multiply(1, 1)
        k_unit = k_unit.astype(int)
        plt.figure()
        plt.scatter(data[:, 2]*100, k_unit, c=data[:, 3])
        plt.title(tt+'(E_yukawa=233,substrate=400)')
        plt.colorbar()
        plt.xlabel('linear_compression_ratio/%')
        plt.ylabel('depth of trap/kT')
        png_filename = filename+tt
        plt.savefig(png_filename)
        plt.show()


def phase_diagram_line(select=True):
    prefix = '/home/tplab/Downloads/'
    name = '91-102kl'
    filename = prefix+name
    data = readdata(filename)

    # k_unit=0.5*data[:,1]*np.multiply(1,1)
    # k_unit=k_unit.astype(int)

    if select:
        tt = 'Psi_3_ratio'  # tt='Psi_3_ratio'
        plt.figure()
        plt.plot(data[:, 2]*100, data[:, 4])
        plt.title(tt+'(>0.9)(E_yukawa=233,)')
        plt.xlabel('linear_compression_ratio/%')
        plt.ylabel(tt)
        png_filename = filename+tt
        plt.savefig(png_filename)
        plt.show()
    else:
        tt = 'Psi_3'  # tt='Psi_3_ratio'
        plt.figure()
        plt.plot(data[:, 2]*100, data[:, 3])
        plt.title(tt+'(E_yukawa=233)')
        plt.xlabel('linear_compression_ratio/%')
        plt.ylabel(tt)
        png_filename = filename+tt
        plt.savefig(png_filename)
        plt.show()


def save_from_gsd(
        simu_index=None, seed=None, frame_cut=0, trajectory=False, save_result_txt=False,
        displacement_field=False, final_cut=False, psik=False, psik_plot=None, neighbor_cloud=False,
        coordination_number=False, lattice_constant=3, p_cairo=False, bond_plot=False,
        bond_plot_gr=False, show_traps=False, trap_filename=None, trap_lcr=None, list_traps=None,
        gr=False, sk=False, log_sk=False, msd=False, single_particle=False, account='tplab'):
    R"""
    Introduction:
        Read a gsd file and save a series of analyzed results as follow.
        trajectory:
        displacement_field:
        psik: global psi_k vs time.
        neighbor_cloud:
        coordination_number:
        coordination_number3_plot:
        final_cut: true to proceed the last frame only.

    Format:
        [Psi_3_global,Psi_6_global]

    example:
        import data_analysis_cycle as da
        i=3003
        while i<3062:
            da.save_from_gsd(simu_index=i,seed=9,
                                bond_plot =True,
                                show_traps=True,
                                trap_filename="/home/tplab/hoomd-examples_0/testkagome_part3-11-6",
                                trap_lcr=0.86)
            i=i+1
    example2:
        i=1233
        while i<1243:
            da.save_from_gsd(simu_index=i,final_cut=True,
                        bond_plot =True,
                        gr=True,
                        sk=True)#,seed=9
            i+=1
    example3:
        import data_analysis_cycle as da
        da.save_from_gsd(simu_index=4728,seed=9,
        coordination_number=True,
        bond_plot=True,
        show_traps=True,
        trap_filename='/home/tplab/hoomd-examples_0/testkagome3-11-6',
        trap_lcr=0.853,
        account='remote',)
    """
    import freud
    # prefix = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_0/"
    prefix = "/home/tplab/hoomd-examples_0/record/"
    log_prefix = prefix
    # prefix='/home/'+account+'/Downloads/'#'/home/tplab/Downloads/'
    # log_prefix='/home/'+account+'/hoomd-examples_0/'#'/home/tplab/hoomd-examples_0/'
    # /media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_0/testhoneycomb3-8-12-part1

    # load time steps
    if seed is None:
        str_index = str(int(simu_index))
        gsd_data = pf.proceed_gsd_file(simu_index=simu_index)
    else:
        str_index = str(int(simu_index))+'_'+str(seed)
        file_gsd = log_prefix+'trajectory_auto'+str_index+'.gsd'  # +'_'+str(seed)
        gsd_data = pf.proceed_gsd_file(filename_gsd_seed=file_gsd, account=account)

    """file_log=log_prefix+'log-output_auto'+str_index+'.log'#+'_'+str(seed)
    log_data = np.genfromtxt(fname=file_log, skip_header=True)
    time_steps = log_data[:,0]"""
    time_steps = 0  # just in case

    # print(gsd_data.num_of_frames)
    # gsd_data.trajectory.
    if msd:
        # print('this function msd is fault!')
        # pass
        # load particle trajectory [N_frames,N_particles,system_dimension]

        """
        iframes = 0
        nframes=gsd_data.num_of_frames
        Np_edge_cut=np.shape(gsd_data.edge_cut_positions_list)
        pos_list = np.zeros([nframes,Np_edge_cut[1],3])#gsd_data.trajectory[0].particles.N,
        while iframes < nframes:
            af = gsd_data.trajectory.read_frame(iframes)
            pos_list[iframes] = af.particles.position[gsd_data.edge_cut_positions_list]
            iframes = iframes + 1
        #select particles never walking across boundaries.
        cut = pa.select_particle_in_box(pos_list)
        particles_reasonable=cut.compute()
        msds = freud.msd.MSD(af.configuration.box)#the class is fault,,'direct'
        msds.compute(positions=pos_list[:,particles_reasonable,:])#,images=pos_list
        #print(msds.msd)#print(msds.particle_msd)

        """
        iframes = 0
        nframes = gsd_data.num_of_frames
        af = gsd_data.trajectory.read_frame(iframes)
        pos_list = np.zeros([nframes, af.particles.N, 3])  # gsd_data.trajectory[0].particles.N,
        while iframes < nframes:
            af = gsd_data.trajectory.read_frame(iframes)
            pos_list[iframes] = af.particles.position
            iframes = iframes + 1
        msds = freud.msd.MSD(af.configuration.box)  # the class is fault,,'direct'
        msds.compute(positions=pos_list)  # ,images=pos_list
        # print(msds.msd)#print(msds.particle_msd)
        # Plot the MSD

        plt.figure()
        plt.plot(msds.msd)
        plt.title("Mean Squared Displacement")
        plt.xlabel("$t$")
        plt.ylabel("MSD$(t)$")
        png_filename = prefix + 'msd_'+'index'+str_index+'.png'
        plt.savefig(png_filename)  # png_filename
        plt.close()

        if single_particle:
            plt.figure()
            lenth = np.shape(msds.particle_msd[0, :])
            for i in range(lenth[0]):
                # ,msds.particle_msd[:,1],msds.particle_msd[:,2])
                plt.semilogy(msds.particle_msd[10:-20, i], base=10)
            plt.title("Mean Squared Displacement")
            plt.xlabel("$t$")
            plt.ylabel("MSD$(t)$")
            # plt.set_yscale("log",base=2)
            png_filename = prefix + 'msd_'+'index'+str_index+'_single.png'
            print(png_filename)
            plt.savefig(png_filename)  # png_filename
            plt.close()

    if trajectory:
        gsd_data.plot_trajectory()

    for i in range(gsd_data.num_of_frames):
        if i < 0:
            pass
        else:

            if final_cut:
                i = gsd_data.num_of_frames-1  # i=9#!!! 23
            list_points = np.logical_not(list_traps)
            # gsd_data.read_a_frame(i),hide_figure=False
            a_frame = pa.static_points_analysis_2d(
                points=gsd_data.trajectory[i].particles.position[list_points, 0:2])

            if save_result_txt:
                result_filename = prefix+'index'+str_index
                points = snap.particles.position[:]  # temp
                np.savetxt(result_filename, points)  # temp

            if displacement_field:
                png_filename1 = prefix + 'Displacement_Field_xy_' + \
                    'index'+str_index+'_'+str(int(i))+'.png'
                gsd_data.get_displacement_field_xy(i, plot=True, png_filename=png_filename1)
                png_filename2 = prefix + 'Displacement_Field_hist_log_' + \
                    'index'+str_index+'_'+str(int(i))+'.png'
                gsd_data.get_displacement_field_distribution(
                    i, log_mode=True, png_filename=png_filename2)
                png_filename3 = prefix + 'Displacement_Field_hist_' + \
                    'index'+str_index+'_'+str(int(i))+'.png'
                gsd_data.get_displacement_field_distribution(i, png_filename=png_filename3)

            if psik:
                if not "record_psik" in locals():  # check if the variable exists
                    # load Psi_k s
                    record_psik = np.zeros((gsd_data.num_of_frames, 3))  # [time_steps,psi3,psi6]
                    record_psik[:, 0] = time_steps  # [0:25]*20
                a_frame.get_bond_orientational_order(k_set=3)
                record_psik[i, 1] = a_frame.Psi_k_global_cut_edge
                a_frame.get_bond_orientational_order(k_set=6)
                record_psik[i, 2] = a_frame.Psi_k_global_cut_edge

            if not psik_plot is None:
                png_filename_psik = prefix + 'bond_orientational_order_'+str(
                    int(psik_plot))+'_'+'index'+str_index+'_'+str(int(i))+'.png'
                a_frame.get_bond_orientational_order(
                    k_set=psik_plot, plot=True, png_filename=png_filename_psik)

            if neighbor_cloud:
                folder_name = prefix+"record_"+str_index  # +"/"
                # check if the folder exists
                isExists = os.path.exists(folder_name)
                if isExists:
                    pass
                else:
                    os.makedirs(folder_name)
                png_filename = folder_name+"/"+'neighbor_cloud_1st_minima_index' + \
                    str_index+'_'+str(int(i))+'.png'
                # a_frame.get_neighbor_cloud(png_filename=png_filename)
                a_frame.get_neighbor_cloud_method_1st_minima_bond(png_filename=png_filename)

            if coordination_number:
                R"""
                CN0 % should be 0 for all the particles must be linked by bond.
                CN1 % is likely to be edge?
                CN2 % in body(edge-cutted) shows the mechanical unstability
                CN3 % shows the proportion of honeycomb.
                CN4 % shows the proportion of kagome.
                CN6 % shows the proportion of hexagonal.
                CN5/7 % shows the proportion of disclination.

                record_cn: Nframes of [time_step, CN0, CN1,..., CN12]
                """
                # print('index '+str(i))
                # print(snap.particles.position[137])
                a_frame.get_coordination_number_conditional(
                    lattice_constant=lattice_constant)  # cut edge to remove CN012
                ccn = a_frame.count_coordination_ratio  # [time_steps,psi3,psi6]
                ccn = np.transpose(ccn)
                if not "record_cn" in locals():  # check if the variable exists
                    # load CN_k s
                    record_cn = np.zeros((gsd_data.num_of_frames, np.shape(ccn)[1]+1))
                    # time_steps##gsd frame is different from log frame for period set 100 vs 2e3
                    record_cn[:, 0] = range(10)
                # print(np.shape(ccn)[1])
                record_cn[i, 1:np.shape(ccn)[1]+1] = ccn  # [0:np.shape(ccn)[1]-1]

            if bond_plot:
                if final_cut:
                    # bond_plot+trap_plot
                    png_filename1 = 'bond_hist_index'+str_index+'_'+str(int(i))+'.png'  # prefix +
                    # prefix +
                    png_filename2 = 'bond_plot_1st_minima_index'+str_index+'_'+str(int(i))+'.png'
                else:
                    folder_name = prefix+"record_"+str_index  # +"/"
                    # check if the folder exists
                    isExists = os.path.exists(folder_name)
                    if isExists:
                        pass
                    else:
                        os.makedirs(folder_name)
                    # bond_plot+trap_plot
                    png_filename1 = folder_name+"/" + 'bond_hist_index' + \
                        str_index+'_'+str(int(i))+'.png'
                    png_filename2 = folder_name+"/" + 'bond_plot_1st_minima_index' + \
                        str_index+'_'+str(int(i))+'.png'
                # a_frame is static_points_analysis_2d
                # a_frame.get_first_minima_bond_length_distribution(lattice_constant=3)#,png_filename=png_filename1
                spa = a_frame
                pa.bond_plot_module()
                a_frame.get_first_minima_bond_length_distribution()
                # a_frame.get_first_minima_ridge_length_distribution()
                bpm = pa.bond_plot_module()
                bpm.restrict_axis_property_relative(spa.points, '($\sigma$)')
                # list_bond_index = bpm.get_bonds_with_conditional_ridge_length(spa.voronoi.ridge_length,spa.voronoi.ridge_points,spa.ridge_first_minima_left)
                list_bond_index = bpm.get_bonds_with_conditional_bond_length(
                    spa.bond_length, [0.9, spa.bond_first_minima_left])
                # color_name: https://www.cssportal.com/html-colors/x11-colors.php
                bond_color = 'k'  # 'gold'#'mediumseagreen'#'tan'#'bisque'#'gold'#'darkorange'
                bpm.plot_points_with_given_bonds(
                    spa.points, list_bond_index, 50, bond_color, bond_color, bond_width=1)  # 200
                bpm.plot_traps(LinearCompressionRatio=1.0, trap_filename=trap_filename,
                               mode='map', trap_color='r', trap_size=10)  # array
                semibox = gsd_data.trajectory[0].configuration.box[0:2]/2
                bpm.restrict_axis_limitation([-semibox[0], semibox[0]], [-semibox[1], semibox[1]])
                save_filename = png_filename2
                bpm.save_figure(png_filename=save_filename)
                """
                a_frame.draw_bonds_conditional_bond_oop(check=[0.4, a_frame.bond_first_minima_left], png_filename=png_filename2,
                                                LinearCompressionRatio=trap_lcr,trap_filename=trap_filename,
                                                x_unit='($\sigma$)',axis_limit=[10,10])#show_traps=show_traps,

                """

            if bond_plot_gr:
                if final_cut:
                    # bond_plot+trap_plot
                    png_filename1 = prefix + 'bond_gr_index'+str_index+'_'+str(int(i))+'.png'
                    png_filename2 = prefix + 'bond_plot_gr_1st_minima_index' + \
                        str_index+'_'+str(int(i))+'.png'
                else:
                    folder_name = prefix+"record_"+str_index  # +"/"
                    # check if the folder exists
                    isExists = os.path.exists(folder_name)
                    if isExists:
                        pass
                    else:
                        os.makedirs(folder_name)
                    # bond_plot+trap_plot
                    # png_filename1 = folder_name+"/" +'bond_hist_index'+str_index+'_'+str(int(i))+'.png'
                    png_filename2 = folder_name+"/" + 'bond_plot_1st_minima_index' + \
                        str_index+'_'+str(int(i))+'.png'

                rdf = freud.density.RDF(bins=150, r_max=15.0, r_min=1.0)
                rdf.compute(system=snap)
                a_frame.draw_radial_distribution_function_and_first_minima(
                    rdf, lattice_constant=3, png_filename=png_filename1)
                a_frame.draw_bonds_conditional_bond(
                    check=[0.4, a_frame.bond_first_minima_left],
                    png_filename=png_filename2, show_traps=show_traps,
                    LinearCompressionRatio=trap_lcr, trap_filename=trap_filename)
                """
                rdf = freud.density.RDF(bins=150, r_max=15.0)#
                rdf.compute(system=snap)
                #print(rdf.bin_centers) print(rdf.bin_counts)
                rdf.plot()
                fig_type = 'gr'
                data_filename=prefix+fig_type+'_index'+str_index+'_'+str(int(i))+'.png'
                plt.savefig(data_filename)
                plt.close()
                """
                """
                #checked right
                import gsd.hoomd
                import freud
                traj = gsd.hoomd.open('/home/tplab/hoomd-examples_0/trajectory_auto5208_9.gsd')
                rdf = freud.density.RDF(bins=50,r_max=10)
                rdf.compute(system=traj[-1])
                r =rdf.bin_centers
                y = rdf.rdf
                import matplotlib.pyplot as plt
                fig,ax = plt.subplots()
                rdf.plot(ax=ax)
                plt.savefig('/home/tplab/Downloads/gr.png')
                """

            if sk:
                sk = freud.diffraction.DiffractionPattern()
                snap = gsd_data.trajectory.read_frame(i)
                sk.compute(system=snap)
                if log_sk:
                    fig_type = 'log_sk'
                    data_filename = prefix+fig_type+'_index'+str_index+'_'+str(int(i))+'.png'
                    ax = sk.plot(vmin=0.01, vmax=1)
                else:
                    fig_type = 'sk'
                    data_filename = prefix+fig_type+'_index'+str_index+'_'+str(int(i))+'.png'
                    fig, ax = plt.subplots()
                    im = ax.pcolormesh(sk.k_values, sk.k_values,
                                       sk.diffraction, cmap='afmhot')  # im =
                    # ax.colorbar().remove()
                    fig.colorbar(im)
                    ax.axis('equal')
                """
                #method1
                ax = sk.plot()
                ax.pcolormesh(sk.k_values,sk.k_values,sk.diffraction,cmap='afmhot')

                #method2
                fig,ax = plt.subplots()
                #print(sk.k_values)
                X, Y = np.meshgrid(sk.k_values, sk.k_values)
                im = ax.pcolormesh(X,Y,sk.diffraction,cmap='summer')#'afmhot' im =
                #https://matplotlib.org/stable/tutorials/colors/colormaps.html
                # ?highlight=afmhot
                #ax.colorbar().remove()
                fig.colorbar(im)
                ax.axis('equal')
                """
                plt.savefig(data_filename)
                plt.close()
                # ax.pcolormesh(X, Y, Z,cmap="plasma",)
                """
                maybe that the loglog map is not suitable for my sk.
                linear colorbar is the right choice
                """
            if final_cut:
                break
    if psik:
        plt.figure()
        plt.plot(record_psik[:, 0], record_psik[:, 1], label='Psi_3')  # psi3
        plt.plot(record_psik[:, 0], record_psik[:, 2], label='Psi_6')  # psi6
        plt.legend()
        plt.title('Psi_3 VS Psi_6 '+'index:'+str_index)
        plt.xlabel('time(steps)')
        plt.ylabel('Psi_k(1)')
        # plt.show()
        png_filename = prefix + 'T_VS_Psi_k_'+'index'+str_index+'.png'
        plt.savefig(png_filename)
        plt.close()
    if p_cairo:
        if not "record_pcairo" in locals():  # check if the variable exists
            # record_pcairo: Nframes of [time_step, cn3, cn4, pcairo]
            record_pcairo = np.array(record_cn[:, :4])
            record_pcairo[:, 1] = record_cn[:, 4]
            record_pcairo[:, 2] = record_cn[:, 5]
            record_pcairo[:, 3] = 0

        for i in range(gsd_data.num_of_frames):
            record_pcairo[i, 3] = a_frame.get_cairo_order_parameter(
                record_cn[i, 4], record_cn[i, 5])  # ccn[3],ccn[4])

        fig, ax = plt.subplots()
        if frame_cut == 0:
            ax.plot(record_cn[:, 0], record_cn[:, 4], label='CN_3')
            ax.plot(record_cn[:, 0], record_cn[:, 5], label='CN_4')
            ax.plot(record_cn[:, 0], record_cn[:, 7], label='CN_6')
            ax.plot(record_cn[:, 0], record_pcairo[:, 3], label='Pcairo')
            png_filename = prefix + 'T_VS_Pcairo'+'index'+str_index+'egcut'+'.png'

        plt.legend()
        plt.title('CN_k '+'index:'+str_index)
        plt.xlabel('time(steps)')
        plt.ylabel('Order Parameter(1)')
        # plt.show()
        plt.savefig(png_filename)
        record_filename = prefix + 'T_VS_Pcairo_cut'+'index'+str_index+'.txt'
        np.savetxt(record_filename, record_cn)  # np.savetxt(record_filename,record_cn)
        plt.close()

        scp = pa.show_cairo_order_parameter()
        fig, ax = scp.plot_diagram()
        # fig,ax = plt.subplots()
        if frame_cut == 0:
            ax.plot(record_cn[:, 4], record_cn[:, 5], 'r',)
            # ax.plot(record_cn[:,0],record_pcairo[:,3],label='Pcairo')
            png_filename = prefix + 'trajectory_Pcairo'+'index'+str_index+'egcut'+'.png'

        plt.savefig(png_filename)
        plt.close()

    if False:  # coordination_number:
        # txt_filename = prefix +'T_VS_CN_k_tcut'+'index'+str_index+'egcut'+'.txt'
        # np.savetxt(txt_filename,record_cn)
        plt.figure()
        if frame_cut == 0:  # frame_cut is set to abstract a part of the process to watch in detail
            # plt.plot(record_cn[:,0],record_cn[:,1],label='CN_0')
            # plt.plot(record_cn[:,0],record_cn[:,2],label='CN_1')
            # plt.plot(record_cn[:,0],record_cn[:,3],label='CN_2')
            plt.plot(record_cn[:, 0], record_cn[:, 4], label='CN_3')
            plt.plot(record_cn[:, 0], record_cn[:, 5], label='CN_4')
            plt.plot(record_cn[:, 0], record_cn[:, 6], label='CN_5')
            plt.plot(record_cn[:, 0], record_cn[:, 7], label='CN_6')
            plt.plot(record_cn[:, 0], record_cn[:, 8], label='CN_7')
            # plt.plot(record_cn[:,0],record_cn[:,9],label='CN_8')
            # plt.plot(record_cn[:,0],record_cn[:,-1],label='CN_9')
            png_filename = prefix + 'T_VS_CN_k'+'index'+str_index+'egcut'+'.png'
        else:
            # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,1],label='CN_0')
            # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,2],label='CN_1')
            # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,3],label='CN_2')
            plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 4], label='CN_3')
            plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 5], label='CN_4')
            plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 6], label='CN_5')
            plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 7], label='CN_6')
            plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 8], label='CN_7')
            # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,9],label='CN_8')
            # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,-1],label='CN_9')
            png_filename = prefix + 'T_VS_CN_k_tcut'+'index'+str_index+'egcut'+'.png'
        plt.legend()
        plt.title('CN_k '+'index:'+str_index)
        plt.xlabel('time(steps)')
        plt.ylabel('CN_k(1)')
        # plt.show()
        plt.savefig(png_filename)
        record_filename = prefix + 'T_VS_CN_k_cut'+'index'+str_index+'.txt'
        np.save(record_filename, record_cn)  # np.savetxt(record_filename,record_cn)
        plt.close()


def save_from_gsd_to_cn3(simu_index=None, seed=None, frame_cut=0,
                         final_cut=False,
                         coordination_number=False, lattice_constant=3,
                         account='tplab',
                         prefix=None):
    R"""
    Introduction:
        Read a gsd file and save a series of analyzed results as follow.
        coordination_number:
    Format:
        [Psi_3_global,Psi_6_global]

    example:

    """
    # prefix='/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_0/'#'/home/'+account+'/Downloads/cn3-t/'#'/home/tplab/Downloads/'
    # '/home/'+account+'/hoomd-examples_0/'#'/home/tplab/hoomd-examples_0/'
    log_prefix = '/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/'
    # /media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_0/
    # load time steps
    if seed is None:
        str_index = str(int(simu_index))
        gsd_data = pf.proceed_gsd_file(simu_index=simu_index)
    else:
        str_index = str(int(simu_index))+'_'+str(seed)
        file_gsd = log_prefix+'trajectory_auto'+str_index+'.gsd'  # +'_'+str(seed)
        gsd_data = pf.proceed_gsd_file(filename_gsd_seed=file_gsd, account=account)

    # file_log=log_prefix+'log-output_auto'+str_index+'.log'#+'_'+str(seed)
    # log_data = np.genfromtxt(fname=file_log, skip_header=True)
    time_steps = range(2001)  # log_data[:,0]

    for i in range(gsd_data.num_of_frames):
        if final_cut:
            i = gsd_data.num_of_frames-1  # i=9#!!! 23

        a_frame = pa.static_points_analysis_2d(points=gsd_data.read_a_frame(i))  # hide_figure=False

        if coordination_number:
            R"""
            CN0 % should be 0 for all the particles must be linked by bond.
            CN1 % is likely to be edge?
            CN2 % in body(edge-cutted) shows the mechanical unstability
            CN3 % shows the proportion of honeycomb.
            CN4 % shows the proportion of kagome.
            CN6 % shows the proportion of hexagonal.
            CN5/7 % shows the proportion of disclination.
            """
            # print('index '+str(i))
            # print(snap.particles.position[137])
            a_frame.get_coordination_number_conditional(
                lattice_constant=lattice_constant)  # cut edge to remove CN012
            ccn = a_frame.count_coordination_ratio  # [time_steps,psi3,psi6]
            ccn = np.transpose(ccn)
            if not "record_cn" in locals():  # check if the variable exists
                # load CN_k s
                # hoomd_v43 remove the 1st init_frame, so the record_cn should be added 1
                record_cn = np.zeros((gsd_data.num_of_frames+1, np.shape(ccn)[1]+1))
                # range(10)##gsd frame is different from log frame for period set 100 vs 2e3
                record_cn[:, 0] = time_steps
                record_cn[:, 0] *= 1e3
                record_cn[0, 6] = 1.0
            # print(np.shape(ccn)[1])
            # hoomd_v43 remove the 1st init_frame, so the record_cn should be added 1
            record_cn[i+1, 1:np.shape(ccn)[1]+1] = ccn  # [0:np.shape(ccn)[1]-1]

        if coordination_number:  # False:#
            # txt_filename = prefix +'T_VS_CN_k_tcut'+'index'+str_index+'egcut'+'.txt'
            # np.savetxt(txt_filename,record_cn)
            plt.figure()
            if frame_cut == 0:  # frame_cut is set to abstract a part of the process to watch in detail
                # plt.plot(record_cn[:,0],record_cn[:,1],label='CN_0')
                # plt.plot(record_cn[:,0],record_cn[:,2],label='CN_1')
                # plt.plot(record_cn[:,0],record_cn[:,3],label='CN_2')
                plt.plot(record_cn[:, 0], record_cn[:, 4], label='CN_3')
                plt.plot(record_cn[:, 0], record_cn[:, 5], label='CN_4')
                plt.plot(record_cn[:, 0], record_cn[:, 6], label='CN_5')
                plt.plot(record_cn[:, 0], record_cn[:, 7], label='CN_6')
                plt.plot(record_cn[:, 0], record_cn[:, 8], label='CN_7')
                # plt.plot(record_cn[:,0],record_cn[:,9],label='CN_8')
                # plt.plot(record_cn[:,0],record_cn[:,-1],label='CN_9')
                png_filename = prefix + 'T_VS_CN_k'+'index'+str_index+'egcut'+'.png'
            else:
                # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,1],label='CN_0')
                # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,2],label='CN_1')
                # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,3],label='CN_2')
                plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 4], label='CN_3')
                plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 5], label='CN_4')
                plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 6], label='CN_5')
                plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 7], label='CN_6')
                plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 8], label='CN_7')
                # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,9],label='CN_8')
                # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,-1],label='CN_9')
                png_filename = prefix + 'T_VS_CN_k_tcut'+'index'+str_index+'egcut'+'.png'
            plt.legend()
            plt.title('CN_k '+'index:'+str_index)
            plt.xlabel('time(steps)')
            plt.ylabel('CN_k(1)')
            # plt.show()
            plt.savefig(png_filename)
            record_filename = prefix + 'T_VS_CN_k_cut'+'index'+str_index+'.txt'
            np.save(record_filename, record_cn)  # np.savetxt(record_filename,record_cn)
            plt.close()

        if final_cut:
            break

    """
    #plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,1],label='CN_0')
    #plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,2],label='CN_1')
    #plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,3],label='CN_2')
    plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,4],label='CN_3')
    plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,5],label='CN_4')
    plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,6],label='CN_5')
    plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,7],label='CN_6')
    plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,8],label='CN_7')
    #plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,9],label='CN_8')
    #plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,-1],label='CN_9')
    png_filename = prefix +'T_VS_CN_k_tcut'+'index'+str_index+'egcut'+'.png'
    plt.legend()
    plt.title('CN_k '+'index:'+str_index)
    plt.xlabel('time(steps)')
    plt.ylabel('CN_k(1)')
    #plt.show()
    plt.savefig(png_filename)
    plt.close()
    """
    record_filename = prefix + 'T_VS_CN_k_cut'+'index'+str_index+'.npy'
    np.save(record_filename, record_cn)  # np.savetxt(record_filename,record_cn)
    return record_filename


def save_from_txyz_to_cn3(simu_index=None, seed=None, frame_cut=0,
                          final_cut=False,
                          coordination_number=False, lattice_constant=3,
                          txyz=None,
                          prefix=None):
    R"""
    Introduction:
        Read a gsd file and save a series of analyzed results as follow.
        coordination_number:
    Format:
        [Psi_3_global,Psi_6_global]

    example:

    """

    if seed is None:
        str_index = str(int(simu_index))
    else:
        str_index = str(int(simu_index))+'_'+str(seed)
    Nframes, Nparticles, Ndimensions = np.shape(txyz)

    for i in range(np.shape(txyz)[0]):
        if final_cut:
            i = Nframes-1

        a_frame = pa.static_points_analysis_2d(
            points=txyz[i],
            dis_edge_cut=lattice_constant)  # hide_figure=False

        if coordination_number:
            R"""
            CN0 % should be 0 for all the particles must be linked by bond.
            CN1 % is likely to be edge?
            CN2 % in body(edge-cutted) shows the mechanical unstability
            CN3 % shows the proportion of honeycomb.
            CN4 % shows the proportion of kagome.
            CN6 % shows the proportion of hexagonal.
            CN5/7 % shows the proportion of disclination.
            """
            a_frame.get_coordination_number_conditional(
                lattice_constant=lattice_constant)  # cut edge to remove CN012
            ccn = a_frame.count_coordination_ratio  # [time_steps,psi3,psi6]
            ccn = np.transpose(ccn)
            if not "record_cn" in locals():  # check if the variable exists
                # load CN_k s
                # hoomd_v43 remove the 1st init_frame, so the record_cn should be added 1
                record_cn = np.zeros((Nframes+1, np.shape(ccn)[1]+1))
                # range(10)##gsd frame is different from log frame for period set 100 vs 2e3
                record_cn[:, 0] = range(Nframes+1)
                record_cn[:, 0] *= 1e3
                record_cn[0, 6] = 1.0
            # print(np.shape(ccn)[1])
            # hoomd_v43 remove the 1st init_frame, so the record_cn should be added 1
            record_cn[i+1, 1:np.shape(ccn)[1]+1] = ccn  # [0:np.shape(ccn)[1]-1]

        if final_cut:
            break

    if coordination_number:  # False:#
        # txt_filename = prefix +'T_VS_CN_k_tcut'+'index'+str_index+'egcut'+'.txt'
        # np.savetxt(txt_filename,record_cn)
        plt.figure()
        if frame_cut == 0:  # frame_cut is set to abstract a part of the process to watch in detail
            # plt.plot(record_cn[:,0],record_cn[:,1],label='CN_0')
            # plt.plot(record_cn[:,0],record_cn[:,2],label='CN_1')
            # plt.plot(record_cn[:,0],record_cn[:,3],label='CN_2')
            plt.plot(record_cn[:, 0], record_cn[:, 4], label='CN_3')
            plt.plot(record_cn[:, 0], record_cn[:, 5], label='CN_4')
            plt.plot(record_cn[:, 0], record_cn[:, 6], label='CN_5')
            plt.plot(record_cn[:, 0], record_cn[:, 7], label='CN_6')
            plt.plot(record_cn[:, 0], record_cn[:, 8], label='CN_7')
            # plt.plot(record_cn[:,0],record_cn[:,9],label='CN_8')
            # plt.plot(record_cn[:,0],record_cn[:,-1],label='CN_9')
            png_filename = prefix + 'T_VS_CN_k'+'index'+str_index+'egcut'+'.png'
        else:
            # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,1],label='CN_0')
            # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,2],label='CN_1')
            # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,3],label='CN_2')
            plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 4], label='CN_3')
            plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 5], label='CN_4')
            plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 6], label='CN_5')
            plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 7], label='CN_6')
            plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 8], label='CN_7')
            # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,9],label='CN_8')
            # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,-1],label='CN_9')
            png_filename = prefix + 'T_VS_CN_k_tcut'+'index'+str_index+'egcut'+'.png'
        plt.legend()
        plt.title('CN_k '+'index:'+str_index)
        plt.xlabel('time(steps)')
        plt.ylabel('CN_k(1)')
        # plt.show()
        plt.savefig(png_filename)
        record_filename = prefix + 'T_VS_CN_k_cut'+'index'+str_index+'.txt'
        np.save(record_filename, record_cn)  # np.savetxt(record_filename,record_cn)
        plt.close()
    record_filename = prefix + 'T_VS_CN_k_cut'+'index'+str_index+'.npy'
    np.save(record_filename, record_cn)  # np.savetxt(record_filename,record_cn)
    return record_filename


def save_t_cnk_png(record_cn, prefix, str_index, frame_cut=0):
    plt.figure()
    if frame_cut == 0:  # frame_cut is set to abstract a part of the process to watch in detail
        # plt.plot(record_cn[:,0],record_cn[:,1],label='CN_0')
        # plt.plot(record_cn[:,0],record_cn[:,2],label='CN_1')
        # plt.plot(record_cn[:,0],record_cn[:,3],label='CN_2')
        plt.plot(record_cn[:, 0], record_cn[:, 4], label='CN_3')
        plt.plot(record_cn[:, 0], record_cn[:, 5], label='CN_4')
        plt.plot(record_cn[:, 0], record_cn[:, 6], label='CN_5')
        plt.plot(record_cn[:, 0], record_cn[:, 7], label='CN_6')
        plt.plot(record_cn[:, 0], record_cn[:, 8], label='CN_7')
        # plt.plot(record_cn[:,0],record_cn[:,9],label='CN_8')
        # plt.plot(record_cn[:,0],record_cn[:,-1],label='CN_9')
        png_filename = prefix + 'T_VS_CN_k'+'index'+str_index+'egcut'+'.png'
    else:
        # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,1],label='CN_0')
        # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,2],label='CN_1')
        # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,3],label='CN_2')
        plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 4], label='CN_3')
        plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 5], label='CN_4')
        plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 6], label='CN_5')
        plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 7], label='CN_6')
        plt.plot(record_cn[0:frame_cut, 0], record_cn[0:frame_cut, 8], label='CN_7')
        # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,9],label='CN_8')
        # plt.plot(record_cn[0:frame_cut,0],record_cn[0:frame_cut,-1],label='CN_9')
        png_filename = prefix + 'T_VS_CN_k_tcut'+'index'+str_index+'egcut'+'.png'
    plt.legend()
    plt.title('CN_k '+'index:'+str_index)
    plt.xlabel('time(steps)')
    plt.ylabel('CN_k(1)')
    # plt.show()
    plt.savefig(png_filename)
    record_filename = prefix + 'T_VS_CN_k_cut'+'index'+str_index+'.txt'
    np.save(record_filename, record_cn)  # np.savetxt(record_filename,record_cn)
    plt.close()


def save_from_txyz_to_cn3_f(simu_index=None, seed=None, frame_cut=0,
                            final_cut=False,
                            coordination_number=False, lattice_constant=3,
                            txyz=None,
                            prefix=None):
    R"""
    Introduction:
        Read a gsd file and save a series of analyzed results as follow.
        coordination_number:
    Format:
        [Psi_3_global,Psi_6_global]

    example:

    """

    if seed is None:
        str_index = str(int(simu_index))
    else:
        str_index = str(int(simu_index))+'_'+str(seed)
    Nframes, Nparticles, Ndimensions = np.shape(txyz)

    for i in range(np.shape(txyz)[0]):
        if final_cut:
            i = Nframes-1

        a_frame = pa.static_points_analysis_2d(
            points=txyz[i],
            dis_edge_cut=lattice_constant)  # hide_figure=False

        if coordination_number:
            R"""
            CN0 % should be 0 for all the particles must be linked by bond.
            CN1 % is likely to be edge?
            CN2 % in body(edge-cutted) shows the mechanical unstability
            CN3 % shows the proportion of honeycomb.
            CN4 % shows the proportion of kagome.
            CN6 % shows the proportion of hexagonal.
            CN5/7 % shows the proportion of disclination.
            """
            a_frame.get_coordination_number_conditional(
                lattice_constant=lattice_constant)  # cut edge to remove CN012
            ccn = a_frame.count_coordination_ratio  # [time_steps,psi3,psi6]
            print(np.shape(ccn))
            # ccn = np.transpose(ccn)
            if not "record_cn" in locals():  # check if the variable exists
                # load CN_k s
                # hoomd_v43 remove the 1st init_frame, so the record_cn should be added 1
                record_cn = np.zeros((Nframes+1, np.shape(ccn)[1]+1))
                # range(10)##gsd frame is different from log frame for period set 100 vs 2e3
                record_cn[:, 0] = range(Nframes+1)
                record_cn[:, 0] *= 1e3
                record_cn[0, 6] = 1.0
            # print(np.shape(ccn)[1])
            # hoomd_v43 remove the 1st init_frame, so the record_cn should be added 1
            record_cn[i+1, 1:np.shape(ccn)[1]+1] = ccn  # [0:np.shape(ccn)[1]-1]

        if final_cut:
            break

    return ccn


class data_analysis_workflow:
    R"""
    simu:
    select * from pin_hex_to_honeycomb_part_klt_2m where HarmonicK = 700;
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT   | Psi3     | Psi6     | RandomSeed |
    |      4302 |       700 |                   0.81 |    1 | 0.927068 | 0.123686 |          9 |
    daw = dac.data_analysis_workflow()
    directory,str_simu_index = daw.gsd_to_txyz(simu_index=4302,seed=9,io_only=True)
    trap_filename='/home/remote/hoomd-examples_0/testhoneycomb3-8-12-part1'
    trap_lcr=0.81
    daw.get_defect_motion(directory,str_simu_index,trap_filename,trap_lcr)


    select * from pin_hex_to_honeycomb_klt_2m where HarmonicK = 900;
    +-----------+-----------+------------------------+------+----------+----------+------------+
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT   | Psi3     | Psi6     | RandomSeed |
    |      4634 |       900 |                   0.79 |    1 | 0.862018 | 0.159095 |          9 |

    select * from pin_hex_to_honeycomb_klt_2m where SimuIndex = 5238;
    +-----------+-----------+------------------------+------+----------+----------+------------+
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT   | Psi3     | Psi6     | RandomSeed |
    +-----------+-----------+------------------------+------+----------+----------+------------+
    |      5238 |        60 |                   0.79 |    1 | 0.885731 | 0.196146 |          9 |
    +-----------+-----------+------------------------+------+----------+----------+------------+
    select * from pin_hex_to_kagome_part_klt_2m where CoordinationNum4Rate > 0.8;
    +-----------+-----------+------------------------+------+----------------------+----------------------+------------+
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT   | CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed |
    +-----------+-----------+------------------------+------+----------------------+----------------------+------------+
    |      4428 |       300 |                   0.87 |    1 |                    0 |             0.821229 |          9 |
    |      4435 |      1000 |                   0.87 |    1 |             0.039548 |             0.824859 |          9 |
    |      4439 |       400 |                   0.88 |    1 |                    0 |             0.852273 |          9 |
    |      4440 |       500 |                   0.88 |    1 |            0.0224719 |             0.848315 |          9 |
    |      4445 |      1000 |                   0.88 |    1 |            0.0168539 |             0.837079 |          9 |
    |      4447 |       200 |                   0.89 |    1 |            0.0449438 |             0.825843 |          9 |
    |      4448 |       300 |                   0.89 |    1 |            0.0167598 |             0.865922 |          9 |
    |      4449 |       400 |                   0.89 |    1 |            0.0224719 |             0.820225 |          9 |
    |      4454 |       900 |                   0.89 |    1 |            0.0225989 |             0.858757 |          9 |
    +-----------+-----------+------------------------+------+----------------------+----------------------+------------+

    Honeycomb part pin precise index5346,5387
    mysql> select * from pin_hex_to_honeycomb_part_klt_2m where SimuIndex =5387;
    +-----------+-----------+------------------------+------+----------+----------+------------+
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT   | Psi3     | Psi6     | RandomSeed |
    +-----------+-----------+------------------------+------+----------+----------+------------+
    |      5387 |       114 |                  0.816 |    1 | 0.860764 | 0.191895 |          9 |
    |      5346 |       108 |                   0.81 |    1 | 0.90035  | 0.170656 |          9 |
    +-----------+-----------+------------------------+------+----------+----------+------------+
    """

    def __init__(self):
        R"""
        example:
            import numpy as np
            import data_analysis_cycle as dac
            import points_analysis_2D as pa
            seed=9
            index_list=np.linspace(5390,5399,10)
            lcr_list = np.linspace(0.82,1,10)
            kT=1.0
            #print(index_list,lcr_list)
            daw = dac.data_analysis_workflow()
            for i in range(10):
            directory,str_simu_index =daw.gsd_to_txyz(simu_index=index_list[i],io_only=True)
            txyz_stable = np.load('/home/remote/Downloads/'+str_simu_index+'/txyz_stable.npy')
            dpa = pa.dynamic_points_analysis_2d(txyz_stable)
            dpa.plot_trajectory(directory)
            dpa.plot_bond_neighbor_change_oop(data_name=str_simu_index,prefix=directory,final_cut=True)
        example2:
            import numpy as np
            seed=9
            index_list=np.linspace(5400,5409,10)
            lcr_list = np.linspace(1.1,2,10)
            kT=1.0
            #print(index_list,lcr_list)
            import data_analysis_cycle as dac
            import points_analysis_2D as pa
            daw = dac.data_analysis_workflow()
            for i in range(10):
                directory,str_simu_index =daw.gsd_to_txyz(simu_index=index_list[i],io_only=True)#
                txyz = np.load('/home/remote/Downloads/'+str_simu_index+'/txyz.npy')
                txyz_stable = np.load('/home/remote/Downloads/'+str_simu_index+'/txyz_stable.npy')
                #dpa = pa.dynamic_points_analysis_2d(txyz_stable)
                #dpa.plot_trajectory(directory)
                dpa = pa.dynamic_points_analysis_2d(txyz)
                dpa.plot_bond_neighbor_change_oop(data_name=str_simu_index,prefix=directory,final_cut=True,bond_cut_off=6*lcr_list[i])

        """
        pass

    def get_info_from_mysql_bond(self, account='remote'):
        import opertateOnMysql as osql
        import numpy as np
        import points_analysis_2D_freud as pa
        list_tbname = osql.showTables(' like \'pin%\'')  # _hex_to_honeycomb_klt_2m
        # print()
        tb = np.array(list_tbname[5], dtype=str)
        con = 'where HarmonicK = 60 and RandomSeed = 9'  # HarmonicK = 600,36,60
        cont = 'SimuIndex, RandomSeed, LinearCompressionRatio '
        data = osql.getDataFromMysql(table_name=tb[0], search_condition=con, select_content=cont)
        list_index_lcr = np.array(data, dtype=str)
        list_index_seed = np.char.add('_', list_index_lcr[:, 1])  # list_index[:,0],
        list_index_seed = np.char.add(list_index_lcr[:, 0], list_index_seed)
        # list_index_seed = np.char.add(list_index_seed,'.txt')
        tfn = '/home/'+account+'/hoomd-examples_0/'+'testhoneycomb3-8-12'
        xu = '(1)'
        prefix = '/home/'+account+'/Downloads/'
        for i in range(len(list_index_seed)):
            str_filename = 'index'+list_index_seed[i]
            lcr = list_index_lcr[i, 2].astype(float)
            fn = prefix+str_filename  # +'.txt'
            png = prefix+str_filename+'.png'
            spa = pa.static_points_analysis_2d(filename=fn)
            spa.get_first_minima_bond_length_distribution()
            spa.draw_bonds_conditional_bond_oop(
                check=[1, spa.bond_first_minima_left],
                png_filename=png, x_unit=xu, LinearCompressionRatio=lcr, trap_filename=tfn)
        # print(list_index_seed)

    def gsd_to_txyz(self, account='remote', simu_index=0, seed=9, io_only=False):
        R"""
        input:
            account: (string);
            simu_index: (int)index;
            seed: (int)0-9
            io_only: just return results, not proceeding data.
        return:
            directory: (str)directory with '/';
            data_name: (str)'index_seed' for example.
        """
        str_simu_index = str(int(simu_index))+'_'+str(seed)
        directory = '/home/'+account+'/Downloads/'+str_simu_index  # +'/'

        # check if the folder exists
        isExists = os.path.exists(directory)
        if not isExists:
            os.makedirs(directory)

        directory = directory+'/'
        if not io_only:
            gsd = pa.proceed_gsd_file(account=account, simu_index=simu_index, seed=seed)
            gsd.get_trajectory_data(directory)
            gsd.get_trajectory_stable_data(directory)
        return directory, str_simu_index

    def get_bond_plot(
            self, directory, data_name=None, trap_filename=None, trap_lcr=None, io_only=False):
        R"""
        input:
            directory from self.gsd_to_txyz
            trap_filename:
                '/home/remote/hoomd-examples_0/testhoneycomb3-8-12'
                '/home/remote/hoomd-examples_0/testhoneycomb3-8-12-part1'
                '/home/remote/hoomd-examples_0/testkagome3-11-6'
                '/home/remote/hoomd-examples_0/testkagome_part3-11-6'
            io_only: just return results, not proceeding data.
        return:
            a series of figures with particles(mark neighbor changes), bonds, traps
        example:
            import data_analysis_cycle as da
            get_traj = da.data_analysis()
            directory,data_name = get_traj.gsd_to_txyz('remote',4448,9,io_only=True)
            get_traj.txyz_to_bond_plot(directory,data_name,
                trap_filename='/home/remote/hoomd-examples_0/testkagome_part3-11-6',trap_lcr=0.89,
                    io_only=True)
        """
        # write a routine class
        import pandas as pd
        file_txyz_stable = directory + 'txyz_stable.npy'
        txyz_stable = np.load(file_txyz_stable)
        dpa = pa.dynamic_points_analysis_2d(txyz_stable, mode='simu')
        # particle id should be set as what in txyz_stable!
        bond_cut_off = 6
        if not io_only:
            dpa.compute_nearest_neighbor_displacements(
                csv_prefix=directory, bond_cut_off=bond_cut_off)
            file_ts_id_dxy = directory + 'ts_id_dxy.csv'
            ts_id_dxy = pd.read_csv(file_ts_id_dxy)
            if_nb_change_int, n_particle_nb_stable = dpa.monitor_neighbor_change_event(
                ts_id_dxy=ts_id_dxy,
                csv_prefix=directory)
            dpa.get_hist_neighbor_change_event(if_nb_change_int, n_particle_nb_stable, directory)
        count_nb_change_event_rate = np.load(directory+'count_nb_change_event_rate.npy')
        dpa.plot_hist_neighbor_change_event(count_nb_change_event_rate, directory)
        """
        if_nb_change_int, n_particle_nb_stable, png_filename ==dpa.monitor_neighbor_change_event(
                                                                                                 ts_id_dxy=ts_id_dxy,
                                                                                                 csv_prefix=directory)
        dpa.plot_hist_neighbor_change_event(if_nb_change_int, n_particle_nb_stable, png_filename=)
        """
        if not data_name is None:
            file_list_sum_id_nb_stable = directory + 'list_sum_id_nb_stable.csv'
            list_sum_id_nb_stable = pd.read_csv(file_list_sum_id_nb_stable)
            # dpa.plot_bond_neighbor_change(nb_change=list_sum_id_nb_stable,data_name=data_name,prefix=directory,bond_cut_off=bond_cut_off,
            #                                    show_traps=True,trap_filename='/home/remote/hoomd-examples_0/testhoneycomb3-8-12',trap_lcr=0.79)
            dpa.plot_bond_neighbor_change_oop(
                data_name=data_name, prefix=directory, nb_change=list_sum_id_nb_stable,
                bond_cut_off=bond_cut_off, trap_filename=trap_filename, trap_lcr=trap_lcr)
            """
            dpa.plot_bond_neighbor_change_oop()
            dpa.draw_bonds.draw_bonds_conditional_bond()
            dpa.draw_bonds.plot_neighbor_change(txyz_stable,nb_change)
            dpa.draw_bonds.plot_traps(trap_filename,LinearCompressionRatio)
            """

    def get_defect_motion(self, directory, data_name=None, trap_filename=None, trap_lcr=None):
        file_txyz_stable = directory + 'txyz_stable.npy'
        txyz_stable = np.load(file_txyz_stable)
        dpa = pa.dynamic_points_analysis_2d(txyz_stable, mode='simu')

        file_list_sum_id_nb_stable = directory + 'list_sum_id_nb_stable.csv'
        import pandas as pd
        list_sum_id_nb_stable = pd.read_csv(file_list_sum_id_nb_stable)
        ids = dpa.plot_neighbor_change_evolution(
            1173, 1174, directory, data_name=data_name, nb_change=list_sum_id_nb_stable,
            arrow='annotate', bond_cut_off=6, trap_filename=trap_filename, trap_lcr=trap_lcr)  # '4302_9'
        dpa.plot_neighbor_change_evolution(
            1174, 1174, directory, data_name=data_name, ids=ids, bond_cut_off=6,
            trap_filename=trap_filename, trap_lcr=trap_lcr)

    def get_string_like_motion_rank(
            self, directory, data_name=None, trap_filename=None, trap_lcr=None):
        R"""
        EXP:
            daw = dac.data_analysis_workflow()
            directory,dataname= daw.gsd_to_txyz(simu_index=4302,seed=9,io_only=True)
            daw.get_string_like_motion(directory,dataname,'/home/remote/hoomd-examples_0/testhoneycomb3-8-12-part1',0.81)
            #daw.get_displacment_field(directory,89,106)
        """
        file_txyz_stable = directory + 'txyz_stable.npy'
        txyz_stable = np.load(file_txyz_stable)
        dpa = pa.dynamic_points_analysis_2d(txyz_stable, mode='simu')
        file_list_sum_id_nb_stable = directory + 'list_sum_id_nb_stable.csv'
        import pandas as pd
        list_sum_id_nb_stable = pd.read_csv(file_list_sum_id_nb_stable)
        init_frame = 89
        end_frame = 106

        ids = dpa.plot_string_like_motion_rank(init_frame, end_frame, directory, data_name=data_name,  # 89,106
                                               nb_change=list_sum_id_nb_stable, bond_cut_off=6, trap_filename=trap_filename, trap_lcr=trap_lcr)  # '4302_9'
        # dpa.plot_string_like_motion(end_frame,end_frame,directory,data_name=data_name,ids=ids,
        #        bond_cut_off=6,trap_filename=trap_filename,trap_lcr=trap_lcr)

    def get_string_like_motion(self, directory, data_name=None, trap_filename=None, trap_lcr=None):
        R"""
        EXP:
            daw = dac.data_analysis_workflow()
            directory,dataname= daw.gsd_to_txyz(simu_index=4302,seed=9,io_only=True)
            daw.get_string_like_motion(directory,dataname,'/home/remote/hoomd-examples_0/testhoneycomb3-8-12-part1',0.81)
            #daw.get_displacment_field(directory,89,106)
        """
        file_txyz_stable = directory + 'txyz_stable.npy'
        txyz_stable = np.load(file_txyz_stable)
        dpa = pa.dynamic_points_analysis_2d(txyz_stable, mode='simu')
        init_frame = 1
        end_frame = 2000
        ids = dpa.plot_string_like_motion(init_frame, end_frame, directory, data_name=data_name,  # 89,106
                                          bond_cut_off=6, trap_filename=trap_filename, trap_lcr=trap_lcr)  # '4302_9'
        dpa.plot_string_like_motion(end_frame, end_frame, directory, data_name=data_name, ids=ids,
                                    bond_cut_off=6, trap_filename=trap_filename, trap_lcr=trap_lcr)

    def get_msd(self, pixel_to_um=3.0/32.0, um_to_sigma=1.0/2.0):
        txyz_npy_filename = self.path_to_results+'/'+'txyz_stable'
        traj = np.load(txyz_npy_filename)
        traj_um = traj*pixel_to_um  # pixel to um
        traj_sigma = traj_um*um_to_sigma  # um to sigma
        dpa = pa.dynamic_points_analysis_2d(traj_sigma, mode='exp')
        ts_id_dxy, average_1st_bond_length = dpa.compute_nearest_neighbor_displacements(
            unit='sigma')
        import pandas as pd
        ts_id_dxy = pd.read_csv('ts_id_dxy.csv')
        ts_id_dxy['z'] = 0
        dpa = pa.dynamic_points_analysis_2d(ts_id_dxy, mode='exp')
        txyz_ids_stable = dpa.compute_nearest_neighbor_displacements_stable(dpa.txyz_stable)
        dpa = pa.dynamic_points_analysis_2d(txyz_ids_stable, mode='exp')
        dpa.compute_atmsd_t_chips(0.9)
        time_log = self.path_to_results+'DefaultVideo_5.txt'
        time_log = np.loadtxt(time_log)
        dpa.plot_lindemann_msd(dpa.record_msd, average_1st_bond_length, time_log)
        print('average_1st_bond_length\n', average_1st_bond_length)

    def get_displacment_field(
            self, directory, frame_index_start=0, frame_index_end=-1, subplot=False):
        # import numpy
        # import points_analysis_2D as pa
        file_txyz_stable = directory + 'txyz_stable.npy'
        txyz_stable = np.load(file_txyz_stable)
        dpa = pa.dynamic_points_analysis_2d(txyz_stable, mode='simu')
        dpa.displacement_field_module()
        png_filename = directory+'displacement_field_xy'+'_' + \
            str(frame_index_start)+'_'+str(frame_index_end)+'.png'
        dpa.displacemnt_field.get_displacement_field_xy(
            frame_index_start, frame_index_end, True, png_filename)

    def get_a_frame(self, directory, frame_index):
        file_txyz_stable = directory + 'txyz_stable.npy'
        txyz_stable = np.load(file_txyz_stable)
        dpa = pa.dynamic_points_analysis_2d(txyz_stable, mode='simu')
        dpa.plot_a_frame_of_points(frame_index, directory+str(frame_index)+'.png')

    def get_displacement_1D(self, directory, data_name=None):  # ,id=0
        R"""
        example:
            import data_analysis_cycle as dac
            daw = dac.data_analysis_workflow()
            directory,data_name = daw.gsd_to_txyz(simu_index=4302,io_only=True)
            daw.get_displacement_1D(directory,data_name)
        """
        file_txyz_stable = directory + 'txyz_stable.npy'
        txyz_stable = np.load(file_txyz_stable)
        sz = np.shape(txyz_stable)
        # print(sz)
        """
        prefix = directory
        folder_name = ''
        isExists=os.path.exists(prefix+folder_name)
        if not isExists:
            os.makedirs(prefix+folder_name)
            prefix = prefix+folder_name
        """

        # dpa = pa.dynamic_points_analysis_2d(txyz_stable,mode='simu')
        # import points_analysis_2D as pa
        # dp = pa.dynamic_points_analysis_2d(txyz_stable)
        df = pa.dynamical_facilitation_module()
        for id in range(sz[1]):
            ax = df.plot_displacement_t(txyz_stable[:, id, :2])
            png_filename = directory+data_name+'_'+str(int(id))+'.png'
            ax.set_title(data_name+'_'+str(int(id)))
            plt.savefig(png_filename)
            plt.close()

    def get_displacement_1D_overlap(self, directory, data_name=None):  # ,id=0
        file_txyz_stable = directory + 'txyz_stable.npy'
        txyz_stable = np.load(file_txyz_stable)
        sz = np.shape(txyz_stable)
        df = pa.dynamical_facilitation_module()
        fig, ax = plt.subplots()
        for id in range(sz[1]):
            df.plot_displacement_t(txyz_stable[:, id, :2], ax)
        ax.set_xlabel('t (k steps)')
        ax.set_ylabel('$\delta$r ($\sigma$)')
        ax.set_title(data_name+'_'+str(int(id)))
        png_filename = directory+data_name+'_dr_abs'+'.png'
        plt.savefig(png_filename)
        plt.close()

    def scan_displacement_1D_overlap(self, directory, data_name=None):  # ,id=0
        file_txyz_stable = directory + 'txyz_stable.npy'
        txyz_stable = np.load(file_txyz_stable)
        sz = np.shape(txyz_stable)
        df = pa.dynamical_facilitation_module()
        fig, ax = plt.subplots()
        for id in range(sz[1]):
            df.plot_displacement_t(txyz_stable[:, id, :2], ax)
            df.plot_scan_displacement_t(txyz_stable[:, id, :2], ax)
        ax.set_xlabel('t (k steps)')
        ax.set_ylabel('$\delta$r ($\sigma$)')
        ax.set_title(data_name+'_'+str(int(id)))
        png_filename = directory+data_name+'_dr_abs'+'.png'
        plt.savefig(png_filename)
        plt.close()

    def get_displacement_1D_data(self, directory, data_name=None):
        file_txyz_stable = directory + 'txyz_stable.npy'
        txyz_stable = np.load(file_txyz_stable)
        sz = np.shape(txyz_stable)
        df = pa.dynamical_facilitation_module()

        for id in range(sz[1]):
            df.plot_displacement_t(txyz_stable[:, id, :2])


class transfer_txt_to_array:
    R"""
    introduction:
        to draw traps on bond_plot in exp,
        I have to transfer txt/tsf file into array

    example:
        import numpy as np
        import data_analysis_cycle as dac
        txt_file_name = '/home/remote/Downloads/5410-5419klt'
        ta = dac.transfer_txt_to_array()
        d1=ta.trans_txt_to_array(txt_file_name)
        txt_file_name = '/home/remote/Downloads/5420-5429klt'
        d2=ta.trans_txt_to_array(txt_file_name)
        data = np.concatenate((d1,d2))
        print(data[:,0])
        print(data[:,1])
        data[:,2] = data[:,2]*3/7.44
        print(data[:,2])
        ta.get_scatter('remote',data)
    """

    def __init__(self):
        pass

    def trans_txt_to_array(self, txt_file_name):
        R"""
        This function will save a txt file named 'start index - end index klt', which contains
        n rows of data
        |simu_index | HarmonicK | LinearCompressionRatio | kT |
        Psi3Global | Psi6Global | RandomSeed |

        CHECK: [v]
        """
        # txt_file_name = '/home/remote/Downloads/5410-5419klt'
        data = np.loadtxt(txt_file_name)
        return data

    def get_scatter(self, account, data):
        import matplotlib.pyplot as plt
        import numpy as np
        U_interaction = 300*np.exp(-0.25)
        prefix = '/home/'+account+'/Downloads/'
        postfix = '_pin_liquid_to_honeycomb_part_klt_2m.png'
        # print(data[:,4])
        plt.figure()
        # plot k VS T, Psi3 as value
        plt.scatter(data[:, 2], data[:, 1]*0.5, c=data[:, 4])  # LCR VS K, Psi3 as value
        # plt.show()
        plt.title('k VS T, Psi3 as value, Uparticle='+str(int(U_interaction)))
        plt.xlabel('Linear Compression Ratio (1)')
        plt.ylabel('U trap ($k_BT_m$)[Honeycomb part]')
        plt.colorbar()
        png_filename = prefix+'K_VS_T_Psi3_as_value'+postfix
        plt.savefig(png_filename)
        plt.close()

        plt.figure()
        # plot k VS T, Psi6 as value
        plt.scatter(data[:, 2], data[:, 1]*0.5, c=data[:, 5])  # LCR VS K, Psi6 as value
        # plt.show()
        plt.title('k VS T, Psi6 as value, Uparticle='+str(int(U_interaction)))
        plt.xlabel('Linear Compression Ratio (1)')
        plt.ylabel('U trap ($k_BT_m$)[Honeycomb part]')
        plt.colorbar()
        png_filename = prefix+'K_VS_T_Psi6_as_value'+postfix
        plt.savefig(png_filename)
        plt.close()
