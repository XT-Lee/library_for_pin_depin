import numpy as np
import os
# import points_analysis_2D as pa
R"""
CLASS list:
    proceed_gsd_file:
    proceed_exp_file:
"""


class proceed_file_shell:
    R"""
    """

    def __init__(self):
        pass

    def create_folder(self, prefix='/home/remote/Downloads/', folder='folder_new'):
        R"""
        introduction:
            check and create a new folder.
        input:
            prefix: must end with '/'
            folder: must not end with '/'
        output:
            prefix_new: end with '/'
        """
        folder_name = prefix+folder
        # check if the folder exists
        isExists = os.path.exists(folder_name)
        if isExists:
            pass
        else:
            os.makedirs(folder_name)
        prefix_new = folder_name+'/'
        return prefix_new

    def create_prefix_results(self, account='remote', simu_index=0, seed=9):
        R"""
        input:
            account: (string);
            simu_index: (int)index;
            seed: (int)0-9
            io_only: just return results, not proceeding data.
        return:
            prefix_simu: (str)directory end with '/';
            str_simu_index: (str)'index_seed' for example.
        """
        str_simu_index = str(int(simu_index))+'_'+str(seed)
        prefix0 = '/home/'+account+'/Downloads/'
        prefix_simu = self.create_folder(prefix0, str_simu_index)

        return prefix_simu, str_simu_index


class proceed_gsd_file:
    R"""
    Introduction:
        the class is designed to preproceed the motion of 2D points(.gsd file), 
        to analyze dynamic properties.

    Parameters:
        filename_gsd: a path to gsd file;
        trajectory: trajectory data read from a hoomd gsd file;
        num_of_frames: the number of frames in the trajectory;

    Methods:
        open_gsd:
        read_a_frame:
        get_displacement_field: draw the displacements of final state from initial state.


    Examples:
    """

    def __init__(self, filename_gsd_seed=None, account="tplab", simu_index=None, seed=None):
        # load positions of particles
        if simu_index is None:
            if filename_gsd_seed is None:
                print("Error: input a correct path to gsd file please!\n")
            else:
                self.filename_gsd = filename_gsd_seed
                # filename_gsd="/home/tplab/hoomd-examples_0/trajectory_auto619.gsd"
                # self.data_gsd = np.loadtxt(self.filename_gsd)
                self.__open_gsd()

                """prefix_gsd = '/home/'+account+'/hoomd-examples_0/trajectory_auto'
                simu_index = filename_gsd_seed.strip(prefix_gsd)
                id=simu_index.index('_')
                self.simu_index = simu_index[0:id]"""
        else:
            self.simu_index = int(simu_index)
            if not seed is None:
                simu_index = str(int(simu_index))+'_'+str(int(seed))
            prefix_gsd = '/home/'+account+'/hoomd-examples_1'  # '/record_cairo/trajectory_auto'#
            # '/home/remote/xiaotian_file/link_to_HDD/hoomd-examples_0/trajectory_auto'
            # '/home/'+account+'/hoomd-examples_0/trajectory_auto'
            postfix_gsd = '.gsd'
            self.filename_gsd = prefix_gsd+str(simu_index)+postfix_gsd
            self.__open_gsd()

        self.box = self.trajectory._read_frame(-1).configuration.box

    def __open_gsd(self):
        import gsd.hoomd
        self.trajectory = gsd.hoomd.open(self.filename_gsd)  # open a gsd file
        self.num_of_frames = len(self.trajectory)

    def read_a_frame(self, frame_num, dimension=2):
        snap = self.trajectory._read_frame(frame_num)  # take a snapshot of the N-th frame
        positions = snap.particles.position[:, 0:dimension]  # just record [x,y] ignoring z
        # self.N = self.snap.particles.N
        return positions

    def read_the_typeid(self, dimension=2):
        snap = self.trajectory._read_frame(0)  # take a snapshot of the N-th frame
        positions = snap.particles.typeid[:, 0:dimension]  # just record [x,y] ignoring z
        # self.N = self.snap.particles.N
        return positions

    def get_trajectory_data(self, save_prefix=None, simu_index=None, seed=None):
        R"""
        introduction:
            transform gsd file into an array [Nframes,Nparticles,3],
            recording the trajectory of particles.
        input:
            gsd_file
        return:
            txyz [Nframes,Nparticles,3] or
            (npy file)[Nframes,Nparticles,3]
        example:
            import numpy as np
            import opertateOnMysql as osql
            tb_name = 'pin_hex_to_cairo_egct'
            #SimuIndex | HarmonicK | LinearCompressionRatio | CoordinationNum3Rate | CoordinationNum4Rate | CoordinationNum6Rate | PCairo     | RandomSeed
            cont = ' SimuIndex '
            list_index = osql.getDataFromMysql(table_name=tb_name,select_content=cont)
            list_index = np.array(list_index)
            import proceed_file as pf
            save_prefix = '/home/tplab/Downloads/'
            for index1 in list_index:
                pgf = pf.proceed_gsd_file(simu_index=index1[0])#,seed=9
                pgf.get_trajectory_data(save_prefix)
        """
        frame = 0
        snapi = self.trajectory._read_frame(frame)
        pos_list = np.zeros([self.num_of_frames, snapi.particles.N, 3]
                            )  # gsd_data.trajectory[0].particles.N,
        while frame < self.num_of_frames:
            pos_list[frame] = self.trajectory._read_frame(frame).particles.position
            # print(self.trajectory._read_frame(iframe).configuration.box)
            frame = frame + 1

        self.txyz = pos_list

        if not save_prefix is None:
            if simu_index is None:
                file_txyz_npy = save_prefix+'txyz'
            else:
                if seed is None:
                    file_txyz_npy = save_prefix+'txyz_'+str(simu_index)
                else:
                    file_txyz_npy = save_prefix+'txyz_'+str(simu_index)+'_'+str(seed)
            np.save(file=file_txyz_npy, arr=self.txyz)

    def get_trajectory_stable_data(self, save_prefix=None):
        R"""
        introduction:
            transform trajectory data from simulation with periodic boundary condition 
            into trajectories of which particles never move across boundary(box).
        return:
            txyz_stable: (N_frames,N_particles,3)

        """
        # dedrift?
        frames, particles, dimensions = self.txyz.shape
        if hasattr(self, 'box'):
            # print(locals())#local variable not of class
            self.dtxyz = self.txyz[1:, :, :] - self.txyz[:frames-1, :, :]
            # cross is true
            list_crossleft = self.dtxyz[:, :, 0] > 0.9*self.box[0]
            list_crossbottom = self.dtxyz[:, :, 1] > 0.9*self.box[1]
            list_crossright = self.dtxyz[:, :, 0] < -0.9*self.box[0]
            list_crosstop = self.dtxyz[:, :, 1] < -0.9*self.box[1]
            # mark all the frames where cross event occur as True
            list_crossx = np.logical_or(list_crossleft, list_crossright)
            list_crossy = np.logical_or(list_crossbottom, list_crosstop)
            list_cross = np.logical_or(list_crossx, list_crossy)
            # mark all the particles who have experienced cross event as True
            list_cross_true = np.array(list_cross[0, :])
            # list_cross_true = list_cross_true[0]#remove empty extra dimension
            # print(list_cross_true.shape)
            i = 0
            while i < particles:
                list_cross_true[i] = np.max(list_cross[:, i])
                i = i + 1
            list_stable_id = np.where(list_cross_true[:] == False)
            list_stable_id = list_stable_id[0]  # remove empty extra dimension
            # print(list_stable_id.shape)

            self.txyz_stable = self.txyz[:, list_stable_id, :]
            self.particles = list_stable_id.shape[0]

            if not save_prefix is None:
                file_txyz_npy = save_prefix+'txyz_stable'
                np.save(file=file_txyz_npy, arr=self.txyz_stable)

    # unchecked,,simu_index=None,seed=None
    def get_trajectory_data_with_traps(self, save_prefix=None):
        R"""
        introduction:
            transform gsd file into an array [Nframes,Nparticles,3],
            recording the trajectory of particles.
        input:
            gsd_file
        return:
            txyz [Nframes,Nparticles,3] or
            (npy file)[Nframes,Nparticles,3]
        example:
            import numpy as np
            import opertateOnMysql as osql
            tb_name = 'pin_hex_to_cairo_egct'
            #SimuIndex | HarmonicK | LinearCompressionRatio | CoordinationNum3Rate | CoordinationNum4Rate | CoordinationNum6Rate | PCairo     | RandomSeed
            cont = ' SimuIndex '
            list_index = osql.getDataFromMysql(table_name=tb_name,select_content=cont)
            list_index = np.array(list_index)
            import proceed_file as pf
            save_prefix = '/home/tplab/Downloads/'
            for index1 in list_index:
                pgf = pf.proceed_gsd_file(simu_index=index1[0])#,seed=9
                pgf.get_trajectory_data(save_prefix)
        """
        frame = 0
        snapi = self.trajectory._read_frame(frame)
        last_frame = snapi
        dimensions = 2
        xy_particles_traps = last_frame.particles.position[:, :dimensions]
        ids = np.array(last_frame.particles.typeid)
        list_p = ids == 0
        list_t = ids == 1
        n_particles = np.sum(np.array(list_p, dtype=int))
        # n_traps = np.sum(np.array(list_t,dtype=int))
        # save traps positions as txt
        traps = xy_particles_traps[list_t]
        file_traps_txt = save_prefix+'traps.txt'
        np.savetxt(file_traps_txt, traps)
        del traps, file_traps_txt
        # save box size as txt
        file_box_txt = save_prefix+'box.txt'
        np.savetxt(file_box_txt, self.box[:dimensions])
        del file_box_txt
        # save particles positions as npy
        pos_list = np.zeros([self.num_of_frames, n_particles, dimensions]
                            )  # gsd_data.trajectory[0].particles.N,
        for frame in range(self.num_of_frames):
            pos_list[frame] = self.trajectory._read_frame(
                frame).particles.position[list_p, :dimensions]
            # print(self.trajectory._read_frame(iframe).configuration.box)
        self.txyz = pos_list
        file_txyz_npy = save_prefix+'txyz'
        np.save(file_txyz_npy, self.txyz)

    def get_trajectory_stable_data_with_traps(self, save_prefix=None):
        self.get_trajectory_stable_data(save_prefix)

    def get_trajectory_data_large(self, save_prefix=None):
        R"""
        introduction:
            transform gsd file into an array [Nframes,Nparticles,3],
            recording the trajectory of particles.
        input:
            gsd_file
        return:
            txyz [Nframes,Nparticles,3] or
            (npy file)[Nframes,Nparticles,3]
        """
        frame = 0
        snapi = self.trajectory._read_frame(frame)
        # gsd_data.trajectory[0].particles.N,
        pos_list = np.zeros([int(self.num_of_frames/10+1), snapi.particles.N, 3])
        i = 0
        while frame < self.num_of_frames:
            pos_list[i] = self.trajectory._read_frame(frame).particles.position
            # print(self.trajectory._read_frame(iframe).configuration.box)
            frame = frame + 10
            i = i+1

        self.txyz = pos_list

        if not save_prefix is None:
            file_txyz_npy = save_prefix+'txyz'
            np.save(file=file_txyz_npy, arr=self.txyz)

    def get_extended_positions(self, frame_num=2000, dimension=2):
        R"""
        Introduction:
            extend an array of positions( seeing them as a 1*1 square) into 3*3 squares, 
            and save them as a 9 times larger array.
        Example:
            import proceed_file as pf
            gsd_data = pf.proceed_gsd_file(None,'remote',4302,9)
            array = gsd_data.get_extended_positions()
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(array[:,0],array[:,1])
            plt.axis('equal')
            plt.show()
        Situation:
            checked right.
        """
        snap = self.trajectory._read_frame(frame_num)  # take a snapshot of the N-th frame
        positions = snap.particles.position[:, 0:dimension]  # just record [x,y] ignoring z
        box = snap.configuration.box
        extended_positions = self.get_extended_positions_from_points(box, positions, dimension=2)
        self.box = box
        return extended_positions

    def get_extended_positions_from_points(self, box, positions, dimension=2):
        extended_positions = np.zeros((9*len(positions), dimension))
        pos_left = positions + np.array((-box[0], 0))
        pos_right = positions + np.array((box[0], 0))
        pos_top = positions + np.array((0, box[1]))
        pos_bottom = positions + np.array((0, -box[1]))
        pos_top_left = positions + np.array((-box[0], box[1]))
        pos_top_right = positions + np.array((box[0], box[1]))
        pos_bottom_left = positions + np.array((-box[0], -box[1]))
        pos_bottom_right = positions + np.array((box[0], -box[1]))
        extended_positions = np.concatenate((pos_top_left, pos_top, pos_top_right,
                                            pos_left, positions, pos_right,
                                            pos_bottom_left, pos_bottom, pos_bottom_right), axis=0)
        return extended_positions  # positions#

    def draw_box_extended_positions(self, box, ex_positions):
        """
        fig,ax = plt.subplots

        """


class data_type_transformer:
    def __init__(self):
        pass

    def array_to_csv(self, txyz_stable, csv_prefix):
        R"""
        import numpy as np
        import proceed_file as pf
        #
        prefix = '/home/tplab/xiaotian_file/lxt_code_py/4302_9/'
        filename_txyz_stable = prefix+'txyz_stable.npy'
        csv_prefix = '/home/tplab/Downloads/4302_9/'
        txyz_stable = np.load(filename_txyz_stable)
        trans = pf.data_type_transformer()
        trans.array_to_csv(txyz_stable,csv_prefix)
        """
        import numpy as np
        import pandas as pd
        columns_name = ["time_step", "particle_id", "x", "y", "z"]
        # get frame-wise
        # list_framewise
        frames, particles, dimensions = np.shape(txyz_stable)
        t_id1_xyz_empty = np.zeros((frames, 2+dimensions))  # 2+dimensions
        frames_array = np.linspace(0, frames-1, frames)
        list_particles = range(particles)
        # organize the format from npy to csv
        for id in list_particles:
            t_id1_xyz_empty[:, 0] = frames_array
            t_id1_xyz_empty[:, 1] = id
            t_id1_xyz_empty[:, 2:] = txyz_stable[:, id, :]
            t_id1_xyz_pd = pd.DataFrame(t_id1_xyz_empty)
            t_id1_xyz_pd.columns = columns_name
            if id == 0:
                t_id_xyz_pd = t_id1_xyz_pd
                print(t_id_xyz_pd.tail())
                # print(ts_id_dxy.tail())
            else:  # why 0-2000 rows with id = 1 too?
                t_id_xyz_pd = pd.concat([t_id_xyz_pd, t_id1_xyz_pd])
                print(t_id_xyz_pd.tail())

        pd.DataFrame.to_csv(t_id_xyz_pd, csv_prefix+'t_id_xyz_4302_9.csv')

    def array_to_xyz(self, positions, filename):
        R"""
        INPUT: array, n rows of [x,y]
        output: .xyz files, 

        .xyz format:
        <Nparticles>
        <comment line>
        <element_name> <x> <y> <z>

        example:
        3
        this is a table of elements and xyz positions
        C   1   2   3
        N   1   4   2
        O   5   3   8

        example:
        import proceed_file as pf
        dtt = pf.data_type_transformer()
        import numpy as np
        filename_array = "/home/remote/Downloads/index4298_6"
        xyz = np.loadtxt(filename_array)
        filename_xyz = "/home/remote/Downloads/index4298_6.xyz"
        dtt.array_to_xyz(xyz,filename_xyz)
        """
        num_particles = len(positions)

        with open(filename, 'w') as xyz_file:
            xyz_file.write(f"{num_particles}\n")
            xyz_file.write("Generated by Python\n")

            for position in positions:
                x, y, z = position
                xyz_file.write(f"X {x}\t{y}\t{z}\n")


class proceed_exp_file:
    R"""
    see particle_tracking.py to get trajectories of particles from a video.
    """
    pass


class merge_two_csvs:
    def __init__(self, csv_filename1, csv_filename2, csv_filename_merged=None):
        import pandas as pd
        csv1 = pd.read_csv(csv_filename1)
        csv2 = pd.read_csv(csv_filename2)
        csv_merged = pd.concat([csv1, csv2])
        if not csv_filename_merged is None:
            pd.DataFrame.to_csv(csv_merged, csv_filename_merged)
        # return csv_merged
        self.csv_merged = csv_merged
