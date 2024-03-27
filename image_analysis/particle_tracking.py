# particle tracking
import matplotlib
import matplotlib.pyplot as plt  # , matplotlib.image as imread
import numpy as np
import pandas as pd
# from pandas import DataFrame, Series  # for convenience
import os
import trackpy as tp


class save_points_from_exp:
    R"""
    Introduction:
        This class provides a routine to follow, which shows the steps to recognize images,
        to transform videos into positions & trajectories.

        Personally, my raw images are stored in
        path_to_raw_data = 'home/tplab/xiaotian_file/from_hard_disc_20221209/
        Xiaotian_file_20200514/1各期工作汇总/20190909项目(油相晶体)（ing）/20221129（测过冷度；24h不漏）'
    exp:
        import particle_tracking as  pa
        spf = pa.save_points_from_exp()
        path_to_folder = '/home/remote/xiaotian_file/data/20230113'
        video_name = 'DefaultVideo_8'
        spf.set_worksapce(path_to_folder,video_name)
        Diameter, minmass = spf.check_first_last_frame()

        spf.recognize_all_frames(Diameter, minmass)
        spf.get_stable_trajectory()
    """

    def __init__(self):
        pass

    # step 1: set worksapce
    def set_worksapce(self, path_to_folder, video_name):
        R"""
        input:
            path_to_folder = '/home/remote/xiaotian_file/data/20230113'
            video_name = 'DefaultVideo_1'
        """
        # caution: ensure that path_to_images stores only images(.jpg)!
        self.path_to_images = path_to_folder+'/'+video_name
        # for path_to_results, any type of file is ok
        self.path_to_results = path_to_folder+'/'+video_name+'_result'

    # step2: find suitable parameters to recognize particles
    # check if (Diameter,minmass) suit for the first and the last frames
    def check_first_last_frame(self):  # ,filename_image_first=None,filename_image_last=None
        R"""
        return:
            Diameter: minimum diameter(pixel) of particle
            minmass: minimum lightness of particle
        """
        import particle_tracking as ptt
        import os

        for root, dirs, files in os.walk(self.path_to_images):
            files.sort()
            filename_image_first = self.path_to_images+'/'+files[0]
            filename_image_last = self.path_to_images+'/'+files[-1]  # num_iamges = len(files)
            break
        ana = ptt.particle_track()
        ana.single_frame_particle_tracking(filename_image_first, calibration=True)
        ana.single_frame_particle_tracking(filename_image_last, calibration=True)
        print(ana.Diameter, ana.minmass)
        return ana.Diameter, ana.minmass

    # step3: recognize particles on all the images
    # 'txy_um.csv' records particle positions for each frame
    def recognize_all_frames(self, Diameter, minmass):  # ,pixel_to_um=3.0/32.0,save_txy=False
        import particle_tracking as ptt
        ana = ptt.particle_track()
        feature_filename = self.path_to_results+'/'+'feature.csv'
        ana.folder_frames_particle_tracking(
            self.path_to_images, Diameter, minmass, feature_filename)
        """
        if save_txy:
            txy_um_filename = self.path_to_results+'/'+'txy_um.csv'
            ana.get_positions_from_features(feature_filename,pixel_to_um,txy_um_filename)
        """

    # step4: get stable trajectories of particlesg
    # 'txyz_stable' is a list of trajectories contain only those particles
    # which are always in field of view in video
    def get_stable_trajectory(self, pixel_to_um=3.0/32.0):
        import particle_tracking as ptt
        ana = ptt.particle_track()
        feature_filename = self.path_to_results+'/'+'feature.csv'
        track_filename = self.path_to_results+'/'+'track_memory0.csv'
        ana.link_features_to_trajectories(
            feature_filename=feature_filename, track_filename=track_filename)
        txyz_filename = self.path_to_results+'/'+'txyz.csv'
        ana.save_trajectories(track_filename, txyz_filename)
        txyz_npy_filename = self.path_to_results+'/'+'txyz_stable'
        ana.select_stable_trajectory(
            txyz_filename, txyz_npy_filename=txyz_npy_filename, pixel_to_um=pixel_to_um)

    def get_trap_positions(
            self, tsf_filename, trap_positions=None, xy_adjust=[0, 0],
            scale_adjust=1.0, rotate_adjust=0):
        R"""
        input:
            tsf_filename,
            xy_adjust=[0,0],
            scale_adjust=1.0,
            rotate=0
        output:
            trap_positions: n rows of array [x,y]
            trap_filename: self.path_to_results+'/'+'trap_positions'
        """
        if trap_positions is None:
            tsf_filename = self.path_to_results+'/'+tsf_filename
            trap_positions = np.loadtxt(tsf_filename)
            trap_positions = trap_positions[:, :2]

        theta = rotate_adjust/180.0*np.pi
        rotate = np.zeros((2, 2))
        # rotate operator is on the right side,
        # so rotational direction is inverted.
        # to let rotation match intuition,
        # I inverted rotation operator too.
        cc = np.cos(theta)
        ss = np.sin(theta)
        rotate[0, 0] = cc
        rotate[1, 1] = cc
        rotate[1, 0] = -ss
        rotate[0, 1] = ss
        trap_positions = np.matmul(trap_positions, rotate)

        trap_positions = trap_positions*scale_adjust

        trap_positions[:, 0] = trap_positions[:, 0]+xy_adjust[0]
        trap_positions[:, 1] = trap_positions[:, 1]+xy_adjust[1]

        trap_filename = 'trap_positions'  # self.path_to_results+'/'+'trap_positions'
        # np.savetxt(trap_filename, trap_positions)
        return trap_positions, trap_filename

    def get_point_positions(
            self, tsf_filename, point_positions=None, xy_adjust=[0, 0],
            scale_adjust=1.0, rotate_adjust=0):
        R"""
        input:
            tsf_filename,
            xy_adjust=[0,0],
            scale_adjust=1.0,
            rotate=0
        output:
            point_positions: n rows of array [x,y]
            point_filename: self.path_to_results+'/'+'point_positions'
        """
        if point_positions is None:
            tsf_filename = self.path_to_results+'/'+tsf_filename
            point_positions = np.loadtxt(tsf_filename)
            point_positions = point_positions[:, :2]

        theta = rotate_adjust/180.0*np.pi
        rotate = np.zeros((2, 2))
        # rotate operator is on the right side,
        # so rotational direction is inverted.
        # to let rotation match intuition,
        # I inverted rotation operator too.
        cc = np.cos(theta)
        ss = np.sin(theta)
        rotate[0, 0] = cc
        rotate[1, 1] = cc
        rotate[1, 0] = -ss
        rotate[0, 1] = ss
        point_positions = np.matmul(point_positions, rotate)

        point_positions[:, 0] = point_positions[:, 0]+xy_adjust[0]
        point_positions[:, 1] = point_positions[:, 1]+xy_adjust[1]
        point_positions = point_positions*scale_adjust

        point_filename = 'point_positions'  # self.path_to_results+'/'+'point_positions'
        # np.savetxt(point_filename, point_positions)
        return point_positions, point_filename


class particle_track:
    R"""
        algorithm from Eric Weeks
            https://physics.emory.edu/faculty/weeks/idl/
        library from github.soft-matter.trackpy
            http://soft-matter.github.io/trackpy/v0.5.0/tutorial/walkthrough.html

        f collumn name: y 	x 	mass 	size 	ecc 	signal 	raw_mass 	ep 	frame
        """

    def __init__(self):
        pass

    def single_frame_particle_tracking(
            self, filename, D=11, minmass=500, calibration=False, axis_limit=None):
        R"""
        instruction:
            D should be the the diameter of dark ring of the smallest particle in image.
            minmass should be the maximum mass of the darker cluster.
            separation should be 0.9*D if dimer particles occur.

            f collumn name: y 	x 	mass 	size 	ecc 	signal 	raw_mass 	ep 	frame
                DataFrame([x, y, mass, size, ecc, signal, raw_mass])
                where "x, y" are appropriate to the dimensionality of the image,
                mass means total integrated brightness of the blob,
                size means the radius of gyration of its Gaussian-like profile,
                ecc is its eccentricity (0 is circular),
                and raw_mass is the total integrated brightness in raw_image.

            f0: the RGB image
            f1: the gray image


            axis_limit should be a list of 4 ints [xmin,xmax,ymin,ymax] 
                to limit the region(pixel as unit)of image to recognize particles.

        example:
            import particle_tracking as pt
            filename= '/home/tplab/xiaotian_file/data/20220924/DefaultVideo_5.tif'
            track = pt.particle_track()
            track.single_particle_tracking(filename,i=12,D=31,minmass=5000)
            track.multiple_particle_tracking(filename) 
        """
        f0 = plt.imread(filename)
        if axis_limit is None:
            f1 = f0[:, :, 0]  # 3 channel to 1 channel
        else:
            lm = axis_limit
            f1 = f0[lm[0]:lm[1], lm[2]:lm[3], 0]
        sz = np.shape(f1)
        # print(sz)
        # plt.imshow(frames[0])
        # diameter of particles should includes dark ring!
        self.f = tp.locate(f1, D, minmass=minmass, invert=False,
                           separation=0.9*D)  # diameter must be odd in pixels
        # f is feature.
        # collumn name: y 	x 	mass 	size 	ecc 	signal 	raw_mass 	ep 	frame
        if calibration:
            # print(self.f.head())#print(self.f['x'])
            plt.figure(1)
            plt.scatter(self.f['mass'], self.f['size'])
            plt.xlabel('mass')
            plt.ylabel('size')

            plt.figure(2)
            plt.scatter(self.f['mass'], self.f['ecc'])
            plt.xlabel('mass')
            plt.ylabel('ecc')
            # plt.show()

            # selected_feature
            # sf=self.f[self.f['mass'] > 6000 ]
            # sf2=sf[sf['mass'] < 3000]

            plt.figure(3)
            tp.annotate(self.f, f1)  # ,imshow_style=
            # plt.imsave(png_filename)#savefig(png_filename)   miss array
            # plt.close()
        """
        tp.annotate(f, frames[i])
        print(f.head())
        #plt.imshow(frames[0])

        plt.figure()
        plt.hist(f['mass'], bins=20)#here mass is the brightness of a particle
        plt.show()
        # Optionally, label the axes.
        #ax.set(xlabel='mass', ylabel='count');
        """
        self.xy = np.array(self.f[['x', 'y']])  # unit: pixel
        self.Diameter = D
        self.minmass = minmass
        if axis_limit is None:
            self.image = f0
        else:
            lm = axis_limit
            self.image = f0[lm[0]:lm[1], lm[2]:lm[3], :]

        return f1

    def save_feature(self, csv_filename='feature_single_frame.csv'):
        pd.DataFrame.to_csv(self.f, csv_filename)

    def folder_frames_particle_tracking(
            self, dir_path, Diameter=19, minmass=1000, feature_filename='feature.csv'):
        R"""
        introduction:
            input:a folder name which contains all the images to read.
            output: 'feature.csv' saves all the features
        parameters:
            feature has 8 columns [y 	x 	mass 	size 	ecc 	signal 	raw_mass 	ep]
            features has 9 columns [y 	x 	mass 	size 	ecc 	signal 	raw_mass 	ep  frame]
            auto_save: it cost several minutes to recognize particles, hence the parameter controls the program
            to save the results proceeded.

        """
        for root, dirs, files in os.walk(dir_path):
            files.sort()
            frame = 0
            auto_save = 0
            for file in files:
                if file.endswith('.jpg'):
                    full_file = dir_path+'/'+file
                    f0 = plt.imread(full_file)
                    f0 = f0[:, :, 0]
                    image_size = np.shape(f0)
                    feature = tp.locate(f0, Diameter, minmass, invert=False)  # ,separation=0.9*D
                    # feature has 8 columns [y 	x 	mass 	size 	ecc 	signal 	raw_mass 	ep]
                    feature['frame'] = frame

                    if not 'features' in locals():
                        features = feature
                    else:
                        features = pd.concat([features, feature])

                    if frame-auto_save > 999:
                        pd.DataFrame.to_csv(features, 'feature'+str(int(frame))+'.csv')
                        print(str(int(frame))+' frames have been saved.')
                        auto_save = frame

                    frame = frame + 1
                else:
                    print(file+' is not a jpg file!\n')
            # invert y-axis
            # the direction of y-axis for image is different from which for coordination
            features['y'] = image_size[0]-1 - features['y']

            pd.DataFrame.to_csv(features, feature_filename)
            break

    def get_positions_from_features(self, feature_filename, pixel_to_um,
                                    txy_um_filename='txy_um.csv'):
        R"""
        save 'txy_um.csv': ['frame','x','y'] um as unit.
        """
        features = pd.read_csv(feature_filename)
        txy_um = features[['frame', 'x', 'y']]
        txy_um[['x', 'y']].values = txy_um[['x', 'y']].values*pixel_to_um
        pd.DataFrame.to_csv(txy_um, txy_um_filename)

    def link_features_to_trajectories(
            self, search_range=int(19 / 2),
            feature_filename=None, track_filename='track_memory0.csv'):
        R"""
        introduction:
            input: 
                search_range: pxiels of particle radius is recommended.
                feature_filename of a file generated by the library trackpy.
            output: 'track.csv' has 10 columns [y 	x 	mass 	size 	ecc 	signal 	raw_mass 	ep 	frame 	particle].
        Parameters:
        """
        if not 'features' in locals():  # hasattr()
            features = pd.read_csv(feature_filename)
        track = tp.link(features, search_range)  # ,memory=3
        pd.DataFrame.to_csv(track, track_filename)

    def save_trajectories(self, track_filename, txyz_filename='txyz.csv'):  # temp
        R"""
        introduction:
            input: 'track.csv' has 10 columns [y 	x 	mass 	size 	ecc 	signal 	raw_mass 	ep 	frame 	particle].
            output: 'txyz.csv' contains ['frame','particle','x','y','z']
        parameters:
        """
        track = pd.read_csv(track_filename)  # 'track.csv'
        txyz = track[['frame', 'particle', 'x', 'y']]
        txyz['z'] = 0.0
        print(txyz.shape)
        print(txyz.head())
        pd.DataFrame.to_csv(txyz, txyz_filename)

    def select_stable_trajectory(self, tpxyz_filename='txyz.csv', tpxyz=None, txyz_npy_filename='txyz_stable', pixel_to_um=3.0/32.0):  # ,account='remote'
        R"""
        introduction:
            input: each one is ok. 
                filename 'txyz.csv' contains ['frame','particle','x','y','z']
                or pandas.dataframe tpxyz contains ['frame','particle','x','y','z'] 
            return: 
                txyz_ids_stable  is (Nframes,Nparticles,xyz) 3-dimension array,
                which contains ['particle'] which are always in the field of view in video.

        parameters:

        example:
            plt.figure()
            tp.plot_traj(txyz,mpp='um');
        """
        if tpxyz is None:
            txyz = pd.read_csv(tpxyz_filename)
        else:
            txyz = tpxyz

        frame_num = txyz['frame'].values.max()+1

        list_particle_id = np.unique(txyz['particle'].values)
        list_particle_id_stable = np.ones((list_particle_id.max(),), np.int32)
        list_particle_id_stable = -list_particle_id_stable

        count_particle_id_stable = 0
        for particle_id in list_particle_id:
            # draw trajectories only for those particles exist in all the frames.
            txyz_ith = txyz[txyz['particle'] == particle_id]
            if txyz_ith.shape[0] == frame_num:
                list_particle_id_stable[count_particle_id_stable] = particle_id
                count_particle_id_stable = count_particle_id_stable + 1
                # print(txyz_ith.shape)
                # plt.plot(txyz_ith['x'],txyz_ith['y'])
        self.list_particle_id_stable = list_particle_id_stable[:count_particle_id_stable]
        # png_filename = '/home/'+self.account+'/Downloads/'+'traj_id_'+str(particle_id)+'.png'
        """
        #save to file 'txyz_stable.csv'
        for particle_id in self.list_particle_id_stable:
            txyz_id_stable = txyz[txyz['particle'] == particle_id]
            if not 'txyz_ids_stable' in locals():
                txyz_ids_stable = txyz_id_stable
            else:
                txyz_ids_stable = pd.concat([txyz_ids_stable,txyz_id_stable])
        pd.DataFrame.to_csv(txyz_ids_stable,'txyz_stable.csv')  
        """
        # save to numpy array, [frames,particles,xyz].
        txyz_ids_stable = np.zeros((frame_num, count_particle_id_stable, 3))
        particle_count = 0
        for particle_id in self.list_particle_id_stable:
            txyz_id_stable = txyz[txyz['particle'] == particle_id]
            id_xyz = txyz_id_stable[['x', 'y', 'z']].values
            txyz_ids_stable[:, particle_count-1, :] = id_xyz
            particle_count = particle_count + 1

        txyz_ids_stable[:] = txyz_ids_stable[:]*pixel_to_um
        np.save(txyz_npy_filename, txyz_ids_stable)  # it is a 3d array which can not save as table.

    def plot_trajectory_per_particle(self, tpxyz_filename='txyz.csv'):
        # hide plot
        # Backend agg is non-interactive backend. Turning interactive mode off. 'QtAgg' is interactive mode
        matplotlib.use(backend="agg")
        if not hasattr(self, 'list_particle_id_stable'):
            self.select_stable_trajectory(tpxyz_filename=tpxyz_filename)
        txyz = pd.read_csv(tpxyz_filename)
        for particle_id in self.list_particle_id_stable:
            txyz_ith = txyz[txyz['particle'] == particle_id]
            plt.figure()
            plt.plot(txyz_ith['x'], txyz_ith['y'])
            png_filename = 'traj_stable_'+str(int(particle_id))+'_.png'
            plt.savefig(png_filename)
            plt.close()

    def plot_trajectory_all_particle(self, tpxyz_filename='txyz.csv'):
        # hide plot
        # Backend agg is non-interactive backend. Turning interactive mode off. 'QtAgg' is interactive mode
        matplotlib.use(backend="agg")
        if not hasattr(self, 'list_particle_id_stable'):
            self.select_stable_trajectory(tpxyz_filename=tpxyz_filename)
        txyz = pd.read_csv(tpxyz_filename)
        plt.figure()
        for particle_id in self.list_particle_id_stable:
            txyz_ith = txyz[txyz['particle'] == particle_id]
            plt.plot(txyz_ith['x'], txyz_ith['y'])
        png_filename = 'traj_stable.png'
        plt.savefig(png_filename)
        plt.close()

    def single_stack_particle_tracking(self, filename, i=7, D=35, minmass=7000):
        R"""
        from github.soft-matter.trackpy
        http://soft-matter.github.io/trackpy/v0.5.0/tutorial/walkthrough.html

        example:
            import particle_tracking as pt
            filename= '/home/tplab/xiaotian_file/data/20220924/DefaultVideo_5.tif'
            track = pt.particle_track()
            track.single_particle_tracking(filename,i=12,D=31,minmass=5000)
            track.multiple_particle_tracking(filename) 
        """

        # change the following to %matplotlib notebook for interactive plotting
        # %matplotlib inline

        # Optionally, tweak styles.
        # mpl.rc('figure',  figsize=(10, 5))
        # mpl.rc('image', cmap='gray')

        # read a tiff file
        """
        frames = pims.open(filename)
        f0= frames[i]
        sz=np.shape(f0)
        #print(sz)
        #plt.imshow(frames[0])
        #diameter of particles should includes dark edge! 35
        f = tp.locate(frames[i], D, minmass= minmass,invert=False)#diameter must be odd in pixels
        """

        # f means feature.
        # collumn name: y 	x 	mass 	size 	ecc 	signal 	raw_mass 	ep 	frame
        """
        tp.annotate(f, frames[i])
        print(f.head())
        #plt.imshow(frames[0])

        plt.figure()
        plt.hist(f['mass'], bins=20)#here mass is the brightness of a particle
        plt.show()
        # Optionally, label the axes.
        #ax.set(xlabel='mass', ylabel='count');
        """

        self.Diameter = D
        self.minmass = minmass

    def multiple_particle_tracking(self, filename):
        """
        frames = pims.open(filename)
        sz=np.shape(frames)
        f=[]
        #for i in range(sz[0]):
        for i in range(3):
            fi = tp.locate(frames[i], 35, minmass= 2000,invert=False)
            f.append(fi)
        """
        """
        f = tp.batch(frames[:100],35,minmass=5000)# self.Diameter self.minmass what the fuck corrupted tag list
        t = tp.link(f, 5, memory=3)
        t1 = tp.filter_stubs(t)
        # Compare the number of particles in the unfiltered and filtered data.
        print('Before:', t['particle'].nunique())
        print('After:', t1['particle'].nunique())
        plt.figure()
        tp.plot_traj(t);
        """
        pass

    def track_bright_field(self, filename):
        R"""
        import particle_tracking as pt
        filename= '/home/tplab/xiaotian_file/data/20220924/DefaultVideo_5.tif'
        track = pt.particle_track()
        track.track_bright_field(filename)
        """
        """
        frames = pims.open(filename)
        micron_per_pixel = 3.0/32.0
        feature_diameter = 2.1 # um
        radius = int(np.round(feature_diameter/2.0/micron_per_pixel))
        if radius % 2 == 0:
            radius += 1
        print('Using a radius of {:d} px'.format(radius))

        # we use a slightly larger radius
        f_locate = tp.locate(frames[0], radius+2, minmass=2000)
        tp.annotate(f_locate, frames[0], plot_style={'markersize': radius});

        plt.figure()
        plt.hist(f_locate['mass'], bins=20)#here mass is the brightness of a particle
        plt.show()

        """
        pass
