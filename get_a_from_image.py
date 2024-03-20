import particle_tracking as pt
import points_analysis_2D_freud as pa


class get_a_from_image:
    R"""
    introduction:
        input image name,output length of constant lattice(unit = 1 um)

    example:
        import get_a_from_image as gf
        filename='/home/remote/xiaotian_file/20210417/DefaultImage.jpg'
        frame=gf.get_a_from_image(filename)
    """

    def __init__(self, filename, silent=False, save_data=False):
        frame = pt.particle_track()
        # parameters remain undefined,,calibration=True
        frame.single_frame_particle_tracking(filename, 13, 1000)  # , calibration=True
        points = frame.xy
        points[:] = points[:]*3/32  # transform unit from pixel to um
        points[:, 1] = -points[:, 1]  # invert y coordination.

        # particle density, averaged bond length
        result = pa.static_points_analysis_2d(points)  # ,hide_figure=False
        if silent:
            png_filename1 = None
            png_filename2 = None
        else:
            png_filename1 = filename + '_bond_hist.png'
            png_filename2 = filename + '_bond_plot_1st_minima.png'

        lc = 2.0  # lattice constant, particle diameter(2 um) as default
        # here lattice_constant is just used to normalize figure, hence set 2.0 is ok
        result.get_first_minima_bond_length_distribution(
            lattice_constant=lc, png_filename=png_filename1, hist_cutoff=5)
        print('recognized bond length: ' + str(result.bond_length_median * lc) + '+-' +
              str(result.bond_length_std * lc) + ' um')

        import matplotlib.pyplot as plt
        plt.close('all')
        fig, ax = plt.subplots()
        # draw bonds selected
        bpm = pa.bond_plot_module(fig, ax)
        bpm.restrict_axis_property_relative('(um)')
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(
            result.bond_length, [2, result.bond_first_minima_left])

        # p2d.bond_length[:,:2].astype(int)
        bpm.plot_points_with_given_bonds(
            points, list_bond_index, bond_color='k', particle_color='k')
        plt.savefig(png_filename2)

        if save_data:
            frame.save_feature()


class show_points_select:
    R"""
    """

    def __init__(self) -> None:
        pass

    def tune_points(self, points_pix):
        # parameters remain undefined,,calibration=True
        points = points_pix[:]-(1024-1)/2  # centralize the points
        points[:] = points[:]*3/32  # transform unit from pixel to um
        points[:, 1] = -points[:, 1]  # invert y coordination.
        return points

    def show_points(self, points, filename, traps=None, silent=False):
        # particle density, averaged bond length
        result = pa.static_points_analysis_2d(points)  # ,hide_figure=False
        if silent:
            png_filename1 = None
            png_filename2 = None
        else:
            png_filename1 = filename + '_bond_hist.png'
            png_filename2 = filename + '_bond_plot_1st_minima.png'

        lc = 2.0  # lattice constant, particle diameter(2 um) as default
        # here lattice_constant is just used to normalize figure, hence set 2.0 is ok
        result.get_first_minima_bond_length_distribution(
            lattice_constant=lc, png_filename=png_filename1, hist_cutoff=5)
        print('recognized bond length: ' + str(result.bond_length_median * lc) + '+-' +
              str(result.bond_length_std * lc) + ' um')

        import matplotlib.pyplot as plt
        plt.close('all')
        fig, ax = plt.subplots()
        # draw bonds selected
        bpm = pa.bond_plot_module(fig, ax)
        bpm.restrict_axis_property_relative('(um)')
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(
            result.bond_length, [2, result.bond_first_minima_left])

        # p2d.bond_length[:,:2].astype(int)
        bpm.plot_points_with_given_bonds(
            points, list_bond_index, bond_color='k', particle_color='k')
        if not traps is None:
            bpm.plot_traps(traps, mode='array')
        plt.savefig(png_filename2)

    def show_points_finetune(self, points, filename, traps=None, silent=False):
        import numpy as np
        # particle density, averaged bond length
        particle_size = 20  # 50
        circle_color = np.array([100, 143, 255])/255.0  # 'orange'
        bond_width = 2  # 3
        result = pa.static_points_analysis_2d(points)  # ,hide_figure=False
        if silent:
            png_filename1 = None
            png_filename2 = None
        else:
            png_filename1 = filename + '_bond_hist.png'
            png_filename2 = filename + '_bond_plot_1st_minima.png'

        lc = 2.0  # lattice constant, particle diameter(2 um) as default
        # here lattice_constant is just used to normalize figure, hence set 2.0 is ok
        result.get_first_minima_bond_length_distribution(
            lattice_constant=lc, png_filename=png_filename1, hist_cutoff=5)
        print('recognized bond length: ' + str(result.bond_length_median * lc) + '+-' +
              str(result.bond_length_std * lc) + ' um')

        import matplotlib.pyplot as plt
        plt.close('all')
        fig, ax = plt.subplots()
        # draw bonds selected
        bpm = pa.bond_plot_module(fig, ax)
        # bpm.restrict_axis_property_relative(x_unit='(um)')
        bpm.restrict_axis_property_relative(hide_axis=True)
        # bpm.plot_scale_bar(-10, -9)
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(
            result.bond_length, [2, 6.5])  # result.bond_first_minima_left 7.0
        print(result.bond_first_minima_left)

        # p2d.bond_length[:,:2].astype(int)
        bpm.plot_points_with_given_bonds(
            points, list_bond_index, bond_color='k', particle_color='k',
            particle_size=particle_size, bond_width=bond_width)
        bpm.restrict_axis_limitation([-40, 15], [-35, 20])

        if not traps is None:
            bpm.plot_traps(traps)  # mode='array'
        bpm.save_figure(png_filename2)  # plt.savefig(png_filename2)
