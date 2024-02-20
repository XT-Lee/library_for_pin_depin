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
    def __init__(self,filename,silent=False,save_data=False):
        frame=pt.particle_track()
        frame.single_frame_particle_tracking(filename,13,600)#parameters remain undefined,,calibration=True
        points=frame.xy
        points[:]=points[:]*3/32# transform unit from pixel to um
        points[:,1]=-points[:,1]#invert y coordination.
        
        #particle density, averaged bond length
        result = pa.static_points_analysis_2d(points)#,hide_figure=False
        if silent:
            png_filename1= None
            png_filename2 = None
        else:
            png_filename1= filename +'_bond_hist.png'
            png_filename2 = filename +'_bond_plot_1st_minima.png'

        lc = 2.0#lattice constant, particle diameter(2 um) as default
        result.get_first_minima_bond_length_distribution(lattice_constant=lc,png_filename=png_filename1,hist_cutoff=5)#here lattice_constant is just used to normalize figure, hence set 2.0 is ok
        print('recognized bond length: '+str(result.bond_length_median*lc)+'+-'+str(result.bond_length_std*lc)+' um')
        
        import matplotlib.pyplot as plt
        plt.close('all')
        fig,ax = plt.subplots()
        #draw bonds selected
        bpm = pa.bond_plot_module(fig,ax)#
        bpm.restrict_axis_property_relative('(um)')
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(result.bond_length,[2,result.bond_first_minima_left])
        
        bpm.plot_points_with_given_bonds(points,list_bond_index,bond_color='k',particle_color='k')#p2d.bond_length[:,:2].astype(int)
        plt.savefig(png_filename2)

        if save_data:
            frame.save_feature()