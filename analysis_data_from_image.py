from turtle import width
import pandas as pd
import numpy as np
import points_analysis_2D_freud as pa

class data_analyzer_from_csv:
    def __init__(self,csv_filename='feature_single_frame.csv'):
        self.get_csv(csv_filename)
        self.data_preproceed()

    def get_csv(self,csv_filename):
        R"""
        csv_record:
                collumn name: y 	x 	mass 	size 	ecc 	signal 	raw_mass 	ep 	frame
                DataFrame([x, y, mass, size, ecc, signal, raw_mass])
                where "x, y" are appropriate to the dimensionality of the image,
                mass means total integrated brightness of the blob,
                size means the radius of gyration of its Gaussian-like profile,
                ecc is its eccentricity (0 is circular),
                and raw_mass is the total integrated brightness in raw_image.
        """
        self.csv_record = pd.read_csv(csv_filename)
    
    def data_preproceed(self):
        pix_size = 1024
        pix_to_um = 3/32
        self.csv_record['y'] = pix_size-1 - self.csv_record['y'] - (pix_size/2) #flip the y-axis, and centralize
        self.csv_record['x'] = self.csv_record['x'] - (pix_size/2)
        self.points = self.csv_record[['x','y']].values
        self.points[:]=self.points[:]* pix_to_um #transform from pixel to um
    
    def add_traps_to_points_square(self,init_locate=[-36.56,-30.75]):
        import workflow_analysis as wa
        at = wa.archimedean_tilings()
        at.generate_type_n(2,a=4.92)
        positions = set_positions_by_box_or_n_size(at,n_size=16)
        self.traps = positions[:,:2] + init_locate
        
    def draw_points_square(self):
        R"""
        import get_a_from_image as gai
        filename = 'DefaultVideo_12-00730.jpg'
        gi = gai.get_a_from_image(filename,save_data=True)

        import analysis_data_from_image as adi
        dac = adi.data_analyzer_from_csv()
        dac.add_traps_to_points_square()
        dac.draw_points_square()
        """
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        p2d = pa.static_points_analysis_2d(self.points,hide_figure=False)#
        
        p2d.get_first_minima_bond_length_distribution(png_filename='bond_hist.png')
        #draw bonds selected
        bpm = pa.bond_plot_module(fig,ax)#
        bpm.restrict_axis_property_relative(hide_axis=True)#'(um)',
        span=[0,30]
        bpm.restrict_axis_limitation(span,span)#[-10,10],[-10,10]
        ax.plot([],[],color='k',linewidth=10)#[3,8],[-9,-9]
        ax.text(x=5.5,y=-8.5,s='5 um',ha='center',va='baseline',fontdict=dict(fontsize=20))#ha:to x, va: to y,
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(p2d.bond_length,[2,p2d.bond_first_minima_left])
        
        bpm.plot_points_with_given_bonds(self.points,list_bond_index,bond_color='k',particle_color='k')#p2d.bond_length[:,:2].astype(int)
        bpm.plot_traps(self.traps)
        bpm.save_figure('image_with_traps.png')#plt.show() fig.savefig
        
    def add_traps_to_points_honeycomb_part(self,init_locate=[-11.8+1,-10.2-1]):
        import workflow_analysis as wa
        at = wa.archimedean_tilings()
        at.generate_type_n_part(3,a=4.45*0.71)#4.45
        positions = set_positions_by_box_or_n_size(at,n_size=[6,8])
        self.traps = positions[:,:2] + init_locate
        
    def draw_points_honeycomb_part(self):
        R"""
        import get_a_from_image as gai
        filename = 'DefaultImage_12.jpg'
        gi = gai.get_a_from_image(filename,save_data=True)#13,1000

        import analysis_data_from_image as adi
        dac = adi.data_analyzer_from_csv()
        dac.add_traps_to_points_honeycomb_part()
        dac.draw_points_honeycomb_part()
        """
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        p2d = pa.static_points_analysis_2d(self.points,hide_figure=False)#
        
        p2d.get_first_minima_bond_length_distribution(lattice_constant=3,png_filename='bond_hist.png')
        #draw bonds selected
        bpm = pa.bond_plot_module(fig,ax)#
        bpm.restrict_axis_property_relative(hide_axis=True)#'(um)', hide_axis=True
        
        bpm.restrict_axis_limitation([-5,25],[-7,20])#[-20,25],[-15,20]
        ax.plot([16,21],[-5,-5],color='k',linewidth=10)#[3,8],[-9,-9]
        ax.text(x=18.5,y=-4,s='5 um',ha='center',va='baseline',fontdict=dict(fontsize=20))#ha:to x, va: to y,
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(p2d.bond_length,[2.0,4.0])#2.5<3.6,5.5  #p2d.bond_first_minima_left
        
        bpm.plot_points_with_given_bonds(self.points,list_bond_index,bond_color='k',particle_color='k')#p2d.bond_length[:,:2].astype(int)
        bpm.plot_traps(self.traps)
        bpm.save_figure('image_with_traps.png')#plt.show() fig.savefig
    
    def add_traps_to_points_kagome_part(self,init_locate=[-40,-30]):#[-38.26,30.55]
        import workflow_analysis as wa
        at = wa.archimedean_tilings()
        at.generate_type_n_part(8,a=4.47*0.88)#4.47*0.88
        positions = set_positions_by_box_or_n_size(at,n_size=[6,8])
        positions[:,0] = positions[:,0]*1.04
        self.traps = positions[:,:2] + init_locate
        
        
    def draw_points_kagome_part(self):
        R"""
        import get_a_from_image as gai
        filename = 'DefaultVideo_7-02467.jpg'
        gi = gai.get_a_from_image(filename,save_data=True)#13,600

        import analysis_data_from_image as adi
        dac = adi.data_analyzer_from_csv()
        dac.add_traps_to_points_kagome_part()
        dac.draw_points_kagome_part()
        """
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        pt = np.zeros(np.shape(self.points))
        pt[:,0] = self.points[:,1]
        pt[:,1] = self.points[:,0]#exchange x and y axis
        self.points = pt
        p2d = pa.static_points_analysis_2d(pt,hide_figure=False)#
        
        p2d.get_first_minima_bond_length_distribution(lattice_constant=3,png_filename='bond_hist.png')
        #draw bonds selected
        bpm = pa.bond_plot_module(fig,ax)#
        bpm.restrict_axis_property_relative(hide_axis=True)#'(um)', hide_axis=True
        
        bpm.restrict_axis_limitation([-40,-10],[-20,10])#[-40,20],[-32,30]
        ax.plot([-21,-16],[-8,-8],color='k',linewidth=10)#[3,8],[-9,-9],,zorder=3
        ax.text(x=-18.5,y=-7,s='5 um',ha='center',va='baseline',fontdict=dict(fontsize=20))#ha:to x, va: to y,
        list_bond_index = bpm.get_bonds_with_conditional_bond_length(p2d.bond_length,[2.0,p2d.bond_first_minima_left])#  #
        
        bpm.plot_points_with_given_bonds(self.points,list_bond_index,bond_color='k',particle_color='k')#p2d.bond_length[:,:2].astype(int)
        bpm.plot_traps(self.traps)
        bpm.save_figure('image_with_traps.png')#plt.show() fig.savefig

def set_positions_by_box_or_n_size(particle_pointer,perturb=False,box=None,n_size=None):#checked right
        R"""
        introduction:
            code from set_new_gsd_file_2types_by_box_or_n_size 
            in symmetry_transformation_v4_3.system_parameters_generators
            
            Generate a gsd file containing an array of particles and an array of traps.
            caution: it only works when particle is type_n, trap is type_n_part, 
            for n_size of traps are copied directly from n_size of particles 
        parameters:
            lattice_pointer(substrate_pointer): 
                lp = workflow_analysis.archimedean_tilings(), after lp.generate_typex(),
                lp.a1,lp.a2,lp.a3 are lattice constant;
                lp.position record the positions of points in a single lattice
            box: [Lx,Ly] the size of box.
            n_size:
                [nx,ny] lattices the box records, calculated from the parameter box.
        example:
            particles = wa.archimedean_tilings()
            particles.generate_type8_part(a=3)
            n_size = [3,2]
            particle_points = particles.generate_lattices(n_size)

            traps = wa.archimedean_tilings()
            traps.generate_type10(a=0.8*3)# a*lcr !
            isg = pg.initial_state_generator()
            isg.set_new_gsd_file_2types(particles,n_size,particle_points,traps)

        example_show_result:
            import matplotlib.pyplot as plt
            import symmetry_transformation_v4_3.system_parameters_generators as pg
            isg = pg.initial_state_generator()
            isg.read_gsd_file()
            points = isg.particles.position
            import numpy as np
            ids = np.array(isg.snap.particles.typeid)
            list_p = ids == 0
            list_t = ids == 1

            isg.snap.particles.types
            fig,ax = plt.subplots()
            ax.scatter(points[list_p,0],points[list_p,1],color='k')#
            ax.scatter(points[list_t,0],points[list_t,1],color='r')#
            #ax.scatter(dula[:,0],dula[:,1],facecolors='none',edgecolors='k')#,marker = 'x'
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
            ids = np.array(isg.snap.particles.typeid)

        """#https://gsd.readthedocs.io/en/v3.2.0/python-module-gsd.hoomd.html#gsd.hoomd.open
        import points_analysis_2D_freud as pa

        #generate type 'particle' particles
        pp = particle_pointer
        vec = pp.a1+pp.a2+pp.a3#in case some lattices are parrallelograms
        if not box is None:
            n_size = [int(box[0]/vec[0])+1, int(box[1]/vec[1])+1]
        elif not n_size is None:
            pass
        else:
            print('error: no box or n_size input!')
            print('x')
        positions = pp.generate_lattices_not_centralized(n_size)
        #when the lattice is a parrallelogram, remove the points outside the box 
        pd = pa.static_points_analysis_2d(positions,hide_figure=False)
        #pd.cut_edge_of_positions_by_box(positions,box)#cut_edge_of_positions_by_xylimit(-0.5*bx[0],0.5*bx[0],-0.5*bx[1],0.5*bx[1])
        #positions = positions[pd.inbox_positions_bool]#edge_cut_positions_bool
        del pd
        if perturb:
            perturbation = np.random.random(positions.shape)*0.01
            perturbation[:,2] = 0
        else:
            perturbation = 0
        return positions

def match_particles_and_traps():
    #rotation
    #parallel motion
    pass