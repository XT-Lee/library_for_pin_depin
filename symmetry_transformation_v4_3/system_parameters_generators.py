import numpy as np
import proceed_file as pf#gsd.hoomd
import gsd.hoomd
#import workflow_analysis as wa

class initial_state_generator:
    def __init__(self):
        R"""
        import symmetry_transformation_v4_3.system_parameters_generators as spg
        isg = spg.initial_state_generator()
        isg.get_gsd_sample()
        """
        pass

    def get_gsd_sample(self,simu_index=4302,seed=9,account='remote'):
        prefix = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_0/"
        str_index=str(int(simu_index))+'_'+str(seed)
        file_gsd = prefix+'trajectory_auto'+str_index+'.gsd'#+'_'+str(seed)
        self.gsd_data = pf.proceed_gsd_file(filename_gsd_seed=file_gsd,account=account)
        self.snap = self.gsd_data.trajectory._read_frame(0)
        
        self.angles = self.snap.angles
        self.bonds = self.snap.bonds
        self.box = self.snap.configuration.box#[LX,LY,LZ,0,0,0]
        self.constraints = self.snap.constraints
        self.dihedrals = self.snap.dihedrals
        self.pairs = self.snap.pairs
        self.particles = self.snap.particles
        print(self.snap.particles.typeid)#[0]*N_particles, the value of the i-th element means the type('A'or'B') of the i-th particle.
        #print(self.snap.particles.types)#['A'] mark the particles
        
        #print(self.snap.particles.orientation)#for spherical particle, list([1,0,0,0])
        #print(self.snap.particles.types)#types of particles,list('A','B')
    
    def read_gsd_file(self,file_gsd):
        #prefix = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_0/"
        #file_gsd = prefix+'particle_and_trap.gsd'#'lattice.gsd'#'trajectory_auto'+str_index+'.gsd'#+'_'+str(seed)
        self.trajectory = gsd.hoomd.open(file_gsd)
        self.snap = self.trajectory.read_frame(0)# _read_frame
        self.particles = self.snap.particles
        #print('ok')
       
    def set_new_gsd_file_2types_by_box_or_n_size(self,particle_pointer,trap_pointer,output_gsd_filename,perturb=False,box=None,n_size=None):#checked right
        R"""
        introduction:
            Generate a gsd file containing an array of particles and an array of traps.
            caution: it only works when particle is type_n, trap is type_n_part, 
            for n_size of traps are copied directly from n_size of particles 
        parameters:
            lattice_pointer(substrate_pointer): 
                lp = workflow_analysis.archimedean_tilings(), after lp.generate_typex(),
                lp.a1,lp.a2,lp.a3 are lattice constant;
                lp.position record the positions of points in a single lattice
            box: [Lx,Ly] the roughly expected size of box.
            n_size:
                [nx,ny] lattices the box records, calculated from the parameter box.
            snap.configuration.box: precise box defined by n_size and vec
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
        snap = gsd.hoomd.Frame()#.Frame()

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
        positions = pp.generate_lattices(n_size)
        snap.configuration.box = [vec[0]*n_size[0],vec[1]*n_size[1],0,0,0,0]
        #generate type 'trap' particles
        sp = trap_pointer
        vec_trap = sp.a1+sp.a2+sp.a3
        box = snap.configuration.box
        trap_size = [int(box[0]/vec_trap[0])+1, int(box[1]/vec_trap[1])+1]
        trap_positions = sp.generate_lattices(trap_size)
        #when the lattice is a parrallelogram, remove the points outside the box 
        pd = pa.static_points_analysis_2d(positions,hide_figure=False)
        pd.cut_edge_of_positions_by_box(positions,box)#cut_edge_of_positions_by_xylimit(-0.5*bx[0],0.5*bx[0],-0.5*bx[1],0.5*bx[1])
        positions = positions[pd.inbox_positions_bool]#edge_cut_positions_bool
        del pd
        pd = pa.static_points_analysis_2d(trap_positions,hide_figure=False)
        pd.cut_edge_of_positions_by_box(trap_positions,box)#cut_edge_of_positions_by_xylimit(-0.5*bx[0],0.5*bx[0],-0.5*bx[1],0.5*bx[1])
        trap_positions = trap_positions[pd.inbox_positions_bool]#edge_cut_positions_bool
        self.position_box = pd.position_box
        del pd
        #fill the snap with data
        nps = positions.shape[0]
        nps_trap = trap_positions.shape[0]
        snap.particles.N = nps + nps_trap
        position_total = np.concatenate((positions,trap_positions),axis=0) 
        #print(max(position_total.any()),min(position_total.any()))
        perturb_amp = 0.01
        if perturb:#warning: the particle positions after perturbation are not checked inbox!
            perturbation = np.random.random(position_total.shape)*perturb_amp
            perturbation[:,2] = 0
            snap.particles.position = (position_total + perturbation)*0.999
            #(1-2*perturb_amp/min(snap.configuration.box[:2])) 
            #precisely equalled bond will let delaunay disfunction! 
            # or let particle position as NaN. so fix it!
        else:
            snap.particles.position = position_total 
        

        snap.particles.orientation = [(1,0,0,0)]*(nps+nps_trap)
        list_typeid = [0]*nps+[1]*nps_trap
        snap.particles.typeid = list_typeid
        snap.particles.types = ['particle','trap']
        # wb: overwrite, xb: not overwrite, ab: append after a existing file
        with gsd.hoomd.open(name=output_gsd_filename, mode='wb') as f:
            f.append(snap)
        
    def set_new_gsd_file(self,particle_pointer,output_gsd_filename,perturb=False,box=None,n_size=None):
        R"""
            lattice_pointer: 
                lp = workflow_analysis.archimedean_tilings(), after lp.generate_typex(),
                lp.a1,lp.a2,lp.a3 are lattice constant;
                lp.position record the positions of points in a single lattice
            n_size:
                [nx,ny] lattices the box records.

        """#https://gsd.readthedocs.io/en/v3.2.0/python-module-gsd.hoomd.html#gsd.hoomd.open
        import points_analysis_2D_freud as pa
        snap = gsd.hoomd.Frame()#.Frame()

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
        positions = pp.generate_lattices(n_size)
        snap.configuration.box = [vec[0]*n_size[0],vec[1]*n_size[1],0,0,0,0]
        #generate type 'trap' particles
        box = snap.configuration.box
        
        #when the lattice is a parrallelogram, remove the points outside the box 
        pd = pa.static_points_analysis_2d(positions,hide_figure=False)
        pd.cut_edge_of_positions_by_box(positions,box)#cut_edge_of_positions_by_xylimit(-0.5*bx[0],0.5*bx[0],-0.5*bx[1],0.5*bx[1])
        positions = positions[pd.inbox_positions_bool]#edge_cut_positions_bool
        del pd
        
        #fill the snap with data
        nps = positions.shape[0]
        
        snap.particles.N = nps
        position_total = positions
        #print(max(position_total.any()),min(position_total.any()))
        perturb_amp = 0.01
        if perturb:#warning: the particle positions after perturbation are not checked inbox!
            perturbation = np.random.random(position_total.shape)*perturb_amp
            perturbation[:,2] = 0
            snap.particles.position = (position_total + perturbation)*0.999
            #(1-2*perturb_amp/min(snap.configuration.box[:2])) 
            #precisely equalled bond will let delaunay disfunction! 
            # or let particle position as NaN. so fix it!
        else:
            snap.particles.position = position_total 

        snap.particles.orientation = [(1,0,0,0)]*nps
        list_typeid = [0]*nps
        snap.particles.typeid = list_typeid
        snap.particles.types = ['particle']
        # w: overwrite, x: not overwrite, a: append after a existing file
        with gsd.hoomd.open(name=output_gsd_filename, mode='wb') as f:
            f.append(snap)

    def generate_yukawa(self):
        #frame = gsd.hoomd.Frame()
        """frame.particles.N = N_particles
        frame.particles.position = position
        frame.particles.orientation = orientation
        frame.particles.typeid = [0] * N_particles
        frame.particles.types = ['octahedron']
        frame.configuration.box = [L, L, L, 0, 0, 0]
        with gsd.hoomd.open(name='lattice.gsd', mode='x') as f:
            f.append(frame)
        """
        pass
        

class potential_generator:
    def __init__(self):
        pass

    def generate_yukawa(self,a,kappa,r_min,r_max,width=1000):
        R"""
            u = a*exp(-kappa*r)/r
            f = -a*(kappa*r^-1 + r^-2)*exp(-kappa*r)
        """
        if r_min>0:
            r = np.linspace(r_min,r_max,width)
            list_index = -kappa*r
            u_unit = np.divide(np.exp(list_index),r)
            u = a*u_unit

            interval = (r_max-r_min)/float(width)
            f = np.zeros((width,))
            f[1:-1] = (u[2:] - u[:-2])/(2*interval)
            f[0] = (u[1] - u[0]) / interval
            f[-1 ]= (u[-1] - u[-2]) / interval

            list_u_and_f = np.zeros((width,2))
            list_u_and_f[:,0] = u
            list_u_and_f[:,1] = f
        else:
            print('Error: r_min should be no less than zero!')
            print('Error!')

        return u,f
    
    def save_txt(self,filename,table):
        np.savetxt(filename,table)