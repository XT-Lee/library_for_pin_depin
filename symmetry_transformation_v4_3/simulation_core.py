import hoomd
import gsd.hoomd
# import proceed_file as pf
import numpy as np


class simulation_core_traps:
    def __init__(self, simu_index, seed):
        R"""
        Introduction:
            the class is designed to simulate the evolution of 
        a group of particles interacting with each other, 
        under a series of traps with gaussian potential.
            hence the gsd files, used as initial state here, containing
        two types of particles ['particle','trap'], are recomended 
        to be generated by "system_parameters_generators.
        initial_state_generator.set_new_gsd_file_2types".
            all the codes are checked to run in hoomd v4.3.0 correctly

        Refernce:
            Glotzerlab:
            https://github.com/glotzerlab/gsd
            GSD:
            https://gsd.readthedocs.io/en/v3.2.0/
        """
        self.set_simulation_parameters()
        self.set_file_parameters(simu_index, seed)

    def set_simulation_parameters(self):
        self.mode = 'cpu'  # 'cpu','gpu'

        self.gauss_epsilon = -300
        self.gauss_r_cut = 2.0  # 1.0 equivalent to harmonic rcut=1
        self.gauss_sigma = 0.6  # 1.0 equivalent to harmonic rcut=1
        self.yukawa_epsilon = 300
        self.yukawa_kappa = 0.25
        self.yukawa_r_cut = 15.0  # 5.0
        self.kT = 1.0
        self.dt = 0.002
        self.total_steps = 2e6+1
        self.width = 1  # 1,10 the size of cell, larger for type4 lattice
        self.wca_r_cut = np.power(2, 1/6)  # r=2^(1/6) is the minima of LJ potential
        self.wca_sigma = 1.0
        self.wca_epsilon = 1.0

    def set_file_parameters(self, simu_index, seed):
        self.prefix_read = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/"
        self.prefix_write = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/"
        self.seed = seed
        self.simu_index = simu_index
        self.input_file_gsd = self.prefix_read+'particle_and_trap.gsd'
        self.output_file_gsd = self.prefix_write+'trajectory_auto' + \
            str(int(self.simu_index))+'_'+str(int(self.seed))+'.gsd'
        self.snap_period = 1000

    def __set_init_state_parameters(self):
        """
            https://hoomd-blue.readthedocs.io/en/v4.3.0/howto/molecular.html
        """
        self.trajectory = gsd.hoomd.open(self.input_file_gsd)  # open a gsd file
        self.num_of_frames = len(self.trajectory)
        # self.gsd_data = pf.proceed_gsd_file(filename_gsd_seed=self.input_file_gsd,account=account)
        self.init_snap = self.trajectory._read_frame(0)

    def operate_simulation_langevin(self):
        # initialize
        if self.mode == 'cpu':
            device = hoomd.device.CPU()
        elif self.mode == 'gpu':
            device = hoomd.device.GPU()
        sim = hoomd.Simulation(device=device, seed=self.seed)
        # sim.state
        self.__set_init_state_parameters()
        # sim.create_state_from_snapshot(self.init_snap)
        sim.create_state_from_gsd(filename=self.input_file_gsd)

        # set interaction
        # Apply the gaussian traps on the particles.
        gauss_cell = hoomd.md.nlist.Cell(self.width)  # default_r_cut=15.0 ? 'particle','trap'
        gauss = hoomd.md.pair.Gaussian(
            nlist=gauss_cell, default_r_cut=self.gauss_r_cut, mode='xplor')
        gauss.params[('particle', 'trap')] = dict(
            epsilon=self.gauss_epsilon, sigma=self.gauss_sigma)
        gauss.params[('particle', 'particle')] = dict(epsilon=0.0, sigma=1.0)
        gauss.params[('trap', 'trap')] = dict(epsilon=0.0, sigma=1.0)
        gauss.r_cut[('particle', 'trap')] = 1.0
        gauss.r_cut[('particle', 'particle')] = 0.0
        gauss.r_cut[('trap', 'trap')] = 0.0

        # Apply the yukawa interaction on the particles.
        yukawa_cell = hoomd.md.nlist.Cell(self.width)  # ('trap',)
        yukawa = hoomd.md.pair.Yukawa(
            nlist=yukawa_cell, default_r_cut=self.yukawa_r_cut, mode='xplor')
        yukawa.params[('particle', 'particle')] = dict(
            epsilon=self.yukawa_epsilon, kappa=self.yukawa_kappa)
        yukawa.params[('particle', 'trap')] = dict(epsilon=0.0, kappa=0.0)
        yukawa.params[('trap', 'trap')] = dict(epsilon=0.0, kappa=0.0)

        R"""
        https://hoomd-blue.readthedocs.io/en/v4.3.0/howto/prevent-particles-from-moving.html#how-to-prevent-particles-from-moving
        MD simulations
        Omit the stationary particles from the filter (or filters) 
        that you provide to your integration method (or methods) 
        to prevent them from moving in MD simulations. For example:

        simulation = hoomd.util.make_example_simulation()

        # Select mobile particles with a filter.
        stationary_particles = hoomd.filter.Tags([0])
        mobile_particles = hoomd.filter.SetDifference(hoomd.filter.All(),
                                                    stationary_particles)

        # Integrate the equations of motion of the mobile particles.
        langevin = hoomd.md.methods.Langevin(filter=mobile_particles, kT=1.5)
        simulation.operations.integrator = hoomd.md.Integrator(dt=0.001,
                                                            methods=[langevin])

        simulation.run(100)
        """
        # find the mobile particles
        # list_particles = range(self.snap.particles.N)
        list_ids = self.init_snap.particles.typeid  # range(21)#np.linspace(0,20,21,dtype=int)
        list_Particles_index = list_ids == 0
        tags = list(np.where(list_Particles_index))
        # set system
        # tag n is the n-th particle, must be a list or iterator
        mobile_particles = hoomd.filter.Tags(tags)
        # stationary_particles = hoomd.filter.Tags([1])
        # mobile_particles = hoomd.filter.SetDifference(hoomd.filter.All(),stationary_particles)
        langevin = hoomd.md.methods.Langevin(
            filter=mobile_particles, kT=self.kT,)  # hoomd.filter.All()
        integrator = hoomd.md.Integrator(dt=self.dt,
                                         methods=[langevin],
                                         forces=[gauss, yukawa])

        sim.operations.integrator = integrator

        # https://gsd.readthedocs.io/en/v3.2.0/python-module-gsd.hoomd.html#gsd.hoomd.open
        gsd_writer = hoomd.write.GSD(filename=self.output_file_gsd,
                                     trigger=hoomd.trigger.Periodic(self.snap_period),
                                     mode='xb')  # deprecated: 'xb'
        sim.operations.writers.append(gsd_writer)
        sim.run(self.total_steps)

    def operate_simulation_brownian(self):
        # initialize
        if self.mode == 'cpu':
            device = hoomd.device.CPU()
        elif self.mode == 'gpu':
            device = hoomd.device.GPU()
        sim = hoomd.Simulation(device=device, seed=self.seed)
        # sim.state
        self.__set_init_state_parameters()
        sim.create_state_from_snapshot(self.init_snap)
        # sim.create_state_from_gsd(filename='molecular.gsd')

        # set interaction
        # Apply the gaussian traps on the particles.
        gauss_cell = hoomd.md.nlist.Cell(1)  # default_r_cut=15.0 ? 'particle','trap'
        gauss = hoomd.md.pair.Gaussian(nlist=gauss_cell, default_r_cut=self.gauss_r_cut)
        gauss.params[('particle', 'trap')] = dict(
            epsilon=self.gauss_epsilon, sigma=self.gauss_sigma)
        gauss.params[('particle', 'particle')] = dict(epsilon=0.0, sigma=1.0)
        gauss.params[('trap', 'trap')] = dict(epsilon=0.0, sigma=1.0)
        gauss.r_cut[('particle', 'trap')] = 1.0
        gauss.r_cut[('particle', 'particle')] = 0.0
        gauss.r_cut[('trap', 'trap')] = 0.0

        # Apply the yukawa interaction on the particles.
        yukawa_cell = hoomd.md.nlist.Cell(1)  # ('trap',)
        yukawa = hoomd.md.pair.Yukawa(nlist=yukawa_cell, default_r_cut=self.yukawa_r_cut)
        yukawa.params[('particle', 'particle')] = dict(
            epsilon=self.yukawa_epsilon, kappa=self.yukawa_kappa)
        yukawa.params[('particle', 'trap')] = dict(epsilon=0.0, kappa=0.0)
        yukawa.params[('trap', 'trap')] = dict(epsilon=0.0, kappa=0.0)

        R"""
        https://hoomd-blue.readthedocs.io/en/v4.3.0/howto/prevent-particles-from-moving.html#how-to-prevent-particles-from-moving
        MD simulations
        Omit the stationary particles from the filter (or filters) 
        that you provide to your integration method (or methods) 
        to prevent them from moving in MD simulations. For example:

        simulation = hoomd.util.make_example_simulation()

        # Select mobile particles with a filter.
        stationary_particles = hoomd.filter.Tags([0])
        mobile_particles = hoomd.filter.SetDifference(hoomd.filter.All(),
                                                    stationary_particles)

        # Integrate the equations of motion of the mobile particles.
        langevin = hoomd.md.methods.Langevin(filter=mobile_particles, kT=1.5)
        simulation.operations.integrator = hoomd.md.Integrator(dt=0.001,
                                                            methods=[langevin])

        simulation.run(100)
        """
        # find the mobile particles
        # list_particles = range(self.snap.particles.N)
        list_ids = self.init_snap.particles.typeid  # range(21)#np.linspace(0,20,21,dtype=int)
        list_Particles_index = list_ids == 0
        tags = list(np.where(list_Particles_index))
        # set system
        # tag n is the n-th particle, must be a list or iterator
        mobile_particles = hoomd.filter.Tags(tags)
        # stationary_particles = hoomd.filter.Tags([1])
        # mobile_particles = hoomd.filter.SetDifference(hoomd.filter.All(),stationary_particles)
        brownian = hoomd.md.methods.Brownian(filter=mobile_particles, kT=self.kT)
        integrator = hoomd.md.Integrator(dt=self.dt,
                                         methods=[brownian],
                                         forces=[gauss, yukawa])

        sim.operations.integrator = integrator

        # https://gsd.readthedocs.io/en/v3.2.0/python-module-gsd.hoomd.html#gsd.hoomd.open
        gsd_writer = hoomd.write.GSD(filename=self.output_file_gsd,
                                     trigger=hoomd.trigger.Periodic(self.snap_period),
                                     mode='xb')  # deprecated: 'xb'
        sim.operations.writers.append(gsd_writer)
        sim.run(self.total_steps)

    def operate_simulation_langevin_wca_yukawa(self):
        # initialize
        if self.mode == 'cpu':
            device = hoomd.device.CPU()
        elif self.mode == 'gpu':
            device = hoomd.device.GPU()
        sim = hoomd.Simulation(device=device, seed=self.seed)
        # sim.state
        self.__set_init_state_parameters()
        sim.create_state_from_snapshot(self.init_snap)
        # sim.create_state_from_gsd(filename='molecular.gsd')

        # set interaction
        # Apply the gaussian traps on the particles.
        gauss_cell = hoomd.md.nlist.Cell(self.width)  # default_r_cut=15.0 ? 'particle','trap'
        gauss = hoomd.md.pair.Gaussian(
            nlist=gauss_cell, default_r_cut=self.gauss_r_cut, mode='xplor')  # see exampleplot. xplor
        gauss.params[('particle', 'trap')] = dict(
            epsilon=self.gauss_epsilon, sigma=self.gauss_sigma)
        gauss.params[('particle', 'particle')] = dict(epsilon=0.0, sigma=1.0)
        gauss.params[('trap', 'trap')] = dict(epsilon=0.0, sigma=1.0)
        gauss.r_cut[('particle', 'trap')] = 1.0
        gauss.r_cut[('particle', 'particle')] = 0.0
        gauss.r_cut[('trap', 'trap')] = 0.0

        # Apply the yukawa interaction on the particles.
        yukawa_cell = hoomd.md.nlist.Cell(self.width)  # ('trap',)
        yukawa = hoomd.md.pair.Yukawa(
            nlist=yukawa_cell, default_r_cut=self.yukawa_r_cut, mode='xplor')
        yukawa.params[('particle', 'particle')] = dict(
            epsilon=self.yukawa_epsilon, kappa=self.yukawa_kappa)
        yukawa.params[('particle', 'trap')] = dict(epsilon=0.0, kappa=0.0)
        yukawa.params[('trap', 'trap')] = dict(epsilon=0.0, kappa=0.0)

        # Apply the WCA interaction on the particles.
        wca_cell = hoomd.md.nlist.Cell(self.width)  # ('trap',)
        wca = hoomd.md.pair.LJ(nlist=wca_cell, default_r_cut=self.wca_r_cut, mode='shift')
        wca.params[('particle', 'particle')] = dict(sigma=self.wca_sigma,
                                                    epsilon=self.wca_epsilon)  # self.wca_sigma = 1.0,self.wca_epsilon = 1.0
        wca.params[('particle', 'trap')] = dict(sigma=1.0, epsilon=0.0)
        wca.params[('trap', 'trap')] = dict(sigma=1.0, epsilon=0.0)

        R"""
        https://hoomd-blue.readthedocs.io/en/v4.3.0/howto/prevent-particles-from-moving.html#how-to-prevent-particles-from-moving
        MD simulations
        Omit the stationary particles from the filter (or filters) 
        that you provide to your integration method (or methods) 
        to prevent them from moving in MD simulations. For example:

        simulation = hoomd.util.make_example_simulation()

        # Select mobile particles with a filter.
        stationary_particles = hoomd.filter.Tags([0])
        mobile_particles = hoomd.filter.SetDifference(hoomd.filter.All(),
                                                    stationary_particles)

        # Integrate the equations of motion of the mobile particles.
        langevin = hoomd.md.methods.Langevin(filter=mobile_particles, kT=1.5)
        simulation.operations.integrator = hoomd.md.Integrator(dt=0.001,
                                                            methods=[langevin])

        simulation.run(100)
        """
        # find the mobile particles
        # list_particles = range(self.snap.particles.N)
        list_ids = self.init_snap.particles.typeid  # range(21)#np.linspace(0,20,21,dtype=int)
        list_Particles_index = list_ids == 0
        tags = list(np.where(list_Particles_index))
        # set system
        # tag n is the n-th particle, must be a list or iterator
        mobile_particles = hoomd.filter.Tags(tags)
        # stationary_particles = hoomd.filter.Tags([1])
        # mobile_particles = hoomd.filter.SetDifference(hoomd.filter.All(),stationary_particles)
        langevin = hoomd.md.methods.Langevin(
            filter=mobile_particles, kT=self.kT,)  # hoomd.filter.All()
        integrator = hoomd.md.Integrator(dt=self.dt,
                                         methods=[langevin],
                                         forces=[gauss, yukawa, wca])

        sim.operations.integrator = integrator

        # https://gsd.readthedocs.io/en/v3.2.0/python-module-gsd.hoomd.html#gsd.hoomd.open
        gsd_writer = hoomd.write.GSD(filename=self.output_file_gsd,
                                     trigger=hoomd.trigger.Periodic(self.snap_period),
                                     mode='xb')  # deprecated: 'xb'
        sim.operations.writers.append(gsd_writer)
        sim.run(self.total_steps)


class simulation_core:
    def __init__(self, simu_index, seed):
        R"""
        Introduction:
            the class is designed to simulate the evolution of 
        a group of particles interacting with each other, 
        under a series of traps with gaussian potential.
            hence the gsd files, used as initial state here, containing
        two types of particles ['particle','trap'], are recomended 
        to be generated by "system_parameters_generators.
        initial_state_generator.set_new_gsd_file_2types".
            all the codes are checked to run in hoomd v4.3.0 correctly

        Refernce:
            Glotzerlab:
            https://github.com/glotzerlab/gsd
            GSD:
            https://gsd.readthedocs.io/en/v3.2.0/
        """
        self.set_simulation_parameters()
        self.set_file_parameters(simu_index, seed)

    def set_simulation_parameters(self):
        self.mode = 'cpu'  # 'cpu','gpu'

        self.yukawa_epsilon = 300
        self.yukawa_kappa = 0.25
        self.yukawa_r_cut = 15.0  # 5.0
        self.kT = 1.0
        self.dt = 0.002
        self.total_steps = 2e6+1
        self.width = 1  # 1,10 the size of cell, larger for type4 lattice
        self.wca_r_cut = np.power(2, 1/6)  # r=2^(1/6) is the minima of LJ potential
        self.wca_sigma = 1.0
        self.wca_epsilon = 1.0
        self.opp_epsilon = 300
        self.opp_r_cut = 15.0
        self.opp_c1 = 100

    def set_file_parameters(self, simu_index, seed):
        self.prefix_read = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/"
        self.prefix_write = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/"
        self.seed = seed
        self.simu_index = simu_index
        self.input_file_gsd = self.prefix_read+'particles.gsd'
        self.output_file_gsd = self.prefix_write+'trajectory_auto' + \
            str(int(self.simu_index))+'_'+str(int(self.seed))+'.gsd'
        self.snap_period = 1000

    def __set_init_state_parameters(self):
        """
            https://hoomd-blue.readthedocs.io/en/v4.3.0/howto/molecular.html
        """
        self.trajectory = gsd.hoomd.open(self.input_file_gsd)  # open a gsd file
        self.num_of_frames = len(self.trajectory)
        # self.gsd_data = pf.proceed_gsd_file(filename_gsd_seed=self.input_file_gsd,account=account)
        self.init_snap = self.trajectory._read_frame(0)

    def operate_simulation_langevin_wca_yukawa(self):
        # initialize
        if self.mode == 'cpu':
            device = hoomd.device.CPU()
        elif self.mode == 'gpu':
            device = hoomd.device.GPU()
        sim = hoomd.Simulation(device=device, seed=self.seed)
        # sim.state
        self.__set_init_state_parameters()
        sim.create_state_from_snapshot(self.init_snap)
        # sim.create_state_from_gsd(filename='molecular.gsd')

        # set interaction
        # Apply the yukawa interaction on the particles.
        yukawa_cell = hoomd.md.nlist.Cell(self.width)  # ('trap',)
        yukawa = hoomd.md.pair.Yukawa(
            nlist=yukawa_cell, default_r_cut=self.yukawa_r_cut, mode='xplor')
        yukawa.params[('particle', 'particle')] = dict(
            epsilon=self.yukawa_epsilon, kappa=self.yukawa_kappa)

        # Apply the WCA interaction on the particles.
        wca_cell = hoomd.md.nlist.Cell(self.width)  # ('trap',)
        wca = hoomd.md.pair.LJ(nlist=wca_cell, default_r_cut=self.wca_r_cut, mode='shift')
        wca.params[('particle', 'particle')] = dict(sigma=self.wca_sigma,
                                                    epsilon=self.wca_epsilon)  # self.wca_sigma = 1.0,self.wca_epsilon = 1.0

        """#find the mobile particles
        #list_particles = range(self.snap.particles.N)
        list_ids = self.init_snap.particles.typeid#range(21)#np.linspace(0,20,21,dtype=int)
        list_Particles_index = list_ids == 0
        tags = list(np.where(list_Particles_index))
        #set system
        mobile_particles= hoomd.filter.Tags(tags)#tag n is the n-th particle, must be a list or iterator 
        #stationary_particles = hoomd.filter.Tags([1])
        #mobile_particles = hoomd.filter.SetDifference(hoomd.filter.All(),stationary_particles)"""
        langevin = hoomd.md.methods.Langevin(
            filter=hoomd.filter.All(),
            kT=self.kT,)  # mobile_particles
        integrator = hoomd.md.Integrator(dt=self.dt,
                                         methods=[langevin],
                                         forces=[yukawa, wca])

        sim.operations.integrator = integrator

        # https://gsd.readthedocs.io/en/v3.2.0/python-module-gsd.hoomd.html#gsd.hoomd.open
        gsd_writer = hoomd.write.GSD(filename=self.output_file_gsd,
                                     trigger=hoomd.trigger.Periodic(self.snap_period),
                                     mode='xb')  # deprecated: 'xb'
        sim.operations.writers.append(gsd_writer)
        sim.run(self.total_steps)

        ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=2e4)
        rho = sim.state.N_particles/sim.state.box.volume
        sim.box

    def operate_simulation_langevin_wca_opp(self):
        # initialize
        if self.mode == 'cpu':
            device = hoomd.device.CPU()
        elif self.mode == 'gpu':
            device = hoomd.device.GPU()
        sim = hoomd.Simulation(device=device, seed=self.seed)
        # sim.state
        self.__set_init_state_parameters()
        sim.create_state_from_snapshot(self.init_snap)
        # sim.create_state_from_gsd(filename='molecular.gsd')

        # set interaction
        # Apply the opp interaction(interfacial dipolar here) on the particles.
        opp_cell = hoomd.md.nlist.Cell(self.width)  # ('trap',)
        opp = hoomd.md.pair.OPP(nlist=opp_cell, default_r_cut=self.opp_r_cut, mode='xplor')
        opp.params[('particle', 'particle')] = dict(C1=self.opp_c1, C2=0,
                                                    eta1=3, eta2=0, k=0, phi=0)  # interfacial dipolar here

        # Apply the WCA interaction on the particles.
        wca_cell = hoomd.md.nlist.Cell(self.width)  # ('trap',)
        wca = hoomd.md.pair.LJ(nlist=wca_cell, default_r_cut=self.wca_r_cut, mode='shift')
        wca.params[('particle', 'particle')] = dict(sigma=self.wca_sigma,
                                                    epsilon=self.wca_epsilon)  # self.wca_sigma = 1.0,self.wca_epsilon = 1.0

        """#find the mobile particles
        #list_particles = range(self.snap.particles.N)
        list_ids = self.init_snap.particles.typeid#range(21)#np.linspace(0,20,21,dtype=int)
        list_Particles_index = list_ids == 0
        tags = list(np.where(list_Particles_index))
        #set system
        mobile_particles= hoomd.filter.Tags(tags)#tag n is the n-th particle, must be a list or iterator 
        #stationary_particles = hoomd.filter.Tags([1])
        #mobile_particles = hoomd.filter.SetDifference(hoomd.filter.All(),stationary_particles)"""
        langevin = hoomd.md.methods.Langevin(
            filter=hoomd.filter.All(),
            kT=self.kT,)  # mobile_particles
        integrator = hoomd.md.Integrator(dt=self.dt,
                                         methods=[langevin],
                                         forces=[opp, wca])

        sim.operations.integrator = integrator

        # https://gsd.readthedocs.io/en/v3.2.0/python-module-gsd.hoomd.html#gsd.hoomd.open
        gsd_writer = hoomd.write.GSD(filename=self.output_file_gsd,
                                     trigger=hoomd.trigger.Periodic(self.snap_period),
                                     mode='xb')  # deprecated: 'xb'
        sim.operations.writers.append(gsd_writer)
        sim.run(self.total_steps)
