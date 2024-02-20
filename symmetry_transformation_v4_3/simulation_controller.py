import workflow_analysis as wa
import symmetry_transformation_v4_3.system_parameters_generators as pg
import symmetry_transformation_v4_3.simulation_core as sco
import numpy as np
import os
import pandas as pd

class simulation_controller_honeycomb_traps:
    def __init__(self):
        pass

    def generate_initial_state_hexagonal_particle_honeycomb_trap_single(self):
        R"""
        Introduction:
                a comprehensive workflow from creating initial state, 
            to record trajectory as a series of .png images, covering a 
            single simulation point (simu_index,seed,lcr,k)
        Example:
            import symmetry_transformation_v4_3.simulation_controller as sc
            import numpy as np
            sct = sc.simulation_controller_traps()
            sct.generate_initial_state_hexagonal_particle_honeycomb_trap()
        """
        

        prefix = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_0/"
        filename_initial_state = 'particle_hex_and_trap_honeycomb'
        gsd_filename = prefix+filename_initial_state+'.gsd'

        
        a_particle = 3
        lcr = 0.81
        particles = wa.archimedean_tilings()
        particles.generate_type1(a=a_particle)
        n_size = [16,8]
        particle_points = particles.generate_lattices(n_size)

        traps = wa.archimedean_tilings()
        traps.generate_type3(a=a_particle*lcr)
        isg = pg.initial_state_generator()
        
        isg.set_new_gsd_file_2types_by_n_size(particles,n_size,particle_points,traps,gsd_filename)
        
        #check the initial configuration of points and traps
        isg = pg.initial_state_generator()
        isg.read_gsd_file( gsd_filename)
        points = isg.particles.position
        
        ids = np.array(isg.snap.particles.typeid)
        list_p = ids == 0
        list_t = ids == 1

        isg.snap.particles.types
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.scatter(points[list_p,0],points[list_p,1],color='k')#
        ax.scatter(points[list_t,0],points[list_t,1],color='r',marker = 'x')#
        #ax.scatter(dula[:,0],dula[:,1],facecolors='none',edgecolors='k')#,marker = 'x'
        ax.set_xlabel('x label')  # Add an x-label to the axes.
        ax.set_ylabel('y label')  # Add a y-label to the axes.
        ax.set_title("Simple Plot")  # Add a title to the axes
        ax.set_aspect('equal','box')
        plt.show()
        plt.close()
        print('particle_hex_and_trap_honeycomb')

        #operate simulation
        sim = sco.simulation_core_traps(9999,9)
        sim.seed=9
        sim.mode = 'cpu'
        sim.gauss_epsilon = -50
        sim.input_file_gsd = gsd_filename
        sim.operate_simulation_langevin()
       
        
        #save trap position txt
        isg = pg.initial_state_generator()
        isg.read_gsd_file( gsd_filename)
        points = isg.particles.position
        ids = np.array(isg.snap.particles.typeid)
        list_p = ids == 0
        list_t = ids == 1
        save_file_txt = prefix + filename_initial_state + '.txt'
        np.savetxt(save_file_txt,points[list_t]) 


        import data_analysis_cycle as dac
        dac.save_from_gsd(9999,9,bond_plot=True,show_traps=True,trap_filename=save_file_txt,trap_lcr=1.0,list_traps=list_t,account='remote')

    def generate_initial_state_hexagonal_particle_honeycomb_trap_scan_k(self):
        R"""
        Introduction:
                a comprehensive workflow from creating initial state, 
            to record trajectory as a series of .png images, covering a 
            group of scanning k simulation points (simu_index,seed,lcr,k)
        Example:
            import symmetry_transformation_v4_3.simulation_controller as sc
            import numpy as np
            sct = sc.simulation_controller_traps()
            sct.generate_initial_state_hexagonal_particle_honeycomb_trap()
        """
        #series setup
        n_files = 10
        list_simu_index = np.linspace(5773,5782,n_files,dtype=int)
        list_gauss_epsilon = np.linspace(-3,-30,n_files)

        prefix = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_0/"
        filename_initial_state = 'particle_hex_and_trap_honeycomb'
        init_gsd_filename = prefix+filename_initial_state+'.gsd'#describe the lcr

        #set_initial_state_particles
        a_particle = 3
        lcr = 0.81
        particles = wa.archimedean_tilings()
        particles.generate_type1(a=a_particle)
        n_size = [16,8]
        particle_points = particles.generate_lattices(n_size)
        #set_initial_state_traps
        traps = wa.archimedean_tilings()
        traps.generate_type3(a=a_particle*lcr)
        isg = pg.initial_state_generator()
        

        isg.set_new_gsd_file_2types_by_n_size(particles,n_size,particle_points,traps,init_gsd_filename)
        
        #check the initial configuration of points and traps
        isg = pg.initial_state_generator()
        isg.read_gsd_file( init_gsd_filename)
        points = isg.particles.position
        
        ids = np.array(isg.snap.particles.typeid)
        list_p = ids == 0
        list_t = ids == 1

        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.scatter(points[list_p,0],points[list_p,1],color='k')#
        ax.scatter(points[list_t,0],points[list_t,1],color='r',marker = 'x')#
        #ax.scatter(dula[:,0],dula[:,1],facecolors='none',edgecolors='k')#,marker = 'x'
        ax.set_xlabel('x label')  # Add an x-label to the axes.
        ax.set_ylabel('y label')  # Add a y-label to the axes.
        ax.set_title("Simple Plot")  # Add a title to the axes
        ax.set_aspect('equal','box')
        plt.show()
        plt.close('all')
        print('particle_hex_and_trap_honeycomb')

        #for seed in range(10):
        seed = 9
        for i in range(len(list_simu_index)):
            #operate simulation
            simu_index = list_simu_index[i]

            sim = sco.simulation_core_traps(simu_index,seed)
            sim.seed=seed
            sim.mode = 'cpu'
            sim.gauss_epsilon = list_gauss_epsilon[i]#-50
            sim.input_file_gsd = init_gsd_filename
            sim.operate_simulation_langevin()
        
            
            """#save trap position txt
            isg = pg.initial_state_generator()
            isg.read_gsd_file( init_gsd_filename)
            points = isg.particles.position
            ids = np.array(isg.snap.particles.typeid)
            list_p = ids == 0
            list_t = ids == 1
            save_file_txt = prefix + filename_initial_state + '.txt'
            np.savetxt(save_file_txt,points[list_t]) 


            import data_analysis_cycle as dac
            dac.save_from_gsd(simu_index,seed,bond_plot=True,show_traps=True,trap_filename=save_file_txt,trap_lcr=1.0,list_traps=list_t,account='remote')
            """

    def generate_simu_index_csv_6373(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_pin/'
        index1 = 6373#5813
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_klt_2m_gauss_'+str(index1)+'.csv'#'pin_hex_to_honeycomb_klt_2m_gauss.csv'
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3
        gauss_epsilon_step = -3
        n_gauss_epsilons = 30
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_lcr = [0.82,0.83,0.84]#[0.77,0.78,0.79,0.80]
        temperature = 1
        t = temperature

        
        for seed in range(10):
            simu_index = index1
            for lcr in list_lcr:
                for gauss_epsilon in list_gauss_epsilon:
                    if ((simu_index == index1) & (seed == 0)):
                        record1 = [[index1,seed,list_lcr[0],gauss_epsilon1,t]]
                        column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']#,'cn3'
                        record_all = pd.DataFrame(record1)
                        record_all.columns = column
                        simu_index = simu_index + 1
                    else:
                        record1 = pd.DataFrame([[simu_index,seed,lcr,gauss_epsilon,t]],columns=column)
                        record_all = pd.concat([record_all,record1])
                        simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_5773_5812(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_pin/'
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_klt_2m_gauss_5773_5812.csv'
        index1 = 5773
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3
        gauss_epsilon_step = -3
        n_gauss_epsilons = 20
        list_gauss_epsilon1 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        gauss_epsilon1 = -100
        gauss_epsilon_step = -100
        n_gauss_epsilons = 10
        list_gauss_epsilon2 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        gauss_epsilon1 = -63
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon3 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_gauss_epsilon = np.concatenate((list_gauss_epsilon1,list_gauss_epsilon2,list_gauss_epsilon3))
        list_lcr = 0.81
        temperature = 1
        t = temperature

        
        for seed in range(10):
            simu_index = index1
            for gauss_epsilon in list_gauss_epsilon:
                if ((simu_index == index1) & (seed == 0)):
                    record1 = [[index1,seed,list_lcr,list_gauss_epsilon[0],t]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr,gauss_epsilon,t]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_5813_5812(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_pin/'
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_klt_2m_gauss_5773_5812.csv'
        index1 = 5773
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3
        gauss_epsilon_step = -3
        n_gauss_epsilons = 20
        list_gauss_epsilon1 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        gauss_epsilon1 = -100
        gauss_epsilon_step = -100
        n_gauss_epsilons = 10
        list_gauss_epsilon2 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        gauss_epsilon1 = -63
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon3 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_gauss_epsilon = np.concatenate((list_gauss_epsilon1,list_gauss_epsilon2,list_gauss_epsilon3))
        list_lcr = 0.81
        temperature = 1
        t = temperature

        
        for seed in range(10):
            simu_index = index1
            for gauss_epsilon in list_gauss_epsilon:
                if ((simu_index == index1) & (seed == 0)):
                    record1 = [[index1,seed,list_lcr,list_gauss_epsilon[0],t]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr,gauss_epsilon,t]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_3_242(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_pin/'
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'
        index1 = 3
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3
        gauss_epsilon_step = -3
        n_gauss_epsilons = 30
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_lcr = [0.77,0.78,0.79,0.80,0.81,0.82,0.83,0.84]
        temperature = 1
        t = temperature

        for seed in range(10):
            simu_index = index1
            for lcr in list_lcr:
                for gauss_epsilon in list_gauss_epsilon:
                    if ((simu_index == index1) & (seed == 0)):
                        record1 = [[index1,seed,list_lcr[0],gauss_epsilon1,t]]
                        column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']#,'cn3'
                        record_all = pd.DataFrame(record1)
                        record_all.columns = column
                    else:
                        record1 = pd.DataFrame([[simu_index,seed,lcr,gauss_epsilon,t]],columns=column)
                        record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)

    def record_simu_csv_into_sql(self):
        import opertateOnMysql as osql
        filename_csv = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_pin/honeycomb_pin_scan_k3_60_simu.csv'
        #osql.createTableInMysql('pin_hex_to_honeycomb_klt_2m_gauss','simu_index integer unsigned not null,seed integer unsigned not null,lcr float, trap_gauss_epsilon float, temperature float')
        osql.loadCsvDataToMysql\
        (path_to_file_name=filename_csv,
        table_name='pin_hex_to_honeycomb_klt_2m_gauss')

    def generate_initial_state_hexagonal_particle_honeycomb_trap_scan_csv(self,csv_filename):
        R"""
        Introduction:
                a comprehensive workflow from creating initial state, 
            to record trajectory as a series of .png images, covering a 
            group of scanning k simulation points (simu_index,seed,lcr,k)

            csv format: column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']
        Example:
            import symmetry_transformation_v4_3.simulation_controller as sc
            import numpy as np
            sct = sc.simulation_controller_traps()
            sct.generate_initial_state_hexagonal_particle_honeycomb_trap()
        """
        #series setup
        df = pd.read_csv(csv_filename)
        col = df.columns
        print(df.head())
        n_size = [16,8]
        a_particle = 3
        list_simu_index = df['simu_index'].values
        list_seed = df['seed'].values
        list_lcr = df['lcr'].values
        list_trap_gauss_epsilon = df['trap_gauss_epsilon'].values
        list_temperature = df['temperature'].values

        for i in range(len(list_simu_index)):
            #operate simulation
            simu_index = list_simu_index[i]
            seed = list_seed[i]
            lcr = list_lcr[i]
            gauss_epsilon = list_trap_gauss_epsilon[i]
            temperature = list_temperature[i]
            init_gsd_filename = generate_initial_gsd_hex_and_type_n(3,a_particle,n_size,lcr)#self.generate_initial_gsd(a_particle,n_size,lcr)
            print(simu_index,seed,lcr,gauss_epsilon,temperature,init_gsd_filename)
            sim = sco.simulation_core_traps(simu_index,seed)
            sim.seed=seed
            sim.gauss_r_cut = 2.0
            sim.gauss_sigma = 0.6
            sim.mode = 'cpu'
            sim.gauss_epsilon = gauss_epsilon#-50
            sim.kT = temperature
            #if seed==0:#select a part to simu
            sim.input_file_gsd = init_gsd_filename
            sim.operate_simulation_langevin()
            print(i+1,'/',len(list_simu_index))

    def generate_initial_gsd(sel,a_particle,n_size,lcr):#deprecated
        prefix = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/"
        #filename_initial_state should be like 
        # 'init_particle_<lattice_type>_<a>_<nx>_<ny>_and_trap_<lattice_type>_<int_lcr>
        # to prevent the mixing of different initial state .gsd file
        str_lcr = str(lcr)#'081'#in case the string like '0.810000000032'
        filename_initial_state = 'init_particle_hex_'+str(a_particle)+'_'+str(n_size[0])+'_'+str(n_size[1])+'_and_trap_honeycomb_'+str_lcr
                
        init_gsd_filename = prefix+filename_initial_state+'.gsd'#describe the lcr
         
        #check if the file exists
        isExists=os.path.exists(init_gsd_filename)
        if isExists:
            pass
        else:
            #set_initial_state_particles
            particles = wa.archimedean_tilings()
            particles.generate_type1(a=a_particle)
            #set_initial_state_traps
            traps = wa.archimedean_tilings()
            traps.generate_type3(a=a_particle*lcr)
            isg = pg.initial_state_generator()
            
            isg.set_new_gsd_file_2types_by_box_or_n_size(particles,traps,init_gsd_filename,perturb=False,n_size=n_size)
            
            """#check the initial configuration of points and traps
            isg = pg.initial_state_generator()
            isg.read_gsd_file( init_gsd_filename)
            points = isg.particles.position
            
            ids = np.array(isg.snap.particles.typeid)
            list_p = ids == 0
            list_t = ids == 1
            #check the init state
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots()
            ax.scatter(points[list_p,0],points[list_p,1],color='k')#
            ax.scatter(points[list_t,0],points[list_t,1],color='r',marker = 'x')#
            #ax.scatter(dula[:,0],dula[:,1],facecolors='none',edgecolors='k')#,marker = 'x'
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
            plt.close('all')
            print('particle_hex_and_trap_honeycomb')"""
            print('create:'+init_gsd_filename)
        return init_gsd_filename

class simulation_controller_honeycomb_part_traps:
    def __init__(self):
        pass    

    def generate_simu_index_csv_lcr080(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_part_pin/'
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_5973.csv'#5933,
        index1 = 5973
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3
        gauss_epsilon_step = -3
        n_gauss_epsilons = 30
        list_gauss_epsilon1 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        gauss_epsilon1 = -100
        gauss_epsilon_step = -100
        n_gauss_epsilons = 10
        list_gauss_epsilon2 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_gauss_epsilon = np.concatenate((list_gauss_epsilon1,list_gauss_epsilon2))
        lcr = 0.80
        temperature = 1
        t = temperature

        
        for seed in range(10):
            simu_index = index1
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1 and seed == 0):
                    record1 = [[index1,seed,lcr,list_gauss_epsilon[0],t]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,lcr,gauss_epsilon,t]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_lcr081(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_part_pin/'
        index1 = 6013
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_'+str(int(index1))+'.csv'#5933,
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3
        gauss_epsilon_step = -3
        n_gauss_epsilons = 30
        list_gauss_epsilon1 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        gauss_epsilon1 = -100
        gauss_epsilon_step = -100
        n_gauss_epsilons = 10
        list_gauss_epsilon2 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_gauss_epsilon = np.concatenate((list_gauss_epsilon1,list_gauss_epsilon2))
        list_lcr = 0.81
        temperature = 1
        t = temperature

        
        seed=9#for seed in range(10):
        simu_index = index1
        for gauss_epsilon in list_gauss_epsilon:
            if (simu_index == index1):
                record1 = [[index1,seed,list_lcr,list_gauss_epsilon[0],t]]
                column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']#,'cn3'
                record_all = pd.DataFrame(record1)
                record_all.columns = column
                simu_index = simu_index + 1
            else:
                record1 = pd.DataFrame([[simu_index,seed,list_lcr,gauss_epsilon,t]],columns=column)
                record_all = pd.concat([record_all,record1])
                simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_lcr081_079(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_part_pin/'
        index1 = 6013
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_'+str(int(index1))+'_not8.csv'#5933,
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3
        gauss_epsilon_step = -3
        n_gauss_epsilons = 30
        list_gauss_epsilon1 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        gauss_epsilon1 = -100
        gauss_epsilon_step = -100
        n_gauss_epsilons = 10
        list_gauss_epsilon2 = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_gauss_epsilon = np.concatenate((list_gauss_epsilon1,list_gauss_epsilon2))
        lcr = 0.81
        temperature = 1
        t = temperature

        list_seed=[0,1,2,3,4,5,6,7,9]#for seed in range(10):
        for seed in list_seed:
            simu_index = index1
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1 and seed == 0):
                    record1 = [[index1,seed,lcr,list_gauss_epsilon[0],t]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,lcr,gauss_epsilon,t]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_6373_6612(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_part_pin/'
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
        index1 = 6373
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3
        gauss_epsilon_step = -3
        n_gauss_epsilons = 30
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_lcr = [0.77,0.78,0.79,0.80,0.81,0.82,0.83,0.84]
        temperature = 1
        t = temperature

        for seed in range(10):
            simu_index = index1
            for lcr in list_lcr:
                for gauss_epsilon in list_gauss_epsilon:
                    if ((simu_index == index1) & (seed == 0)):
                        record1 = [[index1,seed,list_lcr[0],gauss_epsilon1,t]]
                        column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']#,'cn3'
                        record_all = pd.DataFrame(record1)
                        record_all.columns = column
                    else:
                        record1 = pd.DataFrame([[simu_index,seed,lcr,gauss_epsilon,t]],columns=column)
                        record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_6553_6612(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_part_pin/'
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_6553_6612.csv'
        index1 = 6553
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -100
        gauss_epsilon_step = -100
        n_gauss_epsilons = 10
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        
        list_lcr = [0.77,0.78,0.79,0.82,0.83,0.84]
        temperature = 1
        t = temperature

        for seed in range(10):
            simu_index = index1
            for lcr in list_lcr:
                for gauss_epsilon in list_gauss_epsilon:
                    if ((simu_index == index1) & (seed == 0)):
                        record1 = [[index1,seed,list_lcr[0],gauss_epsilon1,t]]
                        column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']#,'cn3'
                        record_all = pd.DataFrame(record1)
                        record_all.columns = column
                        simu_index = simu_index + 1
                    else:
                        record1 = pd.DataFrame([[simu_index,seed,lcr,gauss_epsilon,t]],columns=column)
                        record_all = pd.concat([record_all,record1])
                        simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_initial_state_hexagonal_particle_honeycomb_part_trap_scan_csv(self,csv_filename):
        R"""
        Introduction:
                a comprehensive workflow from creating initial state, 
            to record trajectory as a series of .png images, covering a 
            group of scanning k simulation points (simu_index,seed,lcr,k)

            csv format: column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']
        Example:
            import symmetry_transformation_v4_3.simulation_controller as sc
            #import symmetry_transformation_v4_3.simulation_core as sco
            sct = sc.simulation_controller_honeycomb_part_traps()
            prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_part_pin/'
            file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_5933.csv'
            sct.generate_initial_state_hexagonal_particle_honeycomb_part_trap_scan_csv(file_csv)
        """
        #series setup
        df = pd.read_csv(csv_filename)
        col = df.columns
        print(df.head())
        n_size = [16,8]
        a_particle = 3
        list_simu_index = df['simu_index'].values
        list_seed = df['seed'].values
        list_lcr = df['lcr'].values
        list_trap_gauss_epsilon = df['trap_gauss_epsilon'].values
        list_temperature = df['temperature'].values

        for i in range(len(list_simu_index)):
            #operate simulation
            simu_index = list_simu_index[i]
            seed = list_seed[i]
            lcr = list_lcr[i]
            gauss_epsilon = list_trap_gauss_epsilon[i]
            temperature = list_temperature[i]
            init_gsd_filename = self.generate_initial_gsd(a_particle,n_size,lcr)
            print(simu_index,seed,lcr,gauss_epsilon,temperature,init_gsd_filename)
            sim = sco.simulation_core_traps(simu_index,seed)
            sim.seed=seed
            sim.gauss_r_cut = 2.0
            sim.gauss_sigma = 0.6
            sim.mode = 'cpu'
            sim.gauss_epsilon = gauss_epsilon#-50
            sim.kT = temperature
            sim.input_file_gsd = init_gsd_filename
            #if seed==0:#select a part to simu
            sim.operate_simulation_langevin()
            print(i+1,'/',len(list_simu_index))
    
    def generate_initial_state_hexagonal_particle_honeycomb_part_gaus_eq_harmo_trap_scan_csv(self,csv_filename):
        R"""
        Introduction:
                a comprehensive workflow from creating initial state, 
            to record trajectory as a series of .png images, covering a 
            group of scanning k simulation points (simu_index,seed,lcr,k)

            csv format: column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature']
        Example:
            import symmetry_transformation_v4_3.simulation_controller as sc
            #import symmetry_transformation_v4_3.simulation_core as sco
            sct = sc.simulation_controller_honeycomb_part_traps()
            prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_part_pin/'
            file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_5933.csv'
            sct.generate_initial_state_hexagonal_particle_honeycomb_part_trap_scan_csv(file_csv)
        """
        #series setup
        df = pd.read_csv(csv_filename)
        
        col = df.columns
        print(df.head())
        
        n_size = [16,8]
        a_particle = 3
        list_simu_index = df['simu_index'].values
        list_seed = df['seed'].values+10#[v]
        list_lcr = df['lcr'].values
        list_trap_gauss_epsilon = df['trap_gauss_epsilon'].values
        list_temperature = df['temperature'].values

        for i in range(len(list_simu_index)):
            #operate simulation
            simu_index = list_simu_index[i]
            seed = list_seed[i]
            lcr = list_lcr[i]
            gauss_epsilon = list_trap_gauss_epsilon[i]
            temperature = list_temperature[i]
            init_gsd_filename = self.generate_initial_gsd(a_particle,n_size,lcr)
            print(simu_index,seed,lcr,gauss_epsilon,temperature,init_gsd_filename)
            sim = sco.simulation_core_traps(simu_index,seed)
            sim.gauss_sigma = 0.6#[v]
            sim.gauss_r_cut = 2.0#[v]
            sim.seed=seed
            sim.mode = 'cpu'
            sim.gauss_epsilon = gauss_epsilon#-50
            sim.kT = temperature
            sim.input_file_gsd = init_gsd_filename
            if seed==19:#select a part to simu
                sim.operate_simulation_langevin()
            print(i+1,'/',len(list_simu_index))

    def generate_initial_gsd(sel,a_particle,n_size,lcr):
        prefix = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/"
        #filename_initial_state should be like 
        # 'init_particle_<lattice_type>_<a>_<nx>_<ny>_and_trap_<lattice_type>_<int_lcr>
        # to prevent the mixing of different initial state .gsd file
        str_lcr = str(lcr)#'081'#in case the string like '0.810000000032'
        filename_initial_state = 'init_particle_hex_'+str(a_particle)+'_'+str(n_size[0])+'_'+str(n_size[1])+'_and_trap_honeycomb_part_'+str_lcr
                
        init_gsd_filename = prefix+filename_initial_state+'.gsd'#describe the lcr
         
        #check if the file exists
        isExists=os.path.exists(init_gsd_filename)
        if isExists:
            pass
        else:
            #set_initial_state_particles
            particles = wa.archimedean_tilings()
            particles.generate_type1(a=a_particle)
            #set_initial_state_traps
            traps = wa.archimedean_tilings()
            traps.generate_type3_part(a=a_particle*lcr)
            isg = pg.initial_state_generator()
            isg.set_new_gsd_file_2types_by_box_or_n_size(particles,traps,init_gsd_filename,perturb=False,n_size=n_size)
            """#check the initial configuration of points and traps
            isg = pg.initial_state_generator()
            isg.read_gsd_file( init_gsd_filename)
            points = isg.particles.position
            
            ids = np.array(isg.snap.particles.typeid)
            list_p = ids == 0
            list_t = ids == 1
            #check the init state
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots()
            ax.scatter(points[list_p,0],points[list_p,1],color='k')#
            ax.scatter(points[list_t,0],points[list_t,1],color='r',marker = 'x')#
            #ax.scatter(dula[:,0],dula[:,1],facecolors='none',edgecolors='k')#,marker = 'x'
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            plt.show()
            plt.close('all')
            print('particle_hex_and_trap_honeycomb')"""
            print('create:'+init_gsd_filename)
        return init_gsd_filename

class simulation_controller_type_n_part_traps:
    def __init__(self):
        pass    
    
    def get_type_n_lcr0(self):
        import workflow_analysis as wa
        import numpy as np
        record_lcr0 = np.zeros((11,))
        for i in range(11):
            at = wa.archimedean_tilings()
            at.generate_type_n(i+1)
            cross_lattice = np.cross(at.a1,at.a2)
            area_per_particle = cross_lattice[2]/len(at.position)
            area_hex = np.sqrt(3)/2.0
            lcr0 = np.sqrt(area_hex/area_per_particle)
            record_lcr0[i] = lcr0
            #print('type'+str(i+1)+': '+str(np.round(lcr0,4) ))
            del at
        return record_lcr0
        
    def generate_simu_index_csv_depin(self):#not finished
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_depin/'
        index1 = 6053
        output_file_csv = prefix_write + 'depin_type_n_from_type_n_part_klt_2m_gauss_'+str(int(index1))+'.csv'#5933,
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3
        gauss_epsilon_step = -3
        n_gauss_epsilons = 30
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        
        list_lcr = self.get_type_n_lcr0()
        list_lcr =  list_lcr[3:]
        temperature = 1
        t = temperature
        list_type_n = np.linspace(4,11,8,dtype=int)
        
        seed=9#for seed in range(10):
        simu_index = index1
        for i in range(list_type_n.shape[0]):# type and lcr
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1):
                    record1 = [[index1,seed,list_lcr[0],list_gauss_epsilon[0],t,list_type_n[0]]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr[i],gauss_epsilon,t,list_type_n[i]]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_depin_low_t(self):#not finished
        R"""
        import symmetry_transformation_v4_3.simulation_controller as sc
        sct = sc.simulation_controller_type_n_part_traps()
        sct.generate_simu_index_csv_depin_low_t()
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_depin/'
        index1 = 7163
        output_file_csv = prefix_write + 'depin_type_n_from_type_n_part_klt_2m_gauss_'+str(int(index1))+'.csv'
        sct.generate_initial_state_type_n_particle_type_n_part_trap_depin_scan_csv(output_file_csv)
        """
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_depin/'
        index1 = 7163
        output_file_csv = prefix_write + 'depin_type_n_from_type_n_part_klt_2m_gauss_'+str(int(index1))+'.csv'#5933,
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        
        list_lcr = self.get_type_n_lcr0()
        
        temperature = 0.1
        t = temperature
        list_type_n = np.array([4,5,6,9],dtype=int)
        list_i = list_type_n - 1
        list_lcr =  list_lcr[list_i]
        
        seed=9#for seed in range(10):
        simu_index = index1
        for i in range(list_type_n.shape[0]):# type and lcr
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1):
                    record1 = [[index1,seed,list_lcr[0],list_gauss_epsilon[0],t,list_type_n[0]]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr[i],gauss_epsilon,t,list_type_n[i]]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_depin_special_low_t(self):#not finished
        R"""
        import symmetry_transformation_v4_3.simulation_controller as sc
        sct = sc.simulation_controller_type_n_part_traps()
        sct.generate_simu_index_csv_depin_low_t()
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_depin/'
        index1 = 7163
        output_file_csv = prefix_write + 'depin_type_n_from_type_n_part_klt_2m_gauss_'+str(int(index1))+'.csv'
        sct.generate_initial_state_type_n_particle_type_n_part_trap_depin_scan_csv(output_file_csv)
        """
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_depin/'
        index1 = 7203
        output_file_csv = prefix_write + 'depin_type_n_from_type_n_part_special_klt_2m_gauss_'+str(int(index1))+'.csv'#5933,
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        
        list_lcr = self.get_type_n_lcr0()
        
        temperature = 0.1
        t = temperature
        list_type_n = np.array([62,72],dtype=int)
        list_i = [5,6]#list_type_n - 1
        list_lcr =  list_lcr[list_i]
        
        seed=9#for seed in range(10):
        simu_index = index1
        for i in range(list_type_n.shape[0]):# type and lcr
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1):
                    record1 = [[index1,seed,list_lcr[0],list_gauss_epsilon[0],t,list_type_n[0]]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr[i],gauss_epsilon,t,list_type_n[i]]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)

    def generate_simu_index_csv_pin_3_30(self):#not finished
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
        index1 = 6613
        output_file_csv = prefix_write + 'pin_hex_to_type_n_part_klt_2m_gauss_'+str(int(index1))+'.csv'#5933,
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3#check the depin trans rate first. start from depin strength
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        
        list_lcr = self.get_type_n_lcr0()
        list_lcr =  list_lcr[3:]
        temperature = 1
        t = temperature
        list_type_n = [7,10,11]
        
        seed=9#for seed in range(10):
        simu_index = index1
        for i in range(len(list_type_n)):# type and lcr
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1):
                    record1 = [[index1,seed,list_lcr[0],list_gauss_epsilon[0],t,list_type_n[0]]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr[i],gauss_epsilon,t,list_type_n[i]]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_type_4_pin_3_30(self):#not finished
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
        index1 = 6963#6823#[x]
        list_type_n = 4#[x]
        output_file_csv = prefix_write + 'pin_hex_to_type_'+str(int(list_type_n))+'_part_klt_2m_gauss_'+str(int(index1))+'.csv'#[x]
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3#check the depin trans rate first. start from depin strength
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_lcr = np.linspace(0.64,0.68,5)#[x]
        temperature = 1
        t = temperature
        seed=9#for seed in range(10):
        simu_index = index1
        for i in range(len(list_lcr)):# type and lcr
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1):
                    record1 = [[index1,seed,list_lcr[0],list_gauss_epsilon[0],t,list_type_n]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr[i],gauss_epsilon,t,list_type_n]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_type_5_pin_3_30(self):#not finished
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
        index1 = 7013#6823#[x]
        list_type_n = 5#[x]
        output_file_csv = prefix_write + 'pin_hex_to_type_'+str(int(list_type_n))+'_part_klt_2m_gauss_'+str(int(index1))+'.csv'#[x]
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3#check the depin trans rate first. start from depin strength
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_lcr = np.linspace(0.71,0.75,5)#[x]
        temperature = 1
        t = temperature
        seed=9#for seed in range(10):
        simu_index = index1
        for i in range(len(list_lcr)):# type and lcr
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1):
                    record1 = [[index1,seed,list_lcr[0],list_gauss_epsilon[0],t,list_type_n]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr[i],gauss_epsilon,t,list_type_n]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_type_6_pin_3_30(self):#not finished
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
        index1 = 7063#6823#[x]
        list_type_n = 6#[x]
        output_file_csv = prefix_write + 'pin_hex_to_type_'+str(int(list_type_n))+'_part_klt_2m_gauss_'+str(int(index1))+'.csv'#[x]
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3#check the depin trans rate first. start from depin strength
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_lcr = np.linspace(0.75,0.79,5)#[x]
        temperature = 1
        t = temperature
        seed=9#for seed in range(10):
        simu_index = index1
        for i in range(len(list_lcr)):# type and lcr
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1):
                    record1 = [[index1,seed,list_lcr[0],list_gauss_epsilon[0],t,list_type_n]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr[i],gauss_epsilon,t,list_type_n]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)

    def generate_simu_index_csv_type_9_pin_3_30(self):#not finished
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
        index1 = 7113#6823#[x]
        list_type_n = 9#[x]
        output_file_csv = prefix_write + 'pin_hex_to_type_'+str(int(list_type_n))+'_part_klt_2m_gauss_'+str(int(index1))+'.csv'#[x]
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3#check the depin trans rate first. start from depin strength
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        list_lcr = np.linspace(0.91,0.95,5)#[x]
        temperature = 1
        t = temperature
        seed=9#for seed in range(10):
        simu_index = index1
        for i in range(len(list_lcr)):# type and lcr
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1):
                    record1 = [[index1,seed,list_lcr[0],list_gauss_epsilon[0],t,list_type_n]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr[i],gauss_epsilon,t,list_type_n]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)

    def generate_simu_index_csv_type_7_pin_3_30(self):#not finished
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
        index1 = 6913#6823#[x]
        output_file_csv = prefix_write + 'pin_hex_to_type_7_part_klt_2m_gauss_'+str(int(index1))+'.csv'#[x]
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3#check the depin trans rate first. start from depin strength
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        
        list_lcr = np.linspace(0.85,0.89,5)#np.linspace(0.9,0.98,9)#0.9~
        #list_lcr =  list_lcr[3:]
        temperature = 1
        t = temperature
        list_type_n = 7#[x]
        
        seed=9#for seed in range(10):
        simu_index = index1
        for i in range(len(list_lcr)):# type and lcr
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1):
                    record1 = [[index1,seed,list_lcr[0],list_gauss_epsilon[0],t,list_type_n]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr[i],gauss_epsilon,t,list_type_n]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)

    def generate_simu_index_csv_type_10_pin_3_30(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
        index1 = 6643
        output_file_csv = prefix_write + 'pin_hex_to_type_10_part_klt_2m_gauss_'+str(int(index1))+'.csv'#5933,
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3#check the depin trans rate first. start from depin strength
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        
        list_lcr = np.linspace(0.97,1.05,9)#self.get_type_n_lcr0()
        #list_lcr =  list_lcr[3:]
        temperature = 1
        t = temperature
        list_type_n = 10#[7,10,11]
        
        seed=9#for seed in range(10):
        simu_index = index1
        for i in range(len(list_lcr)):# type and lcr
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1):
                    record1 = [[index1,seed,list_lcr[0],list_gauss_epsilon[0],t,list_type_n]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr[i],gauss_epsilon,t,list_type_n]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_simu_index_csv_type_11_pin_3_30(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
        index1 = 6733#[x]
        output_file_csv = prefix_write + 'pin_hex_to_type_11_part_klt_2m_gauss_'+str(int(index1))+'.csv'#[x]
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3#check the depin trans rate first. start from depin strength
        gauss_epsilon_step = -3
        n_gauss_epsilons = 10
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        
        list_lcr = np.linspace(0.97,1.05,9)#self.get_type_n_lcr0()
        #list_lcr =  list_lcr[3:]
        temperature = 1
        t = temperature
        list_type_n = 11#[x]
        
        seed=9#for seed in range(10):
        simu_index = index1
        for i in range(len(list_lcr)):# type and lcr
            for gauss_epsilon in list_gauss_epsilon:
                if (simu_index == index1):
                    record1 = [[index1,seed,list_lcr[0],list_gauss_epsilon[0],t,list_type_n]]
                    column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                    record_all = pd.DataFrame(record1)
                    record_all.columns = column
                    simu_index = simu_index + 1
                else:
                    record1 = pd.DataFrame([[simu_index,seed,list_lcr[i],gauss_epsilon,t,list_type_n]],columns=column)
                    record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_initial_state_type_n_particle_type_n_part_trap_depin_scan_csv(self,csv_filename):
        R"""
        Introduction:
                a comprehensive workflow from creating initial state, 
            to record trajectory as a series of .png images, covering a 
            group of scanning k simulation points (simu_index,seed,lcr,k)

            csv format: column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']
        Example:
            import symmetry_transformation_v4_3.simulation_controller as sc
            #import symmetry_transformation_v4_3.simulation_core as sco
            sct = sc.simulation_controller_honeycomb_part_traps()
            prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_part_pin/'
            file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_5933.csv'
            sct.generate_initial_state_hexagonal_particle_honeycomb_part_trap_scan_csv(file_csv)
        """
        #series setup
        df = pd.read_csv(csv_filename)
        box = [50,50]
        a_particle = 3
        list_simu_index = df['simu_index'].values
        list_seed = df['seed'].values
        list_lcr = df['lcr'].values
        list_trap_gauss_epsilon = df['trap_gauss_epsilon'].values
        list_temperature = df['temperature'].values
        list_type_n = df['type_n'].values

        for i in range(len(list_simu_index)):
            if i>=0:
                #operate simulation
                simu_index = list_simu_index[i]
                seed = list_seed[i]
                lcr = list_lcr[i]
                gauss_epsilon = list_trap_gauss_epsilon[i]
                temperature = list_temperature[i]
                init_gsd_filename = self.generate_initial_gsd_type_n_and_part(list_type_n[i],a_particle,box,lcr)
                print(simu_index,seed,lcr,gauss_epsilon,temperature,init_gsd_filename)
                sim = sco.simulation_core_traps(simu_index,seed)
                sim.seed=seed
                sim.width = 10#if it is too small, sparse lattice will disfunction.
                sim.mode = 'gpu'
                sim.gauss_epsilon = gauss_epsilon#-50
                sim.total_steps = 1e6+1
                sim.kT = temperature
                sim.input_file_gsd = init_gsd_filename
                print(i+1,'/',len(list_simu_index))
                sim.operate_simulation_langevin()
        
    def generate_initial_state_hex_particle_type_n_part_trap_scan_csv(self,csv_filename):
        R"""
        Introduction:
                a comprehensive workflow from creating initial state, 
            to record trajectory as a series of .png images, covering a 
            group of scanning k simulation points (simu_index,seed,lcr,k)

            csv format: column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']
        Example:
            import symmetry_transformation_v4_3.simulation_controller as sc
            #import symmetry_transformation_v4_3.simulation_core as sco
            sct = sc.simulation_controller_honeycomb_part_traps()
            prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/honeycomb_part_pin/'
            file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_5933.csv'
            sct.generate_initial_state_hexagonal_particle_honeycomb_part_trap_scan_csv(file_csv)
        """
        #series setup
        df = pd.read_csv(csv_filename)
        n_size = [16,8]
        a_particle = 3
        list_simu_index = df['simu_index'].values
        list_seed = df['seed'].values
        list_lcr = df['lcr'].values
        list_trap_gauss_epsilon = df['trap_gauss_epsilon'].values
        list_temperature = df['temperature'].values
        list_type_n = df['type_n'].values

        for i in range(len(list_simu_index)):
            #operate simulation
            simu_index = list_simu_index[i]
            #if simu_index > 6773:
            seed = list_seed[i]
            lcr = list_lcr[i]
            gauss_epsilon = list_trap_gauss_epsilon[i]
            temperature = list_temperature[i]
            init_gsd_filename = generate_initial_gsd_hex_and_type_n(list_type_n[i],a_particle,n_size,lcr,True)
            #init_gsd_filename = self.generate_initial_gsd_hex_and_type_n_part(list_type_n[i],a_particle,n_size,lcr)
            print(simu_index,seed,lcr,gauss_epsilon,temperature,init_gsd_filename)
            sim = sco.simulation_core_traps(simu_index,seed)
            sim.gauss_sigma = 0.6
            sim.gauss_r_cut = 2.0
            sim.seed=seed
            sim.mode = 'gpu'
            sim.gauss_epsilon = gauss_epsilon#-50
            sim.total_steps = 2e6+1
            sim.kT = temperature
            sim.input_file_gsd = init_gsd_filename
            sim.operate_simulation_langevin()
            print(i+1,'/',len(list_simu_index))
        
    def generate_initial_gsd_type_n_and_part(self,type_n,a_particle,box,lcr):
        type_n = int(type_n)
        if type_n>0 :#and type_n<12:
            pass
        else:
            print('error: type_n should be an int between [1,11]!')
            print('\n')
        prefix = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_0/"
        #filename_initial_state should be like 
        # 'init_particle_<lattice_type>_<a>_<nx>_<ny>_and_trap_<lattice_type>_<int_lcr>
        # to prevent the mixing of different initial state .gsd file
        str_lcr = str(lcr)#'081'#in case the string like '0.810000000032'
        filename_initial_state = 'init_particle_type_'+str(type_n)+'_'+str(a_particle)+'_'+str(box[0])+'_'+str(box[1])+'_and_trap_type_'+str(type_n)+'_part_'+str_lcr
                
        init_gsd_filename = prefix+filename_initial_state+'.gsd'#describe the lcr
         
        #check if the file exists
        isExists=os.path.exists(init_gsd_filename)
        if isExists:
            pass
        else:
            #set_initial_state_particles
            particles = wa.archimedean_tilings()
            particles.generate_type_n(type_n,a=a_particle*lcr)
            #set_initial_state_traps
            traps = wa.archimedean_tilings()
            traps.generate_type_n_part(type_n,a=a_particle*lcr)
            isg = pg.initial_state_generator()
            isg.set_new_gsd_file_2types_by_box_or_n_size(particles,traps,init_gsd_filename,box=box,perturb=True)#,perturb=True
            #set_new_gsd_file_2types_depin_by_box(particles,box,traps,init_gsd_filename,perturb=True)#False
            
            """#check the initial configuration of points and traps
            #isg = pg.initial_state_generator()
            isg.read_gsd_file( init_gsd_filename)
            points = isg.particles.position
            
            ids = np.array(isg.snap.particles.typeid)
            list_p = ids == 0
            list_t = ids == 1
            #check the init state
            import matplotlib.pyplot as plt
            fig,ax = plt.subplots()
            ax.scatter(points[list_p,0],points[list_p,1],color='k')#
            ax.scatter(points[list_t,0],points[list_t,1],color='r',marker = 'x')#
            ax.plot(isg.position_box[:,0],isg.position_box[:,1],c='k')
            #ax.scatter(dula[:,0],dula[:,1],facecolors='none',edgecolors='k')#,marker = 'x'
            ax.set_xlabel('x label')  # Add an x-label to the axes.
            ax.set_ylabel('y label')  # Add a y-label to the axes.
            ax.set_title("Simple Plot")  # Add a title to the axes
            ax.set_aspect('equal','box')
            png_filename='init_type_'+str(type_n)+'.png'
            plt.savefig(png_filename)#plt.show()
            plt.close('all')"""
            print('particle_type_n_and_trap_type_n_part')
            print('create:'+init_gsd_filename)
        return init_gsd_filename
    
    """def generate_initial_gsd_hex_and_type_n_part(self,type_n,a_particle,n_size,lcr):
        type_n = int(type_n)
        if type_n>0 and type_n<12:
            pass
        else:
            print('error: type_n should be an int between [1,11]!')
            print('\n')
        prefix = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/"
        #filename_initial_state should be like 
        # 'init_particle_<lattice_type>_<a>_<nx>_<ny>_and_trap_<lattice_type>_<int_lcr>
        # to prevent the mixing of different initial state .gsd file
        str_lcr = str(lcr)#'081'#in case the string like '0.810000000032'
        filename_initial_state = 'init_particle_hex_'+str(a_particle)+'_'+str(n_size[0])+'_'+str(n_size[1])+'_and_trap_type_'+str(type_n)+'_part_'+str_lcr
        init_gsd_filename = prefix+filename_initial_state+'.gsd'#describe the lcr

        #check if the file exists
        isExists=os.path.exists(init_gsd_filename)
        if isExists:
            pass
        else:
            #set_initial_state_particles
            particles = wa.archimedean_tilings()
            particles.generate_type1(a=a_particle)
            #particle_points = particles.generate_lattices(n_size)
            #set_initial_state_traps
            traps = wa.archimedean_tilings()
            traps.generate_type_n_part(type_n,a=a_particle*lcr)
            isg = pg.initial_state_generator()
            isg.set_new_gsd_file_2types_by_box_or_n_size(particles,traps,init_gsd_filename,perturb=False,n_size=n_size)#False     
            print('create:'+init_gsd_filename)
        return init_gsd_filename"""

    def test_init(self):#checked right
        list_lcr0 = self.get_type_n_lcr0()
        for i in range(11):
            type_n = i+1
            if type_n > 3:
                self.generate_initial_gsd_type_n_and_part(type_n,3,[50,50],list_lcr0[i])

class simulation_controller_kagome_traps:
    def __init__(self):
        pass   
    
    def generate_simu_index_csv_kagome_pin_3_90(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
        index1 = 243#[x]
        output_file_csv = prefix_write + 'pin_hex_to_type_8_klt_2m_gauss_'+str(int(index1))+'.csv'#[x]
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -3#check the depin trans rate first. start from depin strength
        gauss_epsilon_step = -3
        n_gauss_epsilons = 30
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        
        list_lcr = np.linspace(0.83,0.91,9)#self.get_type_n_lcr0()
        #list_lcr =  list_lcr[3:]
        temperature = 1
        t = temperature
        list_type_n = 8#[x]
        
        for seed in range(10):
            simu_index = index1
            for lcr in list_lcr:
                for gauss_epsilon in list_gauss_epsilon:
                    if ((simu_index == index1) & (seed == 0)):
                        record1 = [[index1,seed,list_lcr[0],gauss_epsilon1,t,list_type_n]]
                        column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                        record_all = pd.DataFrame(record1)
                        record_all.columns = column
                    else:
                        record1 = pd.DataFrame([[simu_index,seed,lcr,gauss_epsilon,t,list_type_n]],columns=column)
                        record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1

        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_initial_state_hex_particle_kagome_trap_scan_csv(self,csv_filename):
        R"""
        Introduction:
                a comprehensive workflow from creating initial state, 
            to record trajectory as a series of .png images, covering a 
            group of scanning k simulation points (simu_index,seed,lcr,k)

            csv format: column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']
        Example:
            import symmetry_transformation_v4_3.simulation_controller as sc
            sct = sc.simulation_controller_kagome_traps()
            prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
            index1 = 243#[x]
            output_file_csv = prefix_write + 'pin_hex_to_type_8_klt_2m_gauss_'+str(int(index1))+'.csv'#[x]
            #sct.generate_simu_index_csv_kagome_pin_3_90()
            sct.generate_initial_state_hex_particle_kagome_trap_scan_csv(output_file_csv)
        """
        #series setup
        df = pd.read_csv(csv_filename)
        
        col = df.columns
        #print(df.head())
        
        n_size = [16,8]
        a_particle = 3
        list_simu_index = df['simu_index'].values
        list_seed = df['seed'].values
        list_lcr = df['lcr'].values
        list_trap_gauss_epsilon = df['trap_gauss_epsilon'].values
        list_temperature = df['temperature'].values
        list_type_n = df['type_n'].values

        for i in range(len(list_simu_index)):
            #operate simulation
            simu_index = list_simu_index[i]
            #if simu_index > 6773:
            seed = list_seed[i]
            lcr = list_lcr[i]
            gauss_epsilon = list_trap_gauss_epsilon[i]
            temperature = list_temperature[i]
            init_gsd_filename = generate_initial_gsd_hex_and_type_n(list_type_n[i],a_particle,n_size,lcr)
            #init_gsd_filename = self.generate_initial_gsd_hex_and_type_n_part(list_type_n[i],a_particle,n_size,lcr)
            
            sim = sco.simulation_core_traps(simu_index,seed)
            sim.gauss_sigma = 0.6
            sim.gauss_r_cut = 2.0
            sim.seed=seed
            sim.mode = 'gpu'
            sim.gauss_epsilon = gauss_epsilon#-50
            sim.total_steps = 2e6+1
            sim.kT = temperature
            sim.input_file_gsd = init_gsd_filename
            if seed == 3:
                print(simu_index,seed,lcr,gauss_epsilon,temperature,init_gsd_filename)
                sim.operate_simulation_langevin()
                print(i+1,'/',len(list_simu_index))
        
class simulation_controller_kagome_part_traps:
    def __init__(self):
        pass    
    
    def generate_simu_index_csv_kagome_part_pin_3_90(self):
        #series setup
        prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
        index1 = 513#[x]
        output_file_csv = prefix_write + 'pin_hex_to_type_8_part_klt_2m_gauss_'+str(int(index1))+'.csv'#[x]
        #list_simu_index = np.linspace(index1,index1+n_files-1,n_files,dtype=int)
        gauss_epsilon1 = -6#check the depin trans rate first. start from depin strength
        gauss_epsilon_step = -6
        n_gauss_epsilons = 30
        list_gauss_epsilon = np.linspace(gauss_epsilon1,gauss_epsilon1+gauss_epsilon_step*(n_gauss_epsilons-1),n_gauss_epsilons)
        
        list_lcr = np.linspace(0.83,0.91,9)#self.get_type_n_lcr0()
        #list_lcr =  list_lcr[3:]
        temperature = 1
        t = temperature
        list_type_n = 8#[x]
        
        for seed in range(10):
            simu_index = index1
            for lcr in list_lcr:
                for gauss_epsilon in list_gauss_epsilon:
                    if ((simu_index == index1) & (seed == 0)):
                        record1 = [[index1,seed,list_lcr[0],gauss_epsilon1,t,list_type_n]]
                        column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']#,'cn3'
                        record_all = pd.DataFrame(record1)
                        record_all.columns = column
                    else:
                        record1 = pd.DataFrame([[simu_index,seed,lcr,gauss_epsilon,t,list_type_n]],columns=column)
                        record_all = pd.concat([record_all,record1])
                    simu_index = simu_index + 1
        pd.DataFrame.to_csv(record_all,output_file_csv)
    
    def generate_initial_state_hex_particle_kagome_part_trap_scan_csv(self,csv_filename):
        R"""
        Introduction:
                a comprehensive workflow from creating initial state, 
            to record trajectory as a series of .png images, covering a 
            group of scanning k simulation points (simu_index,seed,lcr,k)

            csv format: column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']
        Example:
            import symmetry_transformation_v4_3.simulation_controller as sc
            sct = sc.simulation_controller_kagome_part_traps()
            prefix_write = '/media/remote/32E2D4CCE2D49607/file_lxt/record_results_v430/type_n_pin/'
            index1 = 513#[x]
            output_file_csv = prefix_write + 'pin_hex_to_type_8_part_klt_2m_gauss_'+str(int(index1))+'.csv'#[x]
            #sct.generate_simu_index_csv_kagome_part_pin_3_90()
            sct.generate_initial_state_hex_particle_kagome_part_trap_scan_csv(output_file_csv)
        """
        #series setup
        df = pd.read_csv(csv_filename)
        
        col = df.columns
        #print(df.head())
        
        n_size = [16,8]
        a_particle = 3
        list_simu_index = df['simu_index'].values
        list_seed = df['seed'].values
        list_lcr = df['lcr'].values
        list_trap_gauss_epsilon = df['trap_gauss_epsilon'].values
        list_temperature = df['temperature'].values
        list_type_n = df['type_n'].values

        for i in range(len(list_simu_index)):
            #operate simulation
            simu_index = list_simu_index[i]
            #if simu_index > 6773:
            seed = list_seed[i]
            lcr = list_lcr[i]
            gauss_epsilon = list_trap_gauss_epsilon[i]
            temperature = list_temperature[i]
            init_gsd_filename = generate_initial_gsd_hex_and_type_n(list_type_n[i],a_particle,n_size,lcr,type_n_part=True)
            #init_gsd_filename = self.generate_initial_gsd_hex_and_type_n_part(list_type_n[i],a_particle,n_size,lcr)
            sim = sco.simulation_core_traps(simu_index,seed)
            sim.gauss_sigma = 0.6
            sim.gauss_r_cut = 2.0
            sim.seed=seed
            sim.mode = 'gpu'
            sim.gauss_epsilon = gauss_epsilon#-50
            sim.total_steps = 2e6+1
            sim.kT = temperature
            sim.input_file_gsd = init_gsd_filename
            if seed == 1 :
                print(simu_index,seed,lcr,gauss_epsilon,temperature,init_gsd_filename)
                sim.operate_simulation_langevin()
                print(i+1,'/',len(list_simu_index))

def generate_initial_gsd_hex_and_type_n(type_n,a_particle,n_size,lcr,type_n_part=False):
    R"""
    generate_initial_gsd_hex_and_type_n: 
        generate a .gsd file as a initial state 
        which set hexagonal particles and type_n traps(type_n_part traps if type_n_part=True). 
    type_n: 
        define which type of archimedean tiling will be set as traps. 
        (1 hex, 2 sq, 3 honeycomb, 8 kagome)
    a_particle:
        the lattice constant of hexagonal lattice
    n_size:
        [nx,ny] lattices of particles the box is filled.
    lcr: 
        rescale the lattice constant of trap lattice.(a_trap = lcr*a_particle)
    type_n_part: 
        set type_n traps if type_n_part=False; set type_n_part traps if type_n_part=True.
    """
    type_n = int(type_n)
    if type_n>0 and type_n<12:
        pass
    else:
        print('error: type_n should be an int between [1,11]!')
        print('\n')
    prefix = "/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_1/"
    #filename_initial_state should be like 
    # 'init_particle_<lattice_type>_<a>_<nx>_<ny>_and_trap_<lattice_type>_<int_lcr>
    # to prevent the mixing of different initial state .gsd file
    str_lcr = str(lcr)#'081'#in case the string like '0.810000000032'
    if type_n_part:
        filename_initial_state = 'init_particle_hex_'+str(a_particle)+'_'+str(n_size[0])+'_'+str(n_size[1])+'_and_trap_type_'+str(type_n)+'_part_'+str_lcr
    else:
        filename_initial_state = 'init_particle_hex_'+str(a_particle)+'_'+str(n_size[0])+'_'+str(n_size[1])+'_and_trap_type_'+str(type_n)+'_'+str_lcr
    init_gsd_filename = prefix+filename_initial_state+'.gsd'#describe the lcr

    #check if the file exists
    isExists=os.path.exists(init_gsd_filename)
    if isExists:
        pass
    else:
        #set_initial_state_particles
        particles = wa.archimedean_tilings()
        particles.generate_type1(a=a_particle)
        #particle_points = particles.generate_lattices(n_size)
        #set_initial_state_traps
        traps = wa.archimedean_tilings()
        if type_n_part:
            traps.generate_type_n_part(type_n,a=a_particle*lcr)
        else:
            traps.generate_type_n(type_n,a=a_particle*lcr)
        isg = pg.initial_state_generator()
        isg.set_new_gsd_file_2types_by_box_or_n_size(particles,traps,init_gsd_filename,perturb=False,n_size=n_size)#False     
        """#check the initial configuration of points and traps
        #isg = pg.initial_state_generator()
        isg.read_gsd_file( init_gsd_filename)
        points = isg.particles.position
        
        ids = np.array(isg.snap.particles.typeid)
        list_p = ids == 0
        list_t = ids == 1
        #check the init state
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        ax.scatter(points[list_p,0],points[list_p,1],color='k')#
        ax.scatter(points[list_t,0],points[list_t,1],color='r',marker = 'x')#
        ax.plot(isg.position_box[:,0],isg.position_box[:,1],c='k')
        #ax.scatter(dula[:,0],dula[:,1],facecolors='none',edgecolors='k')#,marker = 'x'
        ax.set_xlabel('x label')  # Add an x-label to the axes.
        ax.set_ylabel('y label')  # Add a y-label to the axes.
        ax.set_title("Simple Plot")  # Add a title to the axes
        ax.set_aspect('equal','box')
        png_filename='init_type_'+str(type_n)+'.png'
        plt.savefig(png_filename)#plt.show()
        plt.close('all')
        print('particle_hex_and_trap_type_n_part')"""
        print('create:'+init_gsd_filename)
    return init_gsd_filename