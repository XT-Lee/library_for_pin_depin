import numpy as np
import matplotlib.pyplot as plt

class yukawa_crystal_module:
    R"""
    exp:
        import function_plot.example_plot as ex
        ef = ex.functions_plot_module()
        x,y = ef.generate_power_law()
        ef.plot_function_semilogy(x,y)
        x,y = ef.generate_cumulate(x,y)
        ef.plot_function(x,y)
    """
    def __init__(self) :
        self.set_system_param()
        self.set_crystal_type()
        self.get_volume_fraction()

    def set_system_param(self):
        #static
        self.__n_particle_per_lattice_fcc = 4
        self.__n_particle_per_lattice_bcc = 2

        #dynamic
        self.a_neighbor_distance = 1.0
        self.sigma = 1.0#um
        self.lambda_water = 0.16#um

    def set_crystal_type(self,name="fcc"):
        self.__crystal_type = "error"
        if name == "fcc":
            self.__crystal_type = name
            self.__n_particle_per_lattice = self.__n_particle_per_lattice_fcc
            self.__edge_length_close_packing = 2*self.sigma/np.sqrt(2)
        elif name == "bcc":
            self.__crystal_type = name
            self.__n_particle_per_lattice = self.__n_particle_per_lattice_bcc
            self.__edge_length_close_packing = 2*self.sigma/np.sqrt(3)

        self.volume_fraction_close_packing,self.n_density_close_packing = self.get_volume_fraction()

    def get_volume_fraction(self,edge_length_times=1.0):
        volume_particle_in_lattice = 4/3*np.pi*np.power(self.sigma/2,3)*self.__n_particle_per_lattice
        volume_lattice = np.power(edge_length_times*self.__edge_length_close_packing,3)
        volume_fraction = volume_particle_in_lattice/volume_lattice
        n_density = self.__n_particle_per_lattice/volume_lattice
       
        return volume_fraction,n_density

    def get_kappa(self,edge_length_times=1.0):
        self.rho_from_n_density = np.power(self.n_density_close_packing,-1/3)*edge_length_times# n^(-1/3) * L
        self.kappa = self.rho_from_n_density/self.lambda_water
    
    def get_cystal_type(self):
        print(self.__crystal_type)
    
    def get_kappa_from_volume_fraction(self,edge_length_times):
        print("edge_length_times:",edge_length_times)
        volume_fraction,n_density = self.get_volume_fraction(edge_length_times)
        print("volume fraction:",volume_fraction)
        self.get_kappa(edge_length_times)
        print("kappa:",self.kappa)
        #print("\n")