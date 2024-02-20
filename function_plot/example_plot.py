import numpy as np
import matplotlib.pyplot as plt
class functions_plot_module:
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
        pass

    def generate_power_law(self):
        R"""
        introduction: 
            ln(y) = k*ln(x), linear in loglog,
            when -k > 1 , the cumulation is convergent series
        return: 
            x,y
        """
        x = np.linspace(0.1,10,100)
        y = np.power(x,-2)
        return x,y
    
    def generate_cumulate(self,x,y):
        R"""
        return x,y
        """
        y = tuple(y)
        sumy = np.array(y)
        sz = np.shape(sumy)
        for i in range(sz[0]):
            if i>0:
                sumy[i] = sumy[i]+sumy[i-1]
        return x,sumy

    def generate_exp(self):
        R"""
        introduction: 
            ln(y) = x*ln(e), linear in semilogy
            when e > 1 , the cumulation is convergent series
        return: 
            x,y
        """
        x = np.linspace(0.1,0.9,10)
        y = np.exp(-x)*2/1.1
        return x,y
    
    def generate_yukawa(self,epsilon=300,kappa=0.25):
        R"""
        introduction: 
            ln(y) = x*ln(e), linear in semilogy
            when e > 1 , the cumulation is convergent series
        return: 
            x,y
        """
        x = np.linspace(0.9,15,1000)
        ex = epsilon/x
        kx = kappa*x
        y = ex*np.exp(-kx)
        return x,y
    
    def generate_hertzian(self,epsilon,alpha):
        R"""
        introduction: 
            
        return: 
            x,y
        """
        x = np.linspace(0.1,0.99,99)
        ea = epsilon/alpha
        ya = np.power(1-x,alpha)
        list_x_large_bool = x[:] > 1.0
        ya[list_x_large_bool] = 0
        y = ea*ya
        return x,y
    
    def generate_kob_andersen(self,epsilon):
        R"""
        introduction: 
            
        return: 
            x,y
        """
        x = np.linspace(0.95,2.55,99)
        ea = 4*epsilon
        ya = np.power(x,-12) - np.power(x,-6)
        y0 = np.power(2.5,-12) - np.power(2.5,-6)
        y = ea*(ya-y0)
        return x,y

    def generate_dipole_coarse(self,epsilon=1500):
        R"""
        introduction: 
            ln(y) = k*ln(x), linear in loglog,
            when -k > 1 , the cumulation is convergent series
        return: 
            x,y
        example:
            import function_plot.example_plot as ep
            fpm = ep.functions_plot_module()
            x,y = fpm.generate_yukawa(300,0.25)
            x2,y2 = fpm.generate_dipole(1500)
            #2000>,r4-13 crossover
            #1500,r3-15
            #1000,r2.5
            #500,r2
            fpm.plot_function2(x,y,y2)
        """
        x = np.linspace(0.8,15,1000)
        y = epsilon*np.power(x,-3)
        return x,y
    
    def generate_dipole(self,epsilon=1500):
        R"""
        introduction: 
            ln(y) = k*ln(x), linear in loglog,
            when -k > 1 , the cumulation is convergent series
        return: 
            x,y
        example:
            import function_plot.example_plot as ep
            fpm = ep.functions_plot_module()
            x,y = fpm.generate_dipole_coarse(1500)
            x2,y2 = fpm.generate_dipole(1500*2)
            #dipole kqq=3000 ~ r-3 A1500=kqq/2
            fpm.plot_function2(x,y,y2)
        """
        x = np.linspace(0.8,15,1000)
        inv_r = 1/x
        inv_rd = 1/np.sqrt(1+x**2)
        y = epsilon*(inv_r-inv_rd)
        return x,y
    
    def check_PRE15(self):
        n_e = 570#540
        epsilon_relative = 4
        e1 = 1.602176e-19#C
        epsilon0 = 8.8541878e-12#F/m
        kb = 1.3806504e-23#J/K
        temperature = 300#K
        kee_over_kt = e1*e1/(4*np.pi*epsilon0*kb*temperature)#5.56e-8
        u_dipole = n_e*n_e/epsilon_relative*kee_over_kt/2
        print(kee_over_kt)
        print("A_dipole_coarse")#ref:5.56e-8
        print(u_dipole)
        #the result is 0.002<<1500 why?
        #the effective kT is far less than real one, given viscosity
    
    def generate_gaussian(self,epsilon = 300,sigma = 1,rcut = 2):
        R"""
        introduction: 
            
        return: 
            x,y
        example:
            import function_plot.example_plot as fp
            fpm = fp.functions_plot_module()
            fpm.generate_gaussian(1,1,1)
            fpm.generate_gaussian(10,1,1)
            fpm.generate_gaussian(1,0.6,2)
            fpm.generate_gaussian(10,0.6,2)
        example2:
            #tune multiple potentials
            import function_plot.example_plot as ep
            fpm = ep.functions_plot_module()
            eps=1
            sigma=1
            x1,y1 = fpm.generate_gaussian(eps/0.39347,sigma,rcut=1)
            x2,y2 = fpm.generate_harmonic(eps*2)
            fpm.plot_function22(x1,y1,x2,y2)
            x1,y1 = fpm.generate_gaussian_force(eps/0.39347,sigma)
            x2,y2 = fpm.generate_harmonic_force(eps*2)
            fpm.plot_function22(x1,y1,x2,y2)
        """
        # = 
        
        x = np.linspace(-rcut,rcut,41)
        y = -np.exp(-0.5*((x/sigma)**2))*epsilon + np.exp(-0.5*((rcut/sigma)**2))*epsilon
        delta_energy_per_epsilon = (1 - np.exp(-0.5*((rcut/sigma)**2)))*epsilon
        print('sigma',sigma,'rcut',rcut,'dE',delta_energy_per_epsilon)
        return x,y
    
    def generate_harmonic(self,epsilon = 300):
        R"""
        introduction: 
            ln(y) = x*ln(e), linear in semilogy
            when e > 1 , the cumulation is convergent series
        return: 
            x,y
        """
        # = 
        #sigma = 1
        x = np.linspace(-1,1,21)
        y = 0.5*(x**2)*epsilon - 0.5*epsilon
        return x,y

    def generate_gaussian_force(self,epsilon = 300,sigma = 1,rcut=1):
        R"""
        introduction: 
            ln(y) = x*ln(e), linear in semilogy
            when e > 1 , the cumulation is convergent series
        return: 
            x,y
        """
        x = np.linspace(-rcut,rcut,21)
        y = x*np.exp(-0.5*((x/sigma)**2))*epsilon/(sigma**2)
        return x,y
    
    def generate_harmonic_force(self,epsilon = 300):
        R"""
        introduction: 
            ln(y) = x*ln(e), linear in semilogy
            when e > 1 , the cumulation is convergent series
        return: 
            x,y
        """
        # = 
        #sigma = 1
        x = np.linspace(-1,1,21)
        y = x*epsilon
        return x,y

    def generate_xplor(self,r_on=0,r_cut=1.12):
        R"""
        introduction: 
            from HOOMD-blue, md.pair shifting/smoothing mode
            if ron=0, rcut=1: x=(0,0.5), y=1; x=(0.5,1), y decay to 0
        """
        x = np.linspace(0,r_cut,21)
        rc2 = r_cut**2
        ro2 = r_on**2
        r2 = np.power(x,2)
        y1 = np.power(rc2 - r2,2)
        y2 = rc2 + 2*r2 - 3*ro2
        y3 = np.power(rc2 - ro2,3)
        y = y1*y2/y3
        return x,y

    def generate_kauzmann_glass(self,b,m):
        R"""
        introduction: 
            ln(y) = x*ln(e), linear in semilogy
            when e > 1 , the cumulation is convergent series
        return: 
            x,y
        """
        xm = np.linspace(0.01,1,99)#Tg/T
        x = 1/xm#T/Tg
        y = b/(x-m)#m=T0/Tg
        return x,y
    
    def generate_glass_list(self):
        R"""
        introduction: 
            ln(y) = x*ln(e), linear in semilogy
            when e > 1 , the cumulation is convergent series
        return: 
            x,y
        """
        b=1
        xm = np.linspace(0.01,1,99)#Tg/T
        x = 1/xm#T/Tg
        fig,ax = plt.subplots()
        ax.set_xlabel('$T_g/T$')
        ax.set_ylabel('$\eta$')
        
        for m in np.linspace(0.1,0.9,9):
            y = b/(x-m)#m=T0/Tg
            ax.plot(1/x,y)
        plt.show()

    def generate_compare(self):
        xm = np.linspace(0.01,1,99)#Tg/T
        x = 1/xm#T/Tg
        print(x)

    def plot_function_glass(self,x,y):
        fig,ax = plt.subplots()
        ax.plot(1/x,y)
        ax.set_xlabel('$T_g/T$')
        ax.set_ylabel('$\eta$')
        plt.show()

    def plot_function(self,x,y):
        fig,ax = plt.subplots()
        ax.plot(x,y)#semilogy(x,y)#
        #ax.set_xlabel('$r/\sigma$')
        #ax.set_ylabel('$U$')
        plt.show()
    
    def plot_function2(self,x,y1,y2):
        R"""
        intro: U/kt = eps ~ k/2; force_gaus < force_harmo,
            when sigma=0.6, the two potential look similar.
        import function_plot.example_plot as ep
        fpm = ep.functions_plot_module()
        eps=300
        x1,y1 = fpm.generate_gaussian(eps,0.6)
        x2,y2 = fpm.generate_harmonic(2*eps)
        fpm.plot_function2(x1,y1,y2)
        """
        fig,ax = plt.subplots()
        ax.semilogy(x,y1,label='r-3,')#yukawa,k0.25,ep300
        ax.semilogy(x,y2,label='dipole,')#plot
        plt.legend()
        plt.show()
    
    def plot_function22(self,x1,y1,x2,y2):
        R"""
        intro: U/kt = eps ~ k/2; force_gaus < force_harmo,
            when epsilon=2k, sigma=0.6,rcut=2, 
            the two potential look similar.
        import function_plot.example_plot as ep
        fpm = ep.functions_plot_module()

        eps=300
        for sigma in [0.6,1]:
            x1,y1 = fpm.generate_gaussian(eps,sigma)
            x2,y2 = fpm.generate_harmonic(2*eps)
            fpm.plot_function22(x1,y1,x2,y2)
            x1,y1 = fpm.generate_gaussian_force(eps,sigma)
            x2,y2 = fpm.generate_harmonic_force(2*eps)
            fpm.plot_function22(x1,y1,x2,y2)
        """
        fig,ax = plt.subplots()
        ax.plot(x1,y1,label='gaus')
        ax.plot(x2,y2,label='harm')
        plt.legend()
        plt.show()
    
    def plot_function_loglog(self,x,y):
        fig,ax = plt.subplots()
        ax.loglog(x,y)
        plt.show()
    
    def plot_function_semilogy(self,x,y):
        fig,ax = plt.subplots()
        ax.semilogy(x,y)
        plt.show()