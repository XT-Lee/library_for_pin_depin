import numpy as np
import matplotlib.pyplot as plt

class particle_decorator:
    def __init__(self,image_size,particle_position,diameter):
        R"""
        input:
            image_size: (n_x_pixel,n_y_pixel)
            particle_position: xy(pixel)
            diameter: n_d_pixel
        example:
            import particle_tracking as pt
            pa = pt.particle_track()
            prefix = '/home/remote/Downloads/image_proceed/honey_part/'
            image_filename = 'DefaultImage_12.jpg'
            x0 = 141
            y0 = 141
            limit = [x0,1023-x0,y0,1023-y0]
            diameter = 17
            pa.single_frame_particle_tracking(prefix+image_filename,D=diameter,minmass=1400,calibration=False,axis_limit=limit)#D = 16.7 pixel
            import particle_decorate as pd 
            pd.particle_decorator((1023-2*x0+2,1023-2*x0+2),pa.xy,diameter) #((100,100),[[50,50],[0,0]],100)#  
        """
        self.image = np.zeros(image_size)
        self.positions = np.array(particle_position)
        self.d = diameter
        self.draw_round_spot()#

    def draw_round_spot(self,mode='exp'):#'exp','harmony',nx
        image0 = np.zeros((len(self.image[0,:]),len(self.image[0,:]),3))
        """
        
        remaind = np.remainder(nx,2)
        if remaind == 0:
            rx = nx/2
        elif remaind == 1:
            rx = (nx+1)/2
        x = np.linspace(0,rx-1,rx)
        """
        X,Y = np.meshgrid(range(len(self.image[0,:])),range(len(self.image[:,0])))# ,indexing='ij'  then for i in range(x) j in range(x)
        Z = 0
        value_size = 1#255 or 1
        if mode == 'exp':
            rr = 0.4#radius_ratio to match real particle size in image 0.1~0.5
            for i in range(len(self.positions[:,0])):
                Z = Z + value_size*np.exp(-(((X-self.positions[i,0])**2+(Y-self.positions[i,1])**2))/(self.d*rr)**2)
        elif mode == 'harmony':
            rr = 0.5#radius_ratio to match real particle size in image 0.1~0.5
            for i in range(len(self.positions[:,0])):
                Z = Z + value_size*(1-(((X-self.positions[i,0])**2+(Y-self.positions[i,1])**2))/(self.d*rr)**2)\
                    *((((X-self.positions[i,0])**2+(Y-self.positions[i,1])**2)) < (self.d*rr)**2)
        elif mode == 'diffraction':
            """sin(a)/a, a = d*sin(theta)*2*pi/lambda"""
            pass
        image0[:,:,1] = np.array(Z)#,dtype=int
        save_filename = '/home/remote/Downloads/image_proceed/honey_part/img10gd_'+mode+'_imsave.png'
        plt.figure(4)
        plt.imshow(image0)
        plt.imsave(save_filename,image0)
        #plt.savefig(save_filename)
        #plt.show()

    def draw_raw_image(self,image,save_filename):
        plt.imsave(save_filename,image/255)
    
    def draw_map(self):
        traps=self.positions
        rcut=1.0
        cmap_name = 'Reds'#'autumn_r'#'autumn'#newcmp#'binary'#
        transparency = 0.5#0.3

        #set traps
        max = np.max(traps)
        min = np.min(traps)
        length = (max - min)
        steps = length/(rcut/10.0)
        #plt.style.use('_mpl-gallery-nogrid')

        # make data
        X, Y = np.meshgrid(np.linspace(min, max, steps.astype(int)), np.linspace(min, max, steps.astype(int)))
        HarmonicK = 100
        #origin = np.zeros((1,2))
        sz = np.shape(traps)
        i = 0
        Z = ( (0.50*HarmonicK*rcut*rcut-0.50*HarmonicK*((X-traps[i,0])**2 + (Y-traps[i,1])**2))\
            *(((X-traps[i,0])**2 + (Y-traps[i,1])**2) < rcut*rcut) )
        i = i+1
        while i<sz[0]:#sz[0]
            Zi = (0.50*HarmonicK*rcut*rcut-0.50*HarmonicK*((X-traps[i,0])**2 + (Y-traps[i,1])**2))\
                *(((X-traps[i,0])**2 + (Y-traps[i,1])**2) < rcut*rcut)
            Z = Z + Zi
            i = i+1
        
        self.ax.pcolormesh(X, Y, Z,cmap=cmap_name,zorder = -1,alpha=transparency)#,zorder=1