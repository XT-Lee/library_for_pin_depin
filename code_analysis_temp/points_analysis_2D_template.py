
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from scipy.spatial import Voronoi
from scipy.spatial import  Delaunay
from scipy.spatial import  distance
import pandas as pd
import matplotlib 
import os

R"""
Region to write codes with copilot
"""
#start

#read the codes listed in the file
#use a similar style of codes 

class dynamic_points_analysis_2d:#old_class_name: msd
    R"""
    Introduction:
        this class is designed to analyze dynamic properties of a group of paritcle trajectories.
    
    Methods:
        msd_module: mean square displacements.
        displacement_field_module: displacement field.
    
    example:
        #final cut bond plot of a simulation
        import data_analysis_cycle as dac
        import points_analysis_2D as pa
        import numpy as np
        daw = dac.data_analysis_workflow()
        directory,dataname=daw.gsd_to_txyz(simu_index=5387)
        txyz = np.load('/home/remote/Downloads/5387_9/txyz.npy')
        dpa = pa.dynamic_points_analysis_2d(txyz)
        dpa.plot_bond_neighbor_change_oop(data_name=dataname,prefix=directory,final_cut=True,
                        trap_filename='/home/remote/hoomd-examples_0/testhoneycomb3-8-12-part1',
                        trap_lcr=0.816)
    """
    def __init__(self,txyz_stable,mode='simu'):
        R"""
        parameters:
            txyz_stable:array[Nframes,Nparticles,xyz],
                for simu data, 
                    ensure that particles never move across the boundary(box)!
                for exp data, 
                    unit of xyz must be um! 
                    ensure that particles are always in the vision field!
            #BOX: [lx,ly,lz,xy,xz,yz]
            account: 'remote' or 'tplab'
            mode: 'simu' or 'exp'. 'simu' to select particles in box; 'exp' to direct compute msd
        """            
        if mode == 'simu':
            #self.box = box
            #self.__select_stable_trajectories_simu(plot_trajectory)
            self.txyz_stable = txyz_stable
            self.x_unit = '($\sigma$)'
            self.t_unit = '(step)'
        elif mode == 'exp':
            self.txyz_stable = txyz_stable
            self.x_unit = '(um)'
            self.t_unit = '(s)'

        #self.account = account
        self.mode = mode
    
    def get_saturate_time_of_tanh_like_data(self,orderp,threshold=0.95):
        R"""
        introduction:
           the function is written to get the effective evolution time of the order parameter, which ranges from activation to saturation.
        parameters:
            orderp: Nframes rows of [order_parameter]
        """
        import data_decorate as dd
        ddr = dd.data_decorator()
        list_index, data_decorated = ddr.coarse_grainize_and_average_data_log(orderp,coarse_grain_to_n_points=10,navg_odd=5) 
        for i in range(1,np.shape(data_decorated)[0]):
            if data_decorated[i]/orderp[-1] > threshold:
                saturate_time = list_index[i]
                break
        return saturate_time


class covariance_matrix_module:
    def __init__(self, txy):
        R"""
        Introduction:
            get the covariance matrix of a trajectory.
        Parameters:
            txy: whose shape is (Nframes,Nparticles,Ndimensions). here Ndimensions is 2.
                (x,y) are coordinations of given particles.
        Returns:
            tx: whose shape is (Nframes,Nparticles*Ndimensions). Nframes rows of (x0,y0,x1,y1,...,xN,yN).
        """
        self.txy = txy
        if len(np.shape(txy)) != 2:
            self.tx = self.data_transformation_from_trajectory_to_pca(txy)
        else:
            self.tx = txy
    
    def get_covariance_matrix_of_tx_from_averaged_position(self):
        R"""
        Introduction:
            get the covariance matrix of the array txy.suitable for the oscillation trajectory.
            if forced to evaluate directional-moving trajectory, it can hardly find the difference
            between oscillation trajectory and directional-moving trajectory.
        Parameters:
            tx: whose shape is (Nframes,Nparticles*Ndimensions). (x,y), coordinations of given particles.
        Returns:
            cov: a Nparticles*Ndimension order matrix, the covariance matrix of the tx.
        """
        return np.cov(self.tx.T)
    
    def get_covariance_matrix_of_tx_from_initial_position(self):
        R"""
        Introduction:
            get the covariance matrix of the array txy. suitable for the directional-moving trajectory.
            if forced to evaluate oscillation trajectory, the result replies on the original position related to the periodicity.
            eg: originate from center, result is small; originate from the peak, result is large.
        Parameters:
            tx: whose shape is (Nframes,Nparticles*Ndimensions). (x,y), coordinations of given particles.
            txo: tx[:] - tx[0]
        Returns:
            cov: a Nparticles*Ndimension order matrix, the covariance matrix of the tx.
        """
        txo = self.tx - self.tx[0]
        #print(f"txo: {txo}\n")
        #multiply the two matrices txo and txo.T
        mp = np.matmul(txo.T,txo)
        #print(f"mp: {mp}\n")
        mp_averaged = mp/np.shape(txo)[0] 
        return mp_averaged

    def data_transformation_from_trajectory_to_pca(self,txy):
        R"""
        Introduction:
            transform the array txy to a new array with the same shape.
        Parameters:
            txy: whose shape is (Nframes,Nparticles,Ndimensions). here Ndimensions is 2.
                (x,y) are coordinations of given particles.
        Returns:
            tx: whose shape is (Nframes,Nparticles*Ndimensions). Nframes rows of (x0,y0,x1,y1,...,xN,yN).
        """
        print("txy.shape",txy.shape)
        tx = np.reshape(txy, (txy.shape[0], -1))
        print("tx.shape",tx.shape)
        return tx
#end

#are there any other libraries to be imported?


R"""
CLASS list:
    static_points_analysis_2d: old_class_name: PointsAnalysis2D
    dynamic_points_analysis_2d: old_class_name: msd
"""
class static_points_analysis_2d:#old_class_name: PointsAnalysis2D
    R"""
    Introduction:
        the class is designed to analyze the 
        geometric properties(listed as follows) of 2D points.
    
    Parameters:
        points: n rows of (x,y), coordinations of given particles.
        ---
        voronoi: scipy.spatial.Voronoi
        voronoi.ridge_length: num-of-ridges row of (ridge_length)
        voronoi.cut_ridge_length: a length that 
            if a ridge whose length is smaller than which, the ridge will be cut.
        voronoi.cluster_vertices: list the row indices of vertices that may form clusters.
        ---
        delaunay: scipy.spatial.Delaunay
        bond_length: n rows of (start_point_index, end_point_index, bond_length)
        cutted_bonds: n rows of (start_point_index, end_point_index)
        vertices_triangle: input row indices of vertices,
            and get row indices of triangles which cover these vertices.[x]
        count_coordination: record the count of the coordination number who has i neighbours [0n,1n,2n,...,9n]
            that is, 0n is the count of coordination number 0, 1n is the count of coordination number 1,
            5n is the count of coordination number 5,6n is the count of coordination number 6
        count_coordination_ratio:[x]count_coordination whose sum is normalized to 1.
        ---
        Psi_k: n rows of complex number, Bond Orientaional Order(k fold disclination) 
            for each particle.
        Psi_k_global: arithmetic mean of absolute value of Psi_k s.
        Psi_k_rate: A is the num of particles whose psi_k is larger than 0.9, while
            B is the num of all the particles. Psi_k_rate = A/B.

    Methods:
        points_analysis: get the parameters listed before
        get_ridge_length: 
        get_bond_length:
        get_first_minima_bond_length_distribution: 
        get_coordination_number_conditional:
        get_cluster_vertices:
        get_vertices_triangle:[x]
        get_bond_orientational_order: get bond orientational order, Psi_n, for each particle.
        get_bond_orientational_order_selected:
        get_neighbor_cloud: place all the neighbor particles into a plot 
                            with the same position of central particles
        ----
        draw_bonds_conditional_bond: given bond_length condition, draw bonds
        draw_bonds_conditional_ridge: given ridge_length condition, draw bonds
        draw_bond_length_distribution_and_first_minima
        ----
        print_benchmark: print (min,median,max) data for bond or ridge
        ----
        count_n_polygon_fast: vertices cluster method

    Examples:
        import points_analysis_2D
        obj_of_simu_index = points_analysis_2D.PointsAnalysis2D(filename=data_filename)
    """
    def __init__(self,points=None,filename=None,hide_figure = True,dis_edge_cut=None):
        #load positions of particles
        if points is None:
            if filename is None:
                print("Error: input points or file please!\n")
            else:
                self.filename = filename
                self.points = np.loadtxt(filename)
                self.points = self.points[:,0:2]
        else :
            self.points = points[:,:2]
            
        if dis_edge_cut is None:
            self.basic_points_analysis()
        else:
            self.basic_points_analysis(dis_edge_cut)
        #not show figures
        if hide_figure:
            matplotlib.use(backend="agg")#Backend agg is non-interactive backend. Turning interactive mode off. 'QtAgg' is interactive mode
            
    def basic_points_analysis(self,dis_edge_cut=None):
        self.voronoi = Voronoi(self.points)
        self.delaunay = Delaunay(self.points)
        self.get_ridge_length()
        self.get_bond_length()
        #cut the edge of positions
        if dis_edge_cut is None:
            self.__cut_edge_of_positions()#effective lattice constant is given 3 defaultly
        else:
            self.__cut_edge_of_positions(dis_edge_cut)

    def get_ridge_length(self):
        #where the ridge length of vertex-i & vertex-j is ridge_length[i,j]
        self.voronoi.ridge_vertices=np.array(self.voronoi.ridge_vertices)#选不了ridge_vertices,不是数组格式
        self.voronoi.ridge_length=distance.cdist(self.voronoi.vertices[self.voronoi.ridge_vertices[:,0],:],self.voronoi.vertices[self.voronoi.ridge_vertices[:,1],:],'euclidean')
        self.voronoi.ridge_length=self.voronoi.ridge_length.diagonal()

    def get_bond_length(self):
        #self.bond_length: (start_point_index,end_point_index,bond_length)
        shp=np.shape(self.voronoi.ridge_points)
        self.bond_length = np.zeros((shp[0],shp[1]+1))#(start_point,end_point,length)
        self.bond_length_temp=distance.cdist(self.voronoi.points[self.voronoi.ridge_points[:,0],:], self.voronoi.points[self.voronoi.ridge_points[:,1],:], 'euclidean')
        self.bond_length[:,0:2]=self.voronoi.ridge_points
        self.bond_length[:,2]=self.bond_length_temp.diagonal()
        #should I list point pair like small-to-large？
        del self.bond_length_temp #delete temp data

    def get_first_minima_bond_length_distribution(self,lattice_constant=3.0,hist_cutoff=2,
                                    method="global_compare_method", png_filename=None,x_unit='(1)'):
        R"""  
        Introduction:
            It's likely that bonds whose length are larger than 2A are not bonded. 
            Typical bond-lengths of honeycomb are A and 1.73A (bond_first_neighbour & bond_second_neighbour namely),
            hence the range=[0,2] times of lattice constant should be set.
        Parameters:
            lattice_constant: particle diameter or lattice constant is recommended(also the unit A in x-axis in hist)
            
            method:"local_compare_nethod" to get the 1st minima through comparing local minima,
                suitable for continuous peak-valley hist;
                "global_compare_method" to get the 1st minima through comparing all the peaks, 
                selecting the 1st main peak(ignoring tiny peak before which), 
                finding the 1st main minima just after the 1st main peak, 
                suitable systems with powerful perturbation.
            
            hist_cutoff: plot hist of bond_length till n times lattice_constant where n is hist_cutoff.
            
            png_filename="prefix/bond_hist_index1512"

            bond_first_minima_left:the upper limit of 1st neighbour bonds comming from the first minima 
                                    of bond length distribution, with sigma as lower limit.
            
        Warning:
            [x]What if there is no minima found? 
            count_1st_max / count_1st_minima > 10 used to check the effective of minima? 

        Examples:
            s = "/home/tplab/Downloads/"
            index_num = 1387
            index_name = "index"+str(int(index_num))
            fname = s+index_name
            bb = pa.PointsAnalysis2D(filename=fname)
            oname1 = s+"bond_hist_"+index_name
            bb.draw_bond_length_distribution_and_first_minima(png_filename=oname1)
            oname2 = s+"bond_plot_"+index_name
            bb.draw_bonds_conditional_bond(check=[0.9, bb.bond_first_minima_left], png_filename=oname2)
        """
        self.lattice_constant=lattice_constant
        
        #locate the 1st minima of bond-length distribution
        plt.figure()
        count_bins=plt.hist(self.bond_length[:,2]/self.lattice_constant,bins=20,range=[0,hist_cutoff])
        
        self._count=count_bins[0]
        self._bins=count_bins[1]
        self.bond_sorted=np.sort(self.bond_length[:,2]/self.lattice_constant)
        #find the minimum bin, then set the left side as bond_first_neighbour
        i_max=self._bins.size
        i=0
        if method=="local_compare_method":
            while i < i_max-3:#i start from 0, which have to -1;compares i,i+1 and i+2, which have to -2, hence -3
                #np.where( self.bond_sorted[:]>res.bins[i] & self.bond_sorted[:]< res.bins[i+1] ) 
                #print(self._count[i])
                if self._count[i] > self._count[i+1]:
                    if self._count[i+1] <= self._count[i+2]:
                        if self._bins[i+1] > 1/self.lattice_constant:
                            # bond_length should be larger than sigma of particle
                            i+=1
                            break
                i+=1

        elif method=="global_compare_method":
            R"""
            scan the histogram of bond length distribution, get the highest two columns.
            see the column whose bond length is shorter as 'first peak'.
            search for local minima right after the 'first peak'.
            """
            self._count = np.array(self._count)
            self.count_sorted=np.sort(self._count)
            """
            print(self.count_sorted)
            print(self.count_sorted[-1])#max1
            print(self.count_sorted[-2])#max2
            """
            i_bins_for_count_peak_1 = np.where(self._count[:]==self.count_sorted[-1])
            i_bins_for_count_peak_2 = np.where(self._count[:]==self.count_sorted[-2])
            i_bins_for_count_peak_1 = i_bins_for_count_peak_1[0]
            i_bins_for_count_peak_2 = i_bins_for_count_peak_2[0]
            #if there are more than one bins share the same count, select the smallest bin number.
            if np.shape(i_bins_for_count_peak_1)[0]>1:
                i_bins_for_count_peak_1 = min(i_bins_for_count_peak_1)
            if np.shape(i_bins_for_count_peak_2)[0]>1:
                i_bins_for_count_peak_2 = min(i_bins_for_count_peak_2)
            i_bins_for_count_1st_peak = min(i_bins_for_count_peak_1,i_bins_for_count_peak_2)
            i_bins_for_count_1st_peak = int(i_bins_for_count_1st_peak)#force i be a simple data_type(int)
            """
            print(i_bins_for_count_peak_1) 
            print(i_bins_for_count_peak_2)
            print(i_bins_for_count_1st_peak) 
            """
            #list_xy = np.logical_and(list_x,list_y)
            #self.edge_cut_positions_list = np.where(==)
            i = i_bins_for_count_1st_peak
            while i < i_max-3:#i start from 0, which have to -1;compares i,i+1 and i+2, which have to -2, hence -3
                if self._count[i] > self._count[i+1]:
                    if self._count[i+1] <= self._count[i+2]:
                        if self._bins[i+1] <=  self._bins[i+3]:#even if jump, i+1 is still the minimum.edit for bug in index4323_8_2000
                            if self._bins[i+1] > 1/self.lattice_constant:
                                # bond_length should be larger than sigma of particle
                                i+=1
                                break
                i+=1
            R"""
                scan the histogram of bond length distribution, get the highest two columns which are not neighbors.
                see the two columns as the first and second peak of bond length.
                search for the minimum between the two peaks.
            """
            """
            elif method=="first_two_peak_group_method":
                
                self._count = np.array(self._count)
                self.count_sorted=np.sort(self._count)
                group = np.zeros(np.shape(self._count))
                ncount = np.shape(self.count_sorted)[0]
                group_rank = 0#count the num of groups
                for i in range(ncount):
                    index1 = ncount-1-i
                    i_bins_for_count_peak_1 = np.where(self._count[:]==self.count_sorted[index1])
                    i_bins_for_count_peak_1 = i_bins_for_count_peak_1[0]
                    c1 = group[i_bins_for_count_peak_1]==0
                    c0 = group[i_bins_for_count_peak_1-1]==0
                    c2 = group[i_bins_for_count_peak_1+1]==0
                    if c1 and c0 and c2 :
                        group_rank = group_rank + 1
                        group[i_bins_for_count_peak_1] = group_rank
                    elif not c1:
                        print('error: bins are scaned twice!')
                        break
                    elif not c2:
            """
        else:
            print("Err: input 'local_compare_method' or 'global_compare_method', please!")
            print("x")
            
        self.bond_first_minima_left=self._bins[i]
        self.bond_first_neighbour=self.bond_sorted[np.where(self.bond_sorted[:]<self.bond_first_minima_left)]
        
        self.bond_length_median = np.around( np.median(self.bond_first_neighbour),2)
        self.bond_length_std = np.around(np.std(self.bond_first_neighbour),2)

        if not png_filename  is None:
            #plot divide line
            plt.plot([self.bond_first_minima_left,self.bond_first_minima_left],[0,self._count[i]],linestyle='--')
            plt.title("1st peak:"+str( self.bond_length_median )+"+-"+str( self.bond_length_std )+x_unit)
            plt.xlabel("bond length "+x_unit)
            plt.ylabel("count (1)")
            #plt.show()
            #png_filename=prefix+"bond_hist_index"+str(index_num)
            plt.savefig(png_filename)
        plt.close()
        #let bond_first_minima_left be a absolute length, not normalized one
        self.bond_first_minima_left=self.bond_first_minima_left*self.lattice_constant
    
    def get_first_minima_ridge_length_distribution(self,hist_cutoff=5,
                                    method="median_minima_method", png_filename=None,x_unit='(1)'):
        R"""  
        Introduction:
            It's likely that bonds whose voronoi ridges are extremely small are not bonded. 
        Parameters:
            lattice_constant: particle diameter or lattice constant is recommended(also the unit A in x-axis in hist)
            
            method:"local_compare_method" to get the 1st minima through comparing local minima,
                suitable for continuous peak-valley hist; given here exit several local minima, the method has been given up.
                "global_compare_method" to get the 1st minima through comparing all the peaks, 
                selecting the 1st main peak(ignoring tiny peak before which), 
                finding the 1st main minima just after the 1st main peak, 
                suitable systems with powerful perturbation.
            
            hist_cutoff: plot hist of ridge_length till n times lattice_constant where n is hist_cutoff.
            
            png_filename="prefix/bond_hist_index1512"

            bond_first_minima_left:the upper limit of 1st neighbour bonds comming from the first minima 
                                    of bond length distribution, with sigma as lower limit.
            
        Warning:
            [x]What if there is no minima found? 
            count_1st_max / count_1st_minima > 10 used to check the effective of minima? 

        Examples:
            
        """
        #locate the 1st minima of bond-length distribution
        plt.figure()
        count_bins=plt.hist(self.voronoi.ridge_length[:],bins=20,range=[0,hist_cutoff])#
        _count=count_bins[0]
        _bins=count_bins[1]
        ridge_sorted=np.sort(self.voronoi.ridge_length[:])
        #find the minimum bin, then set the left side as bond_first_neighbour
        i_max=_bins.size
        i=0
        """1st half minimum count"""
        if method=="local_compare_method":
            while i < i_max-3:#i start from 0, which have to -1;compares i,i+1 and i+2, which have to -2, hence -3
                #np.where( self.bond_sorted[:]>res.bins[i] & self.bond_sorted[:]< res.bins[i+1] ) 
                #print(self._count[i])
                if _count[i] > _count[i+1]:
                    if _count[i+1] <= _count[i+2]:
                        i+=1
                        break
                i+=1 
        elif method=="median_minima_method":
            R"""
            n_ridges_half: it is believed that extremely short ridges are rare, 
            so the total number of short ridges should be less than half the num of ridges.
            hence n_ridges_half is set as a limit.
            """
            n_ridges_half = int(sum(_count)/2) 
            #i=1
            count_ridges_half = _count[0]
            count_min_i = 0
            while i < i_max-3:
                if count_ridges_half<n_ridges_half:
                    count_ridges_half = count_ridges_half + _count[i+1]
                else:
                    i=count_min_i#+1
                    break
                if _count[count_min_i] > _count[i+1]:
                    count_min_i = i+1
                i+=1

        elif method=="global_compare_method":
            _count = np.array(_count)
            count_sorted=np.sort(_count)
            """
            print(count_sorted)
            print(count_sorted[-1])#max1
            print(count_sorted[-2])#max2
            """
            i_bins_for_count_peak_1 = np.where(_count[:]==count_sorted[-1])
            i_bins_for_count_peak_2 = np.where(_count[:]==count_sorted[-2])
            i_bins_for_count_peak_1 = i_bins_for_count_peak_1[0]
            i_bins_for_count_peak_2 = i_bins_for_count_peak_2[0]
            #if there are more than one bins share the same count, select the smallest bin number.
            if np.shape(i_bins_for_count_peak_1)[0]>1:
                i_bins_for_count_peak_1 = min(i_bins_for_count_peak_1)
            if np.shape(i_bins_for_count_peak_2)[0]>1:
                i_bins_for_count_peak_2 = min(i_bins_for_count_peak_2)
            i_bins_for_count_1st_peak = min(i_bins_for_count_peak_1,i_bins_for_count_peak_2)
            i_bins_for_count_1st_peak = int(i_bins_for_count_1st_peak)#force i be a simple data_type(int)
            """
            print(i_bins_for_count_peak_1) 
            print(i_bins_for_count_peak_2)
            print(i_bins_for_count_1st_peak) 
            """
            #list_xy = np.logical_and(list_x,list_y)
            #edge_cut_positions_list = np.where(==)
            i = i_bins_for_count_1st_peak
            while i < i_max-3:#i start from 0, which have to -1;compares i,i+1 and i+2, which have to -2, hence -3
                if _count[i] > _count[i+1]:
                    if _count[i+1] <= _count[i+2]:
                        i+=1
                        break
                i+=1
        else:
            print("Err: input 'local_compare_method' or 'global_compare_method', please!")
            print("x")
        
        #check the length of ridge
        R"""
        short_ridge_rate: the variable is set to ensure that the ridge_length is not popular. 
        """
        n_ridges = sum(_count)
        short_ridge_rate = _count[i-1]/n_ridges
        if short_ridge_rate<0.1:#short ridge must be rare
            self.ridge_first_minima_left=_bins[i]
        else:
            self.ridge_first_minima_left=_bins[1]

        if not png_filename  is None:
            #plot divide line
            plt.plot([self.ridge_first_minima_left,self.ridge_first_minima_left],[0,_count[i]],linestyle='--')
            plt.title("1st minima:"+str( self.ridge_first_minima_left )+x_unit)
            plt.xlabel("ridge length "+x_unit)
            plt.ylabel("count (1)")
            plt.savefig(png_filename)
        plt.close()

    def scan_conditional_bonds_and_simplices_ml(self,png_filename=None,mode='global_scan'):
        ml = polygon_analyzer_ml(self.voronoi,self.ridge_first_minima_left)
        ml.scan_conditional_bonds_and_simplices_ml(mode,png_filename)

    def get_conditional_bonds_and_simplices_vertex_length(self,ridge_first_minima_left=None):#long_bond_cutoff=6,
        R"""
        return:
            vertex_bonds_index: n rows of [start_point_index, end_point_index]
            list_simplex_cluster: n_vertices rows of [simplex_id, cluster_id],
                where delaunay.simplices[simplex_id], 
                and cluster_id is the cluster the simplex belonging to
        method:
            vertices cluster method: find the shortest ridges and link the related vertices.
        """
        if ridge_first_minima_left is None:
            ridge_first_minima_left = self.ridge_first_minima_left
        list_short_ridge_bool = self.voronoi.ridge_length[:] <= ridge_first_minima_left
        list_short_bonds = self.voronoi.ridge_points[np.logical_not(list_short_ridge_bool)]#[start_point_index, end_point_index]
        """
        select_short_bonds_bool = self.bond_length[list_short_bonds[:,0],2] < long_bond_cutoff
        #select_short_bonds_bool = self.bond_length[list_short_bonds_index,2] < long_bond_cutoff
        list_short_bonds = list_short_bonds[select_short_bonds_bool]
        """
        list_ridge_vertices_index = self.voronoi.ridge_vertices[list_short_ridge_bool]
        list_vertices_index = np.unique(list_ridge_vertices_index)
        n_vertices = np.shape(list_vertices_index)[0]
        '''
        method2:
        record_vertices_cluster = np.ones((n_vertices,10))
        record_vertices_cluster[:]=-record_vertices_cluster[:]
        '''
        #record_vertices_cluster: n_vertices rows of [vertex_id, cluster_id],
        #where vertex_id belongs to list_vertices
        #when cluster_id=-1, the vertex has not been included by any cluster.
        record_vertices_cluster = np.ones((n_vertices,2))
        record_vertices_cluster[:]=-record_vertices_cluster[:]
        record_vertices_cluster[:,0]=list_vertices_index
        cluster_id = 0
        for i in range(n_vertices):
            vertex_id = list_vertices_index[i]
            #print("cluster_id\n",cluster_id)
            list_linked = np.where(list_ridge_vertices_index[:,:]==vertex_id)
            list_linked_vertices_pair = list_ridge_vertices_index[list_linked[0]]
            #print("vertices_pair\n",list_linked_vertices_pair)
            list_vertex_id_of_cluster1 = np.unique(list_linked_vertices_pair)
            for j in list_vertex_id_of_cluster1:
                record_id = record_vertices_cluster[:,0]==j#find the row_index for vertex_id
                if record_vertices_cluster[record_id,1]==-1:
                    record_vertices_cluster[record_id,1]=cluster_id
                elif record_vertices_cluster[record_id,1]==cluster_id:
                    pass
                else:
                    #new cluster merge old one by refreshing a list of vertices in old cluster
                    contradictory_cluster_id = record_vertices_cluster[record_id,1]
                    list_bool_of_cluster_to_merge= (record_vertices_cluster[:,1]==contradictory_cluster_id)
                    record_vertices_cluster[list_bool_of_cluster_to_merge,1]=cluster_id
            cluster_id +=1
        #statistics for clusters
        list_cluster_id = np.unique(record_vertices_cluster[:,1])
        count_polygon = np.zeros((20,2))#(10,2)
        for i in list_cluster_id:
            list_cluster_i = record_vertices_cluster[record_vertices_cluster[:,1]==i,0]
            n_complex_i = np.shape(list_cluster_i)[0]
            count_polygon[n_complex_i,0]=n_complex_i+2#[x]frame=1421,polygon=10;1609,12
            count_polygon[n_complex_i,1]+=1
        count_polygon[1,0]=3
        count_polygon[1,1]=np.shape(list_short_bonds)[0]#[x]may be wrong, n_bond != n_simplex
        #print("polygon_n, count\n",count_polygon)
        #
        count_polygon_relative = np.array(count_polygon)
        count_polygon_relative[:,1] = count_polygon_relative[:,1]/sum(count_polygon_relative[:,1]) \
                                        *(count_polygon_relative[:,0]-2)*100#see one simplex as one weight
        #print("polygon_n, count%\n",count_polygon_relative)
        """
        nm = np.maximum(count_polygon[:,1])
        np.shape(list_cluster_id)
        list_polygon_simplex = -np.ones((10,nm))
        for i in list_cluster_id:
            list_cluster_i = record_vertices_cluster[record_vertices_cluster[:,1]==i,0]
            list_polygon_simplex[]
        """
       
        self.vertex_bonds_index = list_short_bonds#short bonds index offered by ridge comapre method 
        self.list_simplex_cluster = record_vertices_cluster
        return count_polygon_relative
    
    def get_conditional_bonds_and_simplices_bond_length(self,bond_first_minima_left=None):
        R"""
        return:
            self.list_simplex_cluster: n_vertices rows of [simplex_id, cluster_id],
                where delaunay.simplices[simplex_id], 
                and cluster_id is the cluster the simplex belonging to.
                if cluster_id = -1, the simplex is on boundary and should not be drawn.
            count_polygon_relative: [polygon_n,share of total n of simplices]
        method:
            vertices cluster method: find the shortest ridges and link the related vertices.
        """
        if bond_first_minima_left is None:
            bond_first_minima_left = self.bond_first_minima_left
        list_long_bond_bool = self.bond_length[:,2] >= bond_first_minima_left
        list_short_bonds = self.voronoi.ridge_points[np.logical_not(list_long_bond_bool)]#[start_point_index, end_point_index]
        #print(self.delaunay.simplices)#(n_simplex,3)[index_vertex1,index_vertex2,index_vertex3]
        sz = self.delaunay.simplices.shape
        list_simplex_cluster = np.zeros((sz[0],2),dtype=int)
        list_simplex_cluster[:,0] = range(sz[0])
        list_simplex_cluster[:,1] = range(sz[0])#initialize each simplex as independent cluster
        list_simplices_points = np.array(self.delaunay.simplices,dtype=int)
        n_bond = len(list_long_bond_bool)
        for i in range(n_bond):
            if list_long_bond_bool[i]:
                index_p1 = int(self.bond_length[i,0])
                index_p2 = int(self.bond_length[i,1])
                loc1 = np.where(list_simplices_points == index_p1)#list_simplices with p1
                loc2 = np.where(list_simplices_points == index_p2)#list_simplices with p2
                #print(i,loc1[0],loc2[0])
                #get the simplices with p1 and p2
                list_simplices_with_p = np.concatenate((loc1[0],loc2[0]))
                list_simplices_with_p = np.sort(list_simplices_with_p)
                list_simplices_with_p_unique = np.unique(list_simplices_with_p)
                count_cobond = np.zeros((len(list_simplices_with_p_unique),),dtype=int)
                for j in range(len(list_simplices_with_p_unique)):
                    loc_repeat = np.where(list_simplices_with_p==list_simplices_with_p_unique[j]) #
                    n_loc_with_p = len(loc_repeat[0])
                    if n_loc_with_p == 2:
                        count_cobond[j] = n_loc_with_p
                        #print(list_simplices_with_p_unique[j],count_cobond[j])#index_simplices, n_repeat
                        #count_cobond = count_cobond + 1
                list_simplices_cobond = np.where(count_cobond==2)
                n_simplices_cobond = len(list_simplices_cobond[0])
                """if n_simplices_cobond==1:
                    #ban a simplex by set whose cluster_id as -1
                    list_boundary_simplex = list_simplices_with_p_unique[list_simplices_cobond]
                    list_simplex_cluster[list_boundary_simplex,1] = -1"""
                if n_simplices_cobond==2:
                    #merge linked simplices into a cluster whose id is smaller.
                    list_boundary_simplex = list_simplices_with_p_unique[list_simplices_cobond]
                    cluster_id_min = min(list_simplex_cluster[list_boundary_simplex,1])
                    cluster_id_max = max(list_simplex_cluster[list_boundary_simplex,1])
                    list_simplex_cluster[list_simplex_cluster[:,1] == cluster_id_max,1] = cluster_id_min

                #print(list_simplices_with_p)
        self.list_simplex_cluster = list_simplex_cluster

        #get the distribution of cluster
        list_cluster_id = np.unique(list_simplex_cluster[:,1])
        count_polygon = np.zeros((99,2))#(10,2)
        for cluster_id in list_cluster_id:
            if not cluster_id == -1:
                list_simplex_ids_in_cluster_i = list_simplex_cluster[list_simplex_cluster[:,1]==cluster_id,0]
                n_simplex_in_cluster_i = np.shape(list_simplex_ids_in_cluster_i)[0]
                cluster_i_is_polygon_n=n_simplex_in_cluster_i+2
                count_polygon[n_simplex_in_cluster_i,0]=cluster_i_is_polygon_n#[x]frame=1421,polygon=10;1609,12
                count_polygon[n_simplex_in_cluster_i,1]+=n_simplex_in_cluster_i
        """p_valids = np.array([self.points[index_p1], self.points[index_p2]])
        result = self.delaunay.find_simplex(p_valids)
        if result.shape[0] == 1:
            #not show the simplex
            pass
        elif result.shape[0] > 1:
            #merge two clusters
            pass"""
        count_polygon_relative = np.array(count_polygon)
        count_polygon_relative[:,1] = count_polygon[:,1]/sum(count_polygon[:,1]) \
                                        *100#see one simplex as one weight
        return count_polygon_relative
    
    def plot_bond_ridge_rank_idea(self):
        R"""
        bond plot, bond_length_rank vs ridge_length_rank,
        to see if when bond be longer, the ridge be shorter.
        """
        pass

    def draw_polygon_patch_oop(self,fig=None,ax=None,polygon_color='r',polygon_n=6):
        R"""
        parameters:
            vertex_bonds_index: n rows of [start_point_index, end_point_index]
            list_simplex_cluster: n_vertices rows of [simplex_id, cluster_id],
                where delaunay.simplices[simplex_id], 
                and cluster_id is the cluster the simplex belonging to
                (from get_conditional_bonds_and_simplices())
            is_polygon_n: mark the num n of edges of the polygon.
        
        """
        if ax is None:
            fig,ax = plt.subplots()

        vertices_cluster = self.list_simplex_cluster
        points = self.points

        list_cluster_id = np.unique(vertices_cluster[:,1])
        #patches=[]
        for cluster_id in list_cluster_id:
            if cluster_id==-1:
                pass
            else:
                list_simplex_ids_in_cluster_i = vertices_cluster[vertices_cluster[:,1]==cluster_id,0]
                n_simplex_in_cluster_i = np.shape(list_simplex_ids_in_cluster_i)[0]
                cluster_i_is_polygon_n=n_simplex_in_cluster_i+2
                #print(cluster_i_is_polygon_n)
                if cluster_i_is_polygon_n==polygon_n:
                    #patches=[]
                    for simplex_id in list_simplex_ids_in_cluster_i.astype(int):
                        list_points_ids_in_simplex = self.delaunay.simplices[simplex_id]
                        list_points_xy = points[list_points_ids_in_simplex]
                        """
                        polygon = Polygon(list_points_xy, closed=True)
                        polygon.set(color='r')
                        patches.append(polygon)
                        p = PatchCollection(patches)#, alpha=0.4
                        ax.add_collection(p)
                        """
                        
                        ax.fill(list_points_xy[:,0],list_points_xy[:,1],facecolor=polygon_color,edgecolor=polygon_color)
                    #plt.show()
                        
        """
        #rearrange points anti-clockwise
        """
        return fig,ax

    def get_first_minima_radial_distribution_function(self,rdf,lattice_constant=3.0,png_filename=None):#[x]
        #print(rdf.bin_centers) print(rdf.bin_counts)
        self.lattice_constant=lattice_constant
        #locate the 1st minima of bond-length distribution
        self._count=rdf.bin_counts
        self._bins=rdf.bin_centers
        self.bond_sorted=np.sort(self.bond_length[:,2]/self.lattice_constant)
        #find the minimum bin, then set the left side as bond_first_neighbour
        
        "global_compare_method"
        self._count = np.array(self._count)
        self._bins = np.array(self._bins)
        self.count_sorted=np.sort(self._count)

        i_max=self._bins.size
        i=0
        """
        print(self.count_sorted)
        print(self.count_sorted[-1])#max1
        print(self.count_sorted[-2])#max2
        """
        #locate the 1st minima of bond-length distribution
        plt.figure()
        rdf.plot()
        """
        print(self.count_sorted)
        print(self.count_sorted[-1])#max1
        print(self.count_sorted[-2])#max2
        """
        i_bins_for_count_peak_1 = np.where(self._count[:]==self.count_sorted[-1])
        i_bins_for_count_peak_2 = np.where(self._count[:]==self.count_sorted[-2])
        i_bins_for_count_peak_1 = i_bins_for_count_peak_1[0]
        i_bins_for_count_peak_2 = i_bins_for_count_peak_2[0]
        #if there are more than one bins share the same count, select the smallest bin number.
        if np.shape(i_bins_for_count_peak_1)[0]>1:
            i_bins_for_count_peak_1 = min(i_bins_for_count_peak_1)
        if np.shape(i_bins_for_count_peak_2)[0]>1:
            i_bins_for_count_peak_2 = min(i_bins_for_count_peak_2)
        i_bins_for_count_1st_peak = min(i_bins_for_count_peak_1,i_bins_for_count_peak_2)
        i_bins_for_count_1st_peak = int(i_bins_for_count_1st_peak)#force i be a simple data_type(int)
        """
        print(i_bins_for_count_peak_1) 
        print(i_bins_for_count_peak_2)
        print(i_bins_for_count_1st_peak) 
        """
        #list_xy = np.logical_and(list_x,list_y)
        #self.edge_cut_positions_list = np.where(==)
        i = i_bins_for_count_1st_peak
        while i < i_max-3:#i start from 0, which have to -1;compares i,i+1 and i+2, which have to -2, hence -3
            if self._count[i] > self._count[i+1]:
                if self._count[i+1] <= self._count[i+2]:
                    if self._bins[i+1] > 1/self.lattice_constant:
                        # bond_length should be larger than sigma of particle
                        i+=1
                        break
            i+=1
            
        self.bond_first_minima_left=self._bins[i]
        self.bond_first_neighbour=self.bond_sorted[np.where(self.bond_sorted[:]<self.bond_first_minima_left)]
        
        if not png_filename  is None:
            #plot divide line
            plt.plot([self.bond_first_minima_left,self.bond_first_minima_left],[0,0.5],linestyle='--')#self._count[i]
            plt.title("1st peak:"+str( np.around( np.median(self.bond_first_neighbour),2) )+"+-"+str( np.around(np.std(self.bond_first_neighbour),2) ))
            plt.xlabel("interparticle distance (unit=A)")
            plt.ylabel("count (1)")
            #plt.show()
            #png_filename=prefix+"bond_hist_index"+str(index_num)
            plt.savefig(png_filename)
        plt.close()
        #let bond_first_minima_left be a absolute length, not normalized one
        self.bond_first_minima_left=self.bond_first_minima_left*self.lattice_constant
        
    def get_coordination_number_conditional(self,lattice_constant=3,method="bond_length_method"):
        #cut edge to remove CN012
        R"""
        Introduction:
            get_coordination_number_with given bond_length_limit [min,max].
        
            CN0 % should be 0 for all the particles must be linked by bond.
            CN1 % is likely to be edge?
            CN2 % in body(edge-cutted) shows the mechanical unstability
            CN3 % shows the proportion of honeycomb.
            CN4 % shows the proportion of kagome.
            CN6 % shows the proportion of hexagonal.
            CN5/7 % shows the proportion of disclination.
        parameters:
            lattice_constant: 3 sigma in default.
            method: 'bond_length_method','ridge_length_method'.
        Variables:
            __coordination_bond: n rows of (start_point_index, end_point_index, bond_length)
            count_coordination: 10 rows of [count]. The i-th [count_i] represents the count 
                of points whose coordination number is i, where i from 0 to 9. 
            count_coordination_ratio: count_coordination whose sum is normalized to 1.
        """
        if method == "bond_length_method":
            #check whether bond_first_minima_left exist
            try:
                bond_length_limit=self.bond_first_minima_left
            except AttributeError:
                self.get_first_minima_bond_length_distribution(lattice_constant=lattice_constant)
                bond_length_limit=self.bond_first_minima_left

            #select the bonds within the limit(bond length of 1st neighbour)
            self._bond_min=0.0
            self._bond_max=bond_length_limit
            place=np.where((self.bond_length[:,2]>self._bond_min) & (self.bond_length[:,2]<self._bond_max))
            self.__coordination_bond=self.bond_length[place,0:2]
            self.__coordination_bond=self.__coordination_bond[0]#remove a [] in [[]] structure
        elif method == "ridge_length_method":
            self.get_first_minima_ridge_length_distribution()
            self.get_conditional_bonds_and_simplices_vertex_length()
            self.__coordination_bond=self.vertex_bonds_index
        
        #self._particle_id_max=np.max(self.__coordination_bond[:])#get the largest id of particle to restrain while loop
        #print(self.__coordination_bond)

        cn_max = 12#for honeycomb, cn10 truly appears in Index4340_seed9_frame2!
        self.count_coordination=np.zeros([cn_max,1])
        
        for id in self.edge_cut_positions_list[0]:
            place=np.where(self.__coordination_bond[:]==id)
            """
            Exception has occurred: IndexError       (note: full exception trace is shown but execution is paused at: <module>)
index 10 is out of bounds for axis 0 with size 10
  File "/home/tplab/xiaotian_file/lxt_code_py/points_analysis_2D.py", line 340, in get_coordination_number_conditional
    self.count_coordination[i]+=1
  File "/home/tplab/xiaotian_file/lxt_code_py/data_analysis_cycle.py", line 1024, in save_from_gsd
    a_frame.get_coordination_number_conditional()#cut edge to remove CN012
  File "/home/tplab/xiaotian_file/lxt_code_py/workflow_part.py", line 1715, in workflow_simu_to_mysql_pin_hex_to_kagome_cycle_oop_kT
    account=account)
  File "/home/tplab/xiaotian_file/lxt_code_py/control_table_multi_thread.py", line 13, in <module> (Current frame)
    index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_cycle_oop_kT(index1=index1,lcr=lcr1,kT=kT,seed=seed)
            """
            i=len(place[0])
            #print(place[0])
            self.count_coordination[i]+=1
        #print(self.count_coordination)
        self.count_coordination_ratio=self.count_coordination/sum(self.count_coordination)
    
    def get_cairo_order_parameter(self,cn3,cn4):
        R"""
        output:
            p: order_parameter_for_cairo
        introduction:
            p_cairo = (cn3 + cn4)*min(cn3/2cn4,2cn4/cn3)
        """
        sca = show_cairo_order_parameter()
        return sca.compute_cairo_order_parameter(cn3,cn4)

    def get_radial_distribution_function_triu(self, box, dr=0.1):
        R"""
        box: [LX,LY]
        dr: resolution of rdf
        """
        num_particles = len(self.points)
        r_max = np.min(box)
        n_bins = int(r_max / dr)
        
        rdf = np.zeros(n_bins)
        bond_length_all=distance.cdist(self.points[:], self.points[:], 'euclidean')
        bond_length_all_tu=np.triu(bond_length_all,k=1)#upper triangle
        bond_length_all_dr_unit = np.divide(bond_length_all_tu,dr,dtype=int)#discrete array with dr as unit
        del bond_length_all,bond_length_all_tu
        r_temp = dr
        
        while r_temp < r_max:
            nr_temp = r_temp/dr
            inbox_positions_list,inbox_positions_bool = self.cut_edge_of_positions_by_box(self.points,[box[0]-r_temp,box[1]-r_temp],True)
            bond_length_select1 = bond_length_all_dr_unit[inbox_positions_bool]
            bond_length_select2 = bond_length_all_dr_unit[:,inbox_positions_bool]
            list_r_bool1 = bond_length_select1[:,:]==nr_temp
            list_r_bool2 = bond_length_select2[:,:]==nr_temp
            count = np.sum(np.array(list_r_bool1,dtype=int))+np.sum(np.array(list_r_bool2,dtype=int))
            n_particle_inbox = np.sum(np.array(inbox_positions_bool,dtype=int))
            n_ds = 2*np.pi*r_temp*dr*n_particle_inbox

        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                separation_vector = self.points[j] - self.points[i]#self.bond_length
                separation_distance = np.linalg.norm(separation_vector)
                
                bin_index = int(separation_distance / dr)
                
                if bin_index < n_bins:
                    rdf[bin_index] += 2  # Count both particles in the pair
                        
        # Normalize RDF
        particle_n_density = num_particles / (box[0]*box[1])
        normalization_factor = np.pi * dr**2 * particle_n_density
        rdf /= normalization_factor
        
        distances = np.arange(dr / 2.0, r_max, dr)
        
        return distances, rdf
    
    def get_radial_distribution_function(self, box, dr=0.1,r_max=None,png_filename=None,points=None,normalize_r0=None):
        R"""
        box: [LX,LY]
        dr: resolution of rdf
        points: if just trap a small box, this attribute is used to record the points in that small box.
        normalize_r0: the mean 1st neighbor distance expected, to normalize the distance.
        """
        if points is None:
            points = self.points
        num_particles = len(points)
        if r_max is None:
            r_max = np.min(box)/2
        ndr_max = int(r_max/dr)#r:0-102.4 -> 0-102
        
        data_count = ndr_max-1#ndr:1-102 -> 1-101
        rdf = np.zeros(data_count)
        bond_length_all=distance.cdist(self.points[:], self.points[:], 'euclidean')
        bond_length_all_dr_unit = np.array(bond_length_all/dr,dtype=int)#discrete array with dr as unit, 0~0.99 -> 0
        bond_length_all=None
        
        
        for i in range(data_count):
            ndr_temp = i+1
            r_temp = ndr_temp*dr
            inbox_positions_list,inbox_positions_bool = self.cut_edge_of_positions_by_box(points,[box[0]-2*r_temp,box[1]-2*r_temp],True)
            bond_length_select = bond_length_all_dr_unit[inbox_positions_bool]
            list_ndr_bool = bond_length_select[:,:]==ndr_temp
            count = np.sum(np.array(list_ndr_bool,dtype=int))
            n_particle_inbox = np.sum(np.array(inbox_positions_bool,dtype=int))
            n_ds = 2*np.pi*r_temp*dr*n_particle_inbox
            rdf[i] = count/n_ds
        particle_n_density_global = num_particles/(box[0]*box[1])
        rdf_normalized = rdf/particle_n_density_global
        distances = np.linspace(1,data_count,data_count)*dr+0.5*dr#ndr:1-101 -> 1.5-101.5

        if not png_filename is None:
            fig,ax = plt.subplots()
            if normalize_r0 is None:
                ax.plot(distances,rdf_normalized,c='k')
            else:
                distances_n = distances/normalize_r0
                ax.plot(distances_n,rdf_normalized,c='k')
            ax.set_xlabel('r($\sigma$)')
            ax.set_ylabel('g(r)(1)')
            plt.savefig(png_filename)
            plt.close()
        
        if normalize_r0 is None:
            return distances, rdf_normalized
        else:
            return distances_n, rdf_normalized

    def search_neighbor_id(self,id):
        #cut edge to remove CN012
        R"""
        Introduction:
            input a particle id and get the ids of whose neighbors
        Variables:
            __coordination_bond: n rows of (start_point_index, end_point_index)
            place: (2,Nneighbors)[neighbor_id,column]
        """
        place=np.where(self.__coordination_bond[:]==id)
        n_nb=len(place[0])#number of neighbors
        place_nb=np.array(place,dtype=np.int32) 
        #print(place_nb)
        place_nb[1]=1-place_nb[1]
        #print(place_nb)
        id_nb=self.__coordination_bond[place_nb[0],place_nb[1]].astype(np.int32)
        return id_nb

    def get_nearest_neighbor_dispalcement(self):
        R"""
        Introduction:
            lindemann_dispalcement: r(i,t)-<r(j,t)>
        Return: 
            list_id_dxy_nb: 
                list[particle_id,num_neighbors, sum_id_neighbors,dx,dy], 
                id of center particles and their displacements relative to neighbors.
                if the particle has new neighbors, unless inter-frame time is too long to
                catch the event of breaking old bond and forming new bond, number of neighbors
                will change together with the event.  
            dict_c_nb: 
                dict{'id_c':id_nb}, center particles and their neighbors.
                To monitor the event of breaking old bond and forming new bond and hence
                get the life span of a caging state.
        """
        
        #check whether bond_first_minima_left exist
        try:
            bond_length_limit=self.bond_first_minima_left
        except AttributeError:
            self.get_first_minima_bond_length_distribution()
            bond_length_limit=self.bond_first_minima_left

        #select the bonds within the limit(bond length of 1st neighbour)
        self._bond_min=0.0
        self._bond_max=bond_length_limit
        place=np.where((self.bond_length[:,2]>self._bond_min) & (self.bond_length[:,2]<self._bond_max))
        self.__coordination_bond=self.bond_length[place,0:2]
        self.__coordination_bond=self.__coordination_bond[0]#remove a [] in [[]] structure
        
        
        for id in self.edge_cut_positions_list[0]:
            place=np.where(self.__coordination_bond[:]==id)
            n_nb=len(place[0])#number of neighbors
            place_nb=np.array(place,dtype=np.int32) 
            #print(place_nb)
            place_nb[1]=1-place_nb[1]
            #print(place_nb)
            id_nb=self.__coordination_bond[place_nb[0],place_nb[1]].astype(np.int32)
            sum_nb=np.sum(id_nb)
            #print(id_nb)#id s 
            
            #print(self.points[id_nb])
            center_nb = np.average(self.points[id_nb],axis = 0)
            dxy_id = self.points[id]-center_nb
            #print(center_nb)
            if 'list_id_dxy_nb' in locals():
                list_id_dxy_nb.append([id,n_nb,sum_nb,dxy_id[0],dxy_id[1]]) 
                dict_c_nb[str(id)] = id_nb
            else:
                list_id_dxy_nb = []
                list_id_dxy_nb.append([id,n_nb,sum_nb,dxy_id[0],dxy_id[1]])
                dict_c_nb = {str(id):id_nb}
        
        return list_id_dxy_nb,dict_c_nb

    def get_neighbor_cloud_method_voronoi(self,png_filename=None):#checked right
        R"""
        Introduction:
            place all the (voronoi) neighbor particles into a plot 
            with the same position of central particles.
        Example:
            import points_analysis_2D
            prefix="/home/tplab/hoomd-examples_0/"
            filename = prefix +'testhex3-16-8'#index2572~cairo
            obj_of_simu_index = points_analysis_2D.PointsAnalysis2D(filename=filename)
            obj_of_simu_index.get_neighbor_cloud_method_voronoi()
        """
        (indptr, indices) = self.delaunay.vertex_neighbor_vertices

        plt.figure()
        for k in self.edge_cut_positions_list[0]:#range(num):
            nbr = indices[indptr[k]:indptr[k+1]]
            plt.scatter(self.points[nbr,0]-self.points[k,0],self.points[nbr,1]-self.points[k,1],c='k',marker='.') 

        plt.axis('equal')
        plt.xlabel('x(sigma)')
        plt.ylabel('y(sigma)')
        """
        the displayed image size will be the smaller one 
        between axis limitation for xlim/ylim or data itself. 
        Hence the effective xlim/ylim should be set smaller than data.

        To ensure that xlim/ylim < data region, 
        we add physically meaningless points.
        """
        dis=4*1.5#the distance from origin to xmax or ymax
        #to ensure that xlim/ylim < data region
        plt.scatter([dis,-dis],[dis,-dis],c='k',marker='.')
        #restrict ticks to show
        #(let all the images share the same size)
        new_ticks = np.linspace(-dis,dis,int(2*dis+1))
        new_ticks = new_ticks.astype(int)
        #print(new_ticks.astype(str))
        #print((new_ticks))
        plt.xticks(new_ticks,new_ticks.astype(str))
        plt.yticks(new_ticks,new_ticks.astype(str))
        #restrict data region to calculate
        plt.xlim(-dis,dis)
        plt.ylim(-dis,dis)

        if png_filename is None:
            plt.show()
        else:
            plt.savefig(png_filename)
            plt.close()

    def get_neighbor_cloud_method_1st_minima_bond(self,png_filename=None):#checked right
        R"""
        Introduction:
            place all the (1st bond minima) neighbor particles into a plot 
            with the same position of central particles.
        Example:
            import points_analysis_2D
            prefix="/home/tplab/hoomd-examples_0/"
            filename = prefix +'testhex3-16-8'#index2572~cairo
            obj_of_simu_index = points_analysis_2D.PointsAnalysis2D(filename=filename)
            obj_of_simu_index.get_neighbor_cloud_bond_minima()
        """
        #check whether bond_first_minima_left exist
        try:
            bond_length_limit=self.bond_first_minima_left
        except AttributeError:
            self.get_first_minima_bond_length_distribution()
            bond_length_limit=self.bond_first_minima_left
        
        #find pairs of particles whose distance < bond_length_limit
        import scipy
        kdt = scipy.spatial.KDTree(self.points)
        pairs = kdt.query_pairs(bond_length_limit)
        
        plt.figure()
        #count=np.zeros((self.points.shape[0],1))
        for i,j in pairs:
            if self.edge_cut_positions_bool[i]:
                #np.np.concatenate([array1,array2]) 
                # would be useful to accelerate [x]
                #edge should be cut [x]
                plt.scatter(self.points[j,0]-self.points[i,0],self.points[j,1]-self.points[i,1],c='k',marker='.')
                #count[i,0]=count[i,0]+1
                plt.scatter(self.points[i,0]-self.points[j,0],self.points[i,1]-self.points[j,1],c='k',marker='.')
                #count[j,0]=count[j,0]+1
        
        plt.axis('equal')
        plt.xlabel('x(sigma)')
        plt.ylabel('y(sigma)')
        """
        the displayed image size will be the smaller one 
        between axis limitation for xlim/ylim or data itself. 
        Hence the effective xlim/ylim should be set smaller than data.

        To ensure that xlim/ylim < data region, 
        we add physically meaningless points.
        """
        dis=4*1.5#the distance from origin to xmax or ymax
        #to ensure that xlim/ylim < data region
        plt.scatter([dis,-dis],[dis,-dis],c='k',marker='.')
        #restrict ticks to show
        #(let all the images share the same size)
        new_ticks = np.linspace(-dis,dis,int(2*dis+1))
        new_ticks = new_ticks.astype(int)
        #print(new_ticks.astype(str))
        #print((new_ticks))
        plt.xticks(new_ticks,new_ticks.astype(str))
        plt.yticks(new_ticks,new_ticks.astype(str))
        #restrict data region to calculate
        plt.xlim(-dis,dis)
        plt.ylim(-dis,dis)
        
        #print(count)

        if png_filename is None:
            plt.show()
        else:
            #example: "neighbor_cloud_1st_minima_index1369.png"
            plt.savefig(png_filename)
            plt.close()

    def draw_bonds_conditional_bond_cloud(self,check=[0.9,2.0],png_filename=None):
        R"""
        Parameters:
            png_filename: "prefix/bond_plot_index1513"
        Examples:
            import points_analysis_2D as pa
            s = "/home/tplab/Downloads/"
            index_num = 1387
            index_name = "index"+str(index_num)
            fname = s+index_name
            bb = pa.PointsAnalysis2D(filename=fname)
            oname1 = s+"bond_hist_"+index_name
            bb.draw_bond_length_distribution_and_first_minima(png_filename=oname1)
            oname2 = s+"bond_plot_"+index_name
            bb.draw_bonds_conditional_bond(check=[0.9, bb.bond_first_minima_left], png_filename=oname2)
        """
        
        #add lines for edges
        for i in range(np.shape(self.bond_length)[0]):
            if (self.bond_length[i,2] > check[0])&(self.bond_length[i,2] < check[1]) :
                edge = self.bond_length[i,0:2].astype(int)
                pt1,pt2 = [self.points[edge[0]],self.points[edge[1]]]
                line = plt.Polygon([pt1,pt2], closed=None, fill=None, edgecolor='b')
                plt.gca().add_line(line)
        #plt.show()
        plt.title("bond_length:"+str(np.around(check,2))+"um")
        if not png_filename is None:
            plt.savefig(png_filename)
        #plt.close()

    def __cut_edge_of_positions(self,effective_lattice_constant = 3):
        R"""
        Variables:
            effective_lattice_constant： the distance between a particle and its neighbor
            edge_cut_positions_list: list the rows of particles' positions at given snapshot.
        """
        a = effective_lattice_constant
        #sz = len(self.init_positions)#np.size
        #xy = self.init_positions
        xmax = max(self.points[:,0])
        ymax = max(self.points[:,1])
        xmin = min(self.points[:,0]) 
        ymin = min(self.points[:,1])
        xmax = xmax - a
        ymax = ymax - a
        xmin = xmin + a
        ymin = ymin + a
        
        #That directly using 'and' has been banned, so 'logical_and' is necessary
        list_xmin = self.points[:,0] > xmin
        list_xmax = self.points[:,0] < xmax
        list_ymin = self.points[:,1] > ymin
        list_ymax = self.points[:,1] < ymax
        list_x = np.logical_and(list_xmin,list_xmax)
        list_y = np.logical_and(list_ymin,list_ymax)
        list_xy = np.logical_and(list_x,list_y)

        self.edge_cut_positions_list = np.where(list_xy)
        self.edge_cut_positions_bool = list_xy # T for body, F for edge.
    
    def cut_edge_of_positions_by_xylimit(self,xmin,xmax,ymin,ymax):
        R"""
        Variables:
            xmin,xmax,ymin,ymax: seen as vertices to form a box. select the points inside the box.
            edge_cut_positions_list: list the rows of particles' positions at given snapshot.
        """
        
        #That directly using 'and' has been banned, so 'logical_and' is necessary
        list_xmin = self.points[:,0] >= xmin
        list_xmax = self.points[:,0] < xmax
        list_ymin = self.points[:,1] >= ymin
        list_ymax = self.points[:,1] < ymax
        list_x = np.logical_and(list_xmin,list_xmax)
        list_y = np.logical_and(list_ymin,list_ymax)
        list_xy = np.logical_and(list_x,list_y)

        self.edge_cut_positions_list = np.where(list_xy)
        self.edge_cut_positions_bool = list_xy # T for body, F for edge.

    def cut_edge_of_positions_by_box(self,points,box,temp=False):
        R"""
        Variables:
            points:n rows of [x,y]
            box:[LX,LY,LZ,RX,RY,RZ]
            inbox_positions_list: self.
            inbox_positions_bool: self.
        """
        #sz = len(self.init_positions)#np.size
        #xy = self.init_positions
        xmax = box[0]/2.0
        ymax = box[1]/2.0
        xmin = -box[0]/2.0
        ymin = -box[1]/2.0
        
        #That directly using 'and' has been banned, so 'logical_and' is necessary
        list_xmin = points[:,0] >= xmin
        list_xmax = points[:,0] < xmax
        list_ymin = points[:,1] >= ymin
        list_ymax = points[:,1] < ymax
        list_x = np.logical_and(list_xmin,list_xmax)
        list_y = np.logical_and(list_ymin,list_ymax)
        list_xy = np.logical_and(list_x,list_y)

        #get the position of vertices of box
        position_box = np.zeros((5,2))
        position_box[0] = [xmin,ymin]
        position_box[1] = [xmax,ymin]
        position_box[2] = [xmax,ymax]
        position_box[3] = [xmin,ymax]
        position_box[4] = [xmin,ymin]#back to origin
        self.position_box = position_box

        if temp:
            return np.where(list_xy),list_xy
        else:
            self.inbox_positions_list = np.where(list_xy)
            self.inbox_positions_bool = list_xy # T for body, F for edge.

    def get_bond_orientational_order_selected(self,k_set=6,lower_limit=0.9):
        R"""
        Introduction:
            get psi_k ratio
        Parameters:
            k_set: set k-fold to calculate psi_k.
            lower_limit: those particles whose psi_k are larger than lower_limit 
                will be chosen as numerator.
        """
        #check whether Psi_k_abs exist
        try:
            pk=self.Psi_k_abs
        except AttributeError:
            self.get_bond_orientational_order(k_set=k_set)
            pk=self.Psi_k_abs

        place=np.where(pk[:]>lower_limit)
        sp_3=np.shape(pk[place])
        sp_all=np.shape(pk)
        self.Psi_k_rate=sp_3[0]/sp_all[0]

    def draw_bonds_test(self,fignum=1,show=False):
        #draw a figure with bonds
        plt.figure(fignum)
        plt.scatter(self.points[:,0],self.points[:,1],color='k')
        #add lines for edges
        for edge in self.bond_length[:,0:2].astype(int):
            #print(edge)
            pt1,pt2 = [self.points[edge[0]],self.points[edge[1]]]
            #plt.gca().add_line(plt.Line2D(pt1,pt2))
            line = plt.Polygon([pt1,pt2], closed=None, fill=None, edgecolor='b')
            plt.gca().add_line(line)
        if show:
            plt.show()

    def draw_bonds_conditional_bond(self,check=[0.9,2.0],png_filename=None,nb_change=None,x_unit='(um)',
                                    show_traps=False,LinearCompressionRatio=0.79,
                                    trap_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12-part1"):
        R"""
        Parameters:
            png_filename: "prefix/bond_plot_index1513"
            nb_change: particle ids which change neighbors.
            trap_filename:
                '/home/remote/hoomd-examples_0/testhoneycomb3-8-12'
                '/home/remote/hoomd-examples_0/testhoneycomb3-8-12-part1'
                '/home/remote/hoomd-examples_0/testkagome3-11-6'
                '/home/remote/hoomd-examples_0/testkagome_part3-11-6'
        Examples:
            import points_analysis_2D as pa
            s = "/home/tplab/Downloads/"
            index_num = 1387
            index_name = "index"+str(index_num)
            fname = s+index_name
            bb = pa.PointsAnalysis2D(filename=fname)
            oname1 = s+"bond_hist_"+index_name
            bb.draw_bond_length_distribution_and_first_minima(png_filename=oname1)
            oname2 = s+"bond_plot_"+index_name
            bb.draw_bonds_conditional_bond(check=[0.9, bb.bond_first_minima_left], png_filename=oname2)
        """
        bond_check= [check[0],check[1]]
        
        #if dis is None:
        xmax = max(self.points[:,0]) #- 3
        ymax = max(self.points[:,1]) #- 3
        xmin = min(self.points[:,0]) #+ 3
        ymin = min(self.points[:,1]) #+ 3
        dis = min(xmax-xmin,ymax-ymin)/2.0#half of the length of the system.
        dis = dis - 0 #cut the edge if necessary(eg. size & scale of images not match)

        center = [(xmax+xmin)*0.5,(ymax+ymin)*0.5]
        center_origin_distance = np.abs(np.dot(center,center))
        if  center_origin_distance < 1.0:# center is really close to origin
            center = [0,0]

        #draw a figure with edges
        plt.figure()
        plt.axis('equal')
        plt.xlabel('x'+x_unit)
        plt.ylabel('y'+x_unit)
        plt.title("bond_length:"+str(np.around(check,2))+x_unit)
        """
        the displayed image size will be the smaller one 
        between axis limitation for xlim/ylim or data itself. 
        Hence the effective xlim/ylim should be set smaller than data.

        To ensure that xlim/ylim < data region, 
        we add physically meaningless points.
        """
        #restrict ticks to show
        """
        #(let all the images share the same size)
        new_ticks = np.linspace(-dis,dis,int(2*dis+1))
        new_ticks = new_ticks.astype(int)
        #print(new_ticks.astype(str))
        #print((new_ticks))
        plt.xticks(new_ticks,new_ticks.astype(str))
        plt.yticks(new_ticks,new_ticks.astype(str))
        """
        #restrict data region to show
        plt.xlim(-dis+center[0],dis+center[0])
        plt.ylim(-dis+center[1],dis+center[1])
        
        #add lines for edges
        for i in range(np.shape(self.bond_length)[0]):
            if (self.bond_length[i,2] > bond_check[0])&(self.bond_length[i,2] < bond_check[1]) :
                edge = self.bond_length[i,0:2].astype(int)
                pt1,pt2 = [self.points[edge[0]],self.points[edge[1]]]
                line = plt.Polygon([pt1,pt2], closed=None, fill=None, edgecolor='b')
                plt.gca().add_line(line)

        plt.scatter(self.points[:,0],self.points[:,1],color='k')
        """
        particle_ids = np.linspace(0,self.points.shape[0]-1,self.points.shape[0],dtype=int)
        particle_ids_str = particle_ids.astype(str)
        for j in particle_ids:
            plt.annotate(particle_ids_str[j],self.points[j])
        """
        if not nb_change is None:
            plt.scatter(self.points[nb_change,0],self.points[nb_change,1],color='orange')
        
        if show_traps:
            traps=np.loadtxt(trap_filename)
            traps=np.multiply(traps,LinearCompressionRatio)
            plt.scatter(traps[:,0], 
                   traps[:,1],
                   c='r',marker = 'x')

        if not png_filename is None:
            plt.savefig(png_filename)
        plt.close()
    
    def draw_bonds_conditional_bond_oop(self,check,png_filename=None,xy_stable=None,nb_change=None,x_unit='(um)',
                                    LinearCompressionRatio=None,trap_filename=None,
                                    axis_limit=None):
        R"""
        xy(self.points) must be a frame of txyz( not txyz_stable)!
        xy_stable: a frame of txyz_stable
        LinearCompressionRatio=0.79
        trap_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12-part1"
        """
        self.draw_bonds = bond_plot_module()
        self.draw_bonds.restrict_axis_property_relative(self.points,x_unit=x_unit)
        if not axis_limit is None:
            xlim = [-axis_limit[0],axis_limit[0]]#[0,axis_limit[0]]
            ylim = [-axis_limit[1],axis_limit[1]]#[0,axis_limit[1]]
            self.draw_bonds.restrict_axis_limitation(xlim,ylim)
        self.draw_bonds.draw_points_with_conditional_bond(self.points,self.bond_length,bond_length_limmit=check)
        #self.draw_bonds.draw_bonds_conditional_bond(self.points,self.bond_length,bond_length_limmit=check,x_unit=x_unit)
        if not trap_filename is None:
            self.draw_bonds.plot_traps(trap_filename=trap_filename,LinearCompressionRatio=LinearCompressionRatio)
        if not nb_change is None:
            self.draw_bonds.plot_neighbor_change(xy_stable,nb_change)
        if not png_filename is None:
            self.draw_bonds.save_figure(png_filename)

    def draw_bonds_conditional_bond_for_image_oop(self,image,check,png_filename=None,xy_stable=None,nb_change=None,x_unit='(um)',
                                    LinearCompressionRatio=None,trap_filename=None,
                                    axis_limit=None):
        R"""
        xy(self.points) must be a frame of txyz( not txyz_stable)!
        xy_stable: a frame of txyz_stable
        LinearCompressionRatio=0.79
        trap_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12-part1"
        axis_limit=[xmin,xmax,ymin,ymax]
        """
        self.draw_bonds = bond_plot_module_for_image(image)
        self.draw_bonds.restrict_axis_property_relative(self.points,x_unit=x_unit)
        if not axis_limit is None:
            xlim = [axis_limit[0],axis_limit[1]]
            ylim = [axis_limit[2],axis_limit[3]]
            self.draw_bonds.restrict_axis_limitation(xlim,ylim)
        line = self.draw_bonds.draw_points_with_conditional_bond(self.points,self.bond_length,bond_length_limmit=check)
        #self.draw_bonds.draw_bonds_conditional_bond(self.points,self.bond_length,bond_length_limmit=check,x_unit=x_unit)
        if not trap_filename is None:
            self.draw_bonds.plot_traps(trap_filename=trap_filename,LinearCompressionRatio=LinearCompressionRatio)
        if not nb_change is None:
            self.draw_bonds.plot_neighbor_change(xy_stable,nb_change)
        if not png_filename is None:
            self.draw_bonds.save_figure(png_filename)
        return line

    def print_benchmark(self,bond=True,ridge=True):
        if bond :
            print('bond')
            print('min',min(self.bond_length[:,2]))
            print('median',np.median(self.bond_length[:,2]))
            print('max',max(self.bond_length[:,2]))
            print('\n')
        if ridge :
            print('ridge')
            print('min',min(self.voronoi.ridge_length))
            print('median',np.median(self.voronoi.ridge_length))
            print('max',max(self.voronoi.ridge_length))
            print('\n')

    def print_basic_property(self):
        R"""
        but this class can only get points and hence doesnt know energy parameters.
        [x]
        """
        #interaction
        epsilon=300
        kappa=0.25
        #substrate
        k=800.0
        rcut=1.0

        energy_interaction=epsilon*np.exp(-kappa)# when rcut=12.5, Uparticle = kBT = 1; rcut=15, Up=0.47
        energy_substrate=0.5*k*np.multiply(rcut,rcut)
        #pt=np.exp(1)
        print('energy_thermal     1')
        print('energy_interaction '+str(energy_interaction.astype(int)))
        print('energy_substrate   '+str(energy_substrate.astype(int)))
    
    def get_cluster_vertices(self):
        #check whether cutted_ridges exist 
        try:
            v_index_temp=self.voronoi.ridge_vertices[(self.voronoi.cutted_ridges).astype(int)]
        except AttributeError:
            self.draw_bonds_conditional_ridge()
            v_index_temp=self.voronoi.ridge_vertices[(self.voronoi.cutted_ridges).astype(int)]
            
        v_index_temp=v_index_temp[:,0]#v_index_temp is [[array]]
        #get index of vertices
        c=np.concatenate([v_index_temp[:,0],v_index_temp[:,1]])
        c=np.sort(c)
        shp=np.shape(c)
        for j in range(shp[0]-1):
            if c[j]==c[j+1]:
                c[j+1]=-1
        c=np.sort(c)
        for j in range(shp[0]-1):
            if not c[j]==-1:
                start=j
                break
        c=c[j:shp[0]]
        self.voronoi.cluster_vertices=c
        #plt.scatter(self.voronoi.vertices[c,0],self.voronoi.vertices[c,1],color='g')
        #plt.show()
        del c,v_index_temp

    def count_n_polygon_fast(self,n_polygon=6):
        R"""
        arguments:
            n_polygon: annotate n-polygon
        """
        #check whether cluster_vertices exist
        try:
            cv_temp=self.voronoi.vertices[self.voronoi.cluster_vertices]
        except AttributeError:
            self.get_cluster_vertices()
            cv_temp=self.voronoi.vertices[self.voronoi.cluster_vertices]
        
        shp=np.shape(cv_temp)
        cluster_count=np.zeros((shp[0]))
        check=self.voronoi.cut_ridge_length
        cv_dis=distance.cdist(cv_temp,cv_temp,'euclidean')
        
        for i in range(shp[0]):
            '''
            if cv_dis[i,0]==255:
                continue
            else:
                list=np.where(cv_dis[i,:]<check)
                print(list[0])
                #cv_dis[:,list]=255#delete found vertices 
                cv_dis[list,:]=255#delete found vertices 
                count=np.shape(list[0])
                print(cv_dis[0:5,0:5])
                count=count[0]#how many vertices in a cluster
                cluster_count[count]=cluster_count[count]+1
            '''
            list=np.where(cv_dis[i,:]<1.0*check)
            cv_dis[:,list]=255#delete found vertices 
            cv_dis[list,:]=255#delete found vertices 
            count=np.shape(list[0])
            count=count[0]#how many vertices in a cluster
            cluster_count[count]=cluster_count[count]+1
            if count==(n_polygon-2):
                plt.annotate(cluster_count[count].astype(int), (cv_temp[i,0], cv_temp[i,1]))
        """
        #some great polygons whose vertices linked closely(small dis),
        # while twisted polygons whose vertices are farther from each other 
        # than from different vertices clusters.
        # so an algorithm with more precision is needed 
        """
        print('Tetragon,Pentagon,Hexagon...')
        print(cluster_count[2:10])
        
    def count_polygon_n(self,n_edges):   
        R"""
        Introduction:
            a polygon with n edges must be merged by n-2 triangles, hence n-3 bonds must be removed. 
            That is, a quadrilateral contains 2 triangles with 1 bond removed.
        parameters:
            points: n rows of (x,y)
            --------------------------------
            delaunay: scipy.spatial.Delaunay
            bond_length: n rows of (start_point_row, end_point_row, bond_length)
            cutted_bonds: n rows of (start_point_row, end_point_row)
            simplices: n rows of (point_row_1, point_row_2, point_row_3) vertices of triangles
            --------------------------------
            linked_triangles: 
        """
        #check whether cutted_bonds exist
        if not ('self.cutted_bonds' in dir()):#if hasattr(self,'box')
            self.draw_bonds_conditional_bond_oop()
        #get linked triangles pair   
        shp=np.shape(self.cutted_bonds)
        self.linked_triangles = np.zeros(shp)
        for i in range(shp[0]):
            #1st round search
            index1=np.where(self.delaunay.simplices[:]==self.cutted_bonds[i,0])
            #2nd round search
            index2=np.where(self.delaunay.simplices[index1[0]]==self.cutted_bonds[i,1])
            temp=index1[0]
            self.linked_triangles[i]=temp[index2[0]]
        
        #get linked triangles chain
        chain=np.zeros((shp[0],8))#up to 10-polygon
        chain[:]=-1
        all_chain_count=np.zeros((shp[0],1))
        chain_num=1#now we are proceeding the chain_num-th chain [x]unfinished
        tri_count_max=shp[0]#at most the triangle pairs should be linked
        lt_temp=self.linked_triangles[:]#np.zeros((shp[0],2))

        i=0#if all the triangle pairs are read, break the loop
        while sum(all_chain_count)<tri_count_max:
            if lt_temp[i,0]==-1:
                #this triangle pair has been linked
                i=i+1
                continue
            else:
                #init a new chain
                this_chain_count=1
                lt_temp[i,:]=-1#delete the head triangle pair to avoid being searched
                chain[chain_num,this_chain_count-1]=self.linked_triangles[i,0]
                chain[chain_num,this_chain_count]=self.linked_triangles[i,1]
                #search linked triangles
                index1=np.where((lt_temp[:,0]==self.linked_triangles[i,1])|(lt_temp[:,1]==self.linked_triangles[i,1]))#rightward
                index_check1=np.shape(index1[0])
                if index_check1[0]==0:
                    print('error,the triangle pair rightward not found') 
                elif index_check1[0]==1:
                    #find triangles
                    if index1[1]==0:
                        self.linked_triangles[index1[0],1]
                        lt_temp[index1[0],:]=-1
                        this_chain_count=this_chain_count+1

                    elif index1[1]==1:#
                        self.linked_triangles[index1[0],0]
                        lt_temp[index1[0],:]=-1
                        this_chain_count=this_chain_count+1

                    else:
                        print('error')
                            
                elif index_check1[0]==2:
                    #two braches
                    for j in index1[0]:
                        if lt_temp[j,0]== self.linked_triangles[i,1]:
                            pass 

                else:
                    print('error')
                    
                #leftward
            if i==tri_count_max:
                #if all the triangle pairs are read, break the loop
                break
    
        '''
        for i in range(shp[0]):
            if i==0:
                chain[0,0:2]=lt_temp[i]
                lt_temp[i]=[-1,-1]#clear the i-th triangle to avoid being found
                chain_count[count]=chain_count[count]+1#the count of chain-0 is added one
                count=count+1#now there is one more chain
            else:
                index1=np.where((chain[:]==lt_temp[i,0])|(chain[:]==lt_temp[i,1]))
                temp=index1[0]#row-index of linked triangles
                for j in temp:
                    if chain_count[j,0]:
                    lt_temp[j,1]
                #that chain[0] and chain[5] should be linked may exist

        del lt_temp,temp
        
        #https://blog.csdn.net/u013378642/article/details/81775131
        self.linked_triangles_sorted=sorted(self.linked_triangles,key=lambda x:x[0])
        print('t1\n',self.linked_triangles,'\nt2\n',self.linked_triangles_sorted)
        #a chain of triangle could be like [1,162] [162,4] [4,88.]
        '''
        pass

    def count_polygon_n_new(self,n_edges):
        R"""
        Introduction:
            a polygon with n edges must be merged by n-2 triangles, hence n-3 bonds must be removed. 
            That is, a quadrilateral contains 2 triangles with 1 bond removed.
        parameters:
            points: n rows of (x,y)
            --------------------------------
            delaunay: scipy.spatial.Delaunay
            bond_length: n rows of (start_point_row, end_point_row, bond_length)
            cutted_bonds: n rows of (start_point_row, end_point_row)
            simplices: n rows of (point_row_1, point_row_2, point_row_3) vertices of triangles
            --------------------------------
            linked_triangles:
            list_polygon_n: 8
        """
        list_polygon_n = range(10)

        return list_polygon_n
    
    def plot_polygon(self,x,y):
         fig,ax = plt.subplots()
         ax.fill(x,y,color='g')

class dynamic_points_analysis_2d:#old_class_name: msd
    R"""
    Introduction:
        this class is designed to analyze dynamic properties of a group of paritcle trajectories.
    
    Methods:
        msd_module: mean square displacements.
        displacement_field_module: displacement field.
    
    example:
        #final cut bond plot of a simulation
        import data_analysis_cycle as dac
        import points_analysis_2D as pa
        import numpy as np
        daw = dac.data_analysis_workflow()
        directory,dataname=daw.gsd_to_txyz(simu_index=5387)
        txyz = np.load('/home/remote/Downloads/5387_9/txyz.npy')
        dpa = pa.dynamic_points_analysis_2d(txyz)
        dpa.plot_bond_neighbor_change_oop(data_name=dataname,prefix=directory,final_cut=True,
                        trap_filename='/home/remote/hoomd-examples_0/testhoneycomb3-8-12-part1',
                        trap_lcr=0.816)
    """
    def __init__(self,txyz_stable,mode='simu'):
        R"""
        parameters:
            txyz_stable:array[Nframes,Nparticles,xyz],
                for simu data, 
                    ensure that particles never move across the boundary(box)!
                for exp data, 
                    unit of xyz must be um! 
                    ensure that particles are always in the vision field!
            #BOX: [lx,ly,lz,xy,xz,yz]
            account: 'remote' or 'tplab'
            mode: 'simu' or 'exp'. 'simu' to select particles in box; 'exp' to direct compute msd
        """            
        if mode == 'simu':
            #self.box = box
            #self.__select_stable_trajectories_simu(plot_trajectory)
            self.txyz_stable = txyz_stable
            self.x_unit = '($\sigma$)'
            self.t_unit = '(step)'
        elif mode == 'exp':
            self.txyz_stable = txyz_stable
            self.x_unit = '(um)'
            self.t_unit = '(s)'

        #self.account = account
        self.mode = mode

    def plot_trajectory_all_in_one(self,png_prefix=''):
        frames,particles,dimensions=self.txyz_stable.shape
        list_stable_id = range(particles)#txyz_stable.shape[1]
        plt.figure()
        for index_particle in list_stable_id:
            txyz_ith = self.txyz_stable[:,index_particle,:]
            plt.plot(txyz_ith[:,0],txyz_ith[:,1])
        plt.xlabel("$x$ "+self.x_unit )
        plt.ylabel("$y$ "+self.x_unit )
        #png_filename = '/home/'+self.account+'/Downloads/'+'traj_stable.png'
        png_filename=png_prefix+'trajectory_stable.png'
        plt.savefig(png_filename)
        plt.close() 

    def plot_trajectory_single_particle_loop(self,png_prefix=''):
        R"""
        EXP:
            path_to_results = '/home/remote/xiaotian_file/data/20221129/video_5'
            txyz_npy_filename = path_to_results+'/'+'txyz_stable.npy'
            txyz_stable = np.load(txyz_npy_filename)
            msds = pa.dynamic_points_analysis_2d(txyz_stable,mode='exp')
            msds.plot_trajectory_single_particle(path_to_results+'/')
        """
        frames,particles,dimensions=self.txyz_stable.shape
        list_stable_id = range(particles)#txyz_stable.shape[1]
        for particle_id in list_stable_id:
            txyz_ith = self.txyz_stable[:,particle_id,:]
            plt.figure()
            plt.plot(txyz_ith[:,0],txyz_ith[:,1])
            plt.xlabel("$x$ "+self.x_unit )#'(sigma)'
            plt.ylabel("$y$ "+self.x_unit )
            png_filename = 'traj_stable_'+str(int(particle_id))+'.png'
            plt.savefig(png_prefix+png_filename)
            plt.close()   

    def plot_trajectory_single_particle(self,particle_id,png_prefix=''):
        R"""
        """
        txyz_ith = self.txyz_stable[:,particle_id,:]
        plt.figure()
        plt.plot(txyz_ith[:,0],txyz_ith[:,1])
        plt.xlabel("$x$ "+self.x_unit )#'(sigma)'
        plt.ylabel("$y$ "+self.x_unit )
        png_filename = 'traj_stable_'+str(int(particle_id))+'.png'
        plt.savefig(png_prefix+png_filename)
        plt.close()   

    def plot_a_frame_of_points(self,frame_index,png_filename):
        points = self.txyz_stable[frame_index]
        fig,ax = plt.subplots()
        ax.scatter(points[:,0],points[:,1],c='k')
        plt.savefig(png_filename)

    def msd_module(self):
        R"""
        return:
            (class)self.msd
        """
        self.msd = mean_square_displacement(self.txyz_stable)
    
    def displacement_field_module(self):
        R"""
        return:
            (class)self.displacemnt_field
        """
        self.displacemnt_field = displacemnt_field_2D(self.txyz_stable)
    
    def compute_nearest_neighbor_displacements(self,csv_prefix='',bond_cut_off=6):
        R"""
        parameters:
            input: 
                trajectory_stable (unit = sigma or um)
                'stable' means all the particles in the 
                field of view from the 1st frame to the end.
                if the trajectory is not stable, a particle id 
                in different frames would not represent the same particle.

                csv_prefix: (string)the directory for csv file to save.

            return: 
                ts_id_dxy:
                    a pandas.dataframe contains ["particle_id","num_neighbors","sum_id_neighbors", "x","y", "frame"]
                    which contains all the nn displacements over time.
                    Caution: hist_cutoff should be set nearly 2 times of bond length, 
                    or the 1st minima bond length would be incorrectly recognized.
                dict_c_nb:
                    a dict() contains {'particle id': [list id of neighbors]} at given frame.
                    
        introduction:
            DOI: 10.1103/PhysRevE.95.022602
            
            nearest-neighbor relative mean square displacement(nnmsd): <[r(i,t)-<r(j,t)>]^2>
            where j is the neighbors of particle i.

            The dynamic Lindemann parameter: gamma_l_tau = nnmsd/2/a^2
        question:
            if I should calculate the particle averaged first neighbor bond length( A(t) ), 
            and calculate the averaged time bond length(<A>)? 
            dynamic lindemann parameter suits only for the time-scale when the system is stable solid.
        """
        id_dxy_pd_columns = ["particle_id","num_neighbors","sum_id_neighbors", "x","y"]
        #get frame-wise 1st neighbor bond length
        #list_1st_bond_length_framewise
        self.frames,self.particles,self.dimensions=self.txyz_stable.shape
        list_frames = range(self.frames)
        list_1st_bond_length_framewise = np.zeros((self.frames,))
        for frame in list_frames:
            result = static_points_analysis_2d(self.txyz_stable[frame,:,:])
            result.get_first_minima_bond_length_distribution(lattice_constant=1,hist_cutoff=bond_cut_off)#here lattice_constant is just used to normalize figure, hence set 2.0 is ok
            #print('recognized bond length: '+str(result.bond_length_median*lc)+'+-'+str(result.bond_length_std*lc)+' '+unit)
            list_1st_bond_length_framewise[frame] = result.bond_length_median#unit = sigma!
            #displacement neighbor
            id_dxy,dict_c_nb = result.get_nearest_neighbor_dispalcement()
            id_dxy_pd = pd.DataFrame(id_dxy)
            """
            list_id_dxy_nb: 
                list[particle_id,num_neighbors, sum_id_neighbors,dx,dy], 
                id of center particles and their displacements relative to neighbors.
                if the particle has new neighbors, unless inter-frame time is too long to
                catch the event of breaking old bond and forming new bond, number of neighbors
                will change together with the event.  
            """
            #print(id_dxy_pd.head())
            if frame == 0:
                #ts_id_dxy = t_id_dxy
                id_dxy_pd.columns = id_dxy_pd_columns
                id_dxy_pd['frame'] = frame
                ts_id_dxy = id_dxy_pd
                #print(ts_id_dxy.head())
                #print(ts_id_dxy.tail())
            else:
                #ts_id_dxy = np.concatenate((ts_id_dxy,t_id_dxy), axis=0) #[frame,id,dx,dy]
                id_dxy_pd.columns = id_dxy_pd_columns#[id,dx,dy,frame]
                id_dxy_pd['frame'] = frame
                ts_id_dxy = pd.concat([ts_id_dxy,id_dxy_pd])
                #print(ts_id_dxy.tail())
                #print(ts_id_dxy.shape)
        #select stable trajectories
        """
        print(ts_id_dxy[:,1].max)
        list_ids = np.unique(ts_id_dxy[:,1])
        id_check = np.zeros((list_ids.shape[0],),dtype=bool)
        for id in list_ids:
            ts_dxy = ts_id_dxy[ts_id_dxy[:,1]==id]

            if ts_dxy.shape[0] == self.frames:
                id_check[id] = True
        ts_id_dxy_stable = ts_id_dxy[:,id_check,:,:]
        """
        average_1st_bond_length = np.average(list_1st_bond_length_framewise) 
        #particle_tracking.select_stable_trajectory()is great!
        pd.DataFrame.to_csv(ts_id_dxy,csv_prefix+'ts_id_dxy.csv')
        return ts_id_dxy,average_1st_bond_length
    
    def monitor_neighbor_change_event(self,ts_id_dxy,csv_prefix=''):
        R"""
        input:
            ts_id_dxy: 
                a pandas.dataframe contains ["particle_id","num_neighbors","sum_id_neighbors", "x","y", "frame"]
                which contains all the nn displacements over time. 
                get from self.compute_nearest_neighbor_displacements()
            csv_prefix:
                (string)the directory for csv file to save.
        return:
            list_sum_id_nb_stable:
                (csv file)["frame", "particle_id", "sum_id_neighbors", 'if_nb_change'].
            if_nb_change_int:
                (ndarray)[Nframe_nb_stable,Nparticle_nb_stable]. 0 nb stable, 1 nb change.
            n_particle_nb_stable:(int)
        consideration:
            given edge cut in each frame, some of the particles are removed,
            check each particle_id if contains Nframes.
            count id = Nframes, then record; or delete
        """
        #scan particle_id
        #list_sum_id_nb_stable_np [Nframes, Nparticles]: sum_id_neighbors (int)
        list_sum_id_nb = ts_id_dxy[["frame", "particle_id", "sum_id_neighbors"]]
        """
        file_list_sum_id_nb = prefix + 'list_sum_id_nb.csv'
        pd.DataFrame.to_csv(list_sum_id_nb,file_list_sum_id_nb)
        """
        list_id = list_sum_id_nb['particle_id'].values
        list_id_counts = np.unique(list_id,return_counts=True) 
        list_id_counts = np.array(list_id_counts)

        list_id_nb_not_cut_edge= list_id_counts[0,list_id_counts[1,:] == max(list_id_counts[1])]#list_id_counts[1].max]
        n_frame_nb_stable = max(list_id_counts[1])
        n_particle_nb_stable = len(list_id_nb_not_cut_edge)
        
        list_sum_id_nb_stable = list_sum_id_nb[list_sum_id_nb["particle_id"].isin(list_id_nb_not_cut_edge)]
        list_sum_id_nb_stable_np = list_sum_id_nb_stable["sum_id_neighbors"].values
        list_sum_id_nb_stable_np = np.reshape(list_sum_id_nb_stable_np,(n_frame_nb_stable,n_particle_nb_stable))
        #sum_id [Nframes-1, Nparticles]: sum_id_neighbors (int)
        #sum_id_0 [0 ~ Nframes-1, Nparticles] where Nparticles are never cut by edges.
        #sum_id_1 [1 ~ Nframes, Nparticles]
        sum_id_0 = list_sum_id_nb_stable_np[:-1]
        sum_id_1 = list_sum_id_nb_stable_np[1:]
        #sum_id_diff = sum_id_1 - sum_id_0
        sum_id_diff = np.absolute(sum_id_1 - sum_id_0)
        # if sum_id_diff[:,:] != 0 then = 1
        #compare sum_id_neighbors between frame+1 and frame
        #different -> change_neighbor = true
        sum_id_diff = np.array(sum_id_diff,dtype=bool)
        if_nb_change = np.array(list_sum_id_nb_stable_np,dtype=bool)
        if_nb_change[0] = False
        if_nb_change[1:] = sum_id_diff
        if_nb_change_int = np.array(if_nb_change,dtype=int)
        #reshape if_nb_change and plug back into list_sum_id_nb_stable
        list_sum_id_nb_stable.loc[:,'if_nb_change'] = tuple(np.reshape(if_nb_change,n_frame_nb_stable*n_particle_nb_stable)) 
        file_list_sum_id_nb_stable = csv_prefix + 'list_sum_id_nb_stable.csv'
        pd.DataFrame.to_csv(list_sum_id_nb_stable,file_list_sum_id_nb_stable)
        return if_nb_change_int,n_particle_nb_stable
        
    def get_hist_neighbor_change_event(self,if_nb_change_int,n_particle_nb_stable,prefix=None):
        # histogram: frame VS count change_neighbor_events
        count_nb_change_event = np.sum(if_nb_change_int,axis=1)
        count_nb_change_event_rate = count_nb_change_event/n_particle_nb_stable
        np.save(prefix+'count_nb_change_event_rate',count_nb_change_event_rate)
        self.plot_hist_neighbor_change_event(count_nb_change_event_rate,prefix)
    
    def plot_hist_neighbor_change_event(self,count_nb_change_event_rate,prefix=None):
        # histogram: frame VS count change_neighbor_events
        fig,ax = plt.subplots()
        #ax.plot(count_nb_change_event_rate)
        #ax.semilogx(count_nb_change_event_rate)
        ax.loglog(count_nb_change_event_rate)
        ax.set_xlabel('time steps(1000)')
        ax.set_ylabel('nb_change_particles(%)')
        if not prefix is None:
            png_filename = 'hist_nb_change_loglog.png'#_semilog
            fig.savefig(prefix+png_filename)

    def plot_bond_neighbor_change(self,data_name='default_exp',prefix='',
                                    final_cut=False,nb_change=None,bond_cut_off=6,
                                    show_traps=False,trap_filename=None,trap_lcr=None):
        R"""
        Introduction:
            final_cut: true to proceed the last frame only.
            bond_cut_off: should always be set the same condition in a data_set!
        """
        #prefix='/home/'+account+'/Downloads/'#'/home/tplab/Downloads/'
        #log_prefix='/home/'+account+'/hoomd-examples_0/'#'/home/tplab/hoomd-examples_0/'
        #load time steps
        str_index=data_name
        num_of_frames = self.txyz_stable.shape[0]
        
        for i in range(num_of_frames):
            if final_cut:
                i = num_of_frames-1#i=9#!!! 23
            
            a_frame = static_points_analysis_2d(points=self.txyz_stable[i])#hide_figure=False

            if not nb_change is None:
                #num_frames = list_sum_id_nb_stable['frame'].values.max()+1
                #frames = range(num_frames)
                #for frame in frames:
                list_sum_id_nb_stable = nb_change
                snap = list_sum_id_nb_stable[list_sum_id_nb_stable['frame']==i]
                snap_part = snap[snap['if_nb_change'] == True]
                ids = snap_part["particle_id"].values
                #points_nb_chg = txyz_stable[frame,ids,:2]

            if final_cut:
                #bond_plot+trap_plot
                png_filename1 = prefix +'bond_hist_index'+str_index+'_'+str(int(i))+'.png'
                png_filename2 = prefix +'bond_plot_1st_minima_index'+str_index+'_'+str(int(i))+'.png'
            else:
                folder_name=prefix+"record_"+str_index#+"/"
                #check if the folder exists
                isExists=os.path.exists(folder_name)
                if not isExists:
                    os.makedirs(folder_name)
                #bond_plot+trap_plot
                png_filename1 = folder_name+"/" +'bond_hist_index'+str_index+'_'+str(int(i))+'.png'
                png_filename2 = folder_name+"/" +'bond_plot_1st_minima_index'+str_index+'_'+str(int(i))+'.png'
            
            a_frame.get_first_minima_bond_length_distribution(lattice_constant=1,hist_cutoff=bond_cut_off,png_filename=png_filename1)#,png_filename=png_filename1
            a_frame.draw_bonds_conditional_bond(check=[0.4, a_frame.bond_first_minima_left], png_filename=png_filename2,
                                            show_traps=show_traps,LinearCompressionRatio=trap_lcr,trap_filename=trap_filename,
                                            nb_change=ids,x_unit=self.x_unit)
        
            if final_cut:
                break
    
    def plot_bond_neighbor_change_oop(self,data_name='default_exp',prefix='',
                                    final_cut=False,init_cut=False,nb_change=None,bond_cut_off=6,
                                    trap_filename=None,trap_lcr=None):#show_traps=False,
        R"""
        Introduction:
            final_cut: true to proceed the last frame only.
            bond_cut_off: should always be set the same condition in a data_set!
        return:
            
        """
        #prefix='/home/'+account+'/Downloads/'#'/home/tplab/Downloads/'
        #log_prefix='/home/'+account+'/hoomd-examples_0/'#'/home/tplab/hoomd-examples_0/'
        #load time steps
        str_index=data_name
        num_of_frames = self.txyz_stable.shape[0]
        if self.mode == 'simu':
            txyz = np.load(prefix+'txyz.npy')
        elif self.mode == 'exp':
            txyz = self.txyz_stable
        for i in range(num_of_frames):
            if final_cut:
                i = num_of_frames-1#i=9#!!! 23
            if init_cut:
                i = 0
            a_frame = static_points_analysis_2d(points=txyz[i])#hide_figure=False

            if nb_change is None:
                ids = None
            else:
                #num_frames = list_sum_id_nb_stable['frame'].values.max()+1
                #frames = range(num_frames)
                #for frame in frames:
                list_sum_id_nb_stable = nb_change
                snap = list_sum_id_nb_stable[list_sum_id_nb_stable['frame']==i]
                snap_part = snap[snap['if_nb_change'] == True]
                ids = snap_part["particle_id"].values
                #points_nb_chg = txyz_stable[frame,ids,:2]

            if final_cut or init_cut:
                #bond_plot+trap_plot
                png_filename1 = prefix +'bond_hist_index'+str_index+'_'+str(int(i))+'.png'
                png_filename2 = prefix +'bond_plot_1st_minima_index'+str_index+'_'+str(int(i))+'.png'
                ids = None
            else:
                folder_name=prefix+"record_"+str_index#+"/"
                #check if the folder exists
                isExists=os.path.exists(folder_name)
                if not isExists:
                    os.makedirs(folder_name)
                #bond_plot+trap_plot
                png_filename1 = folder_name+"/" +'bond_hist_index'+str_index+'_'+str(int(i))+'.png'
                png_filename2 = folder_name+"/" +'bond_plot_1st_minima_index'+str_index+'_'+str(int(i))+'.png'
            
            a_frame.get_first_minima_bond_length_distribution(lattice_constant=1,hist_cutoff=bond_cut_off,png_filename=png_filename1)#,png_filename=png_filename1
            #a_frame.draw_bonds_conditional_bond(check=[0.4, a_frame.bond_first_minima_left], png_filename=png_filename2,
            #                               show_traps=show_traps,LinearCompressionRatio=trap_lcr,trap_filename=trap_filename,
            #                                nb_change=ids,x_unit=self.x_unit)
            a_frame.draw_bonds_conditional_bond_oop(check=[0.4, a_frame.bond_first_minima_left], png_filename=png_filename2,
                                                    xy_stable=self.txyz_stable[i],nb_change=ids,x_unit=self.x_unit,
                            LinearCompressionRatio=trap_lcr, trap_filename=trap_filename)
            if final_cut or init_cut:
                break

    def plot_neighbor_change_evolution(self,frame_init,frame_final,prefix='',nb_change=None,arrow=False,
                                    data_name='default_exp',ids=None,
                                    bond_cut_off=None,
                                    trap_filename=None,trap_lcr=None):#show_traps=False,):
        R"""
        input:
            arrow: 'direct' or 'annotate'
            bond_cut_off: 6 recommended(2 times lattice constant)
        return:
            ids
        plots:    
            black dots: particles of initial state.
            orange circles: particles of final state.
            green arrows: displacements between final nad initial states.
        """
        str_index=data_name
        txyz = np.load(prefix+'txyz.npy')

        xy_init = self.txyz_stable[frame_init,:,:2]
        xy_final = self.txyz_stable[frame_final,:,:2]
        particle_size=50
        circle_size = particle_size
        lw = 2#circle linewidths
        circle_color = 'limegreen'#'orange'
        arrow_color = 'limegreen'

        bpm = bond_plot_module()
        bpm.restrict_axis_property_relative(xy_init,'($\sigma$)')
        #bpm.ax.set_xlim(3,16)#plt.xlim(-dis+center[0],dis+center[0])
        #bpm.ax.set_ylim(-12,1)#plt.ylim(-dis+center[1],dis+center[1])

        #draw bonds
        if not bond_cut_off is None:
            a_frame = static_points_analysis_2d(points=xy_init)
            a_frame.get_first_minima_bond_length_distribution(lattice_constant=1,hist_cutoff=bond_cut_off)#,png_filename=png_filename1
            check=[0.4, a_frame.bond_first_minima_left]
            bpm.draw_points_with_conditional_bond(xy_init,a_frame.bond_length,check,particle_size=particle_size)
        else:
            bpm.draw_points_with_conditional_bond(xy_init)
        #bpm.ax.set_xlim(3,16)#plt.xlim(-dis+center[0],dis+center[0])
        #bpm.ax.set_ylim(-12,1)#plt.ylim(-dis+center[1],dis+center[1])

        #draw circles & arrows of the next frame
        if ids is None:
            if nb_change is None:
                ids = None
                bpm.ax.scatter(xy_final[:,0],xy_final[:,1],facecolors='none',edgecolors=circle_color,marker='o',s=circle_size,linewidths=lw)
            else:
                list_sum_id_nb_stable = nb_change
                snap = list_sum_id_nb_stable[list_sum_id_nb_stable['frame']==frame_final]
                snap_part = snap[snap['if_nb_change'] == True]
                ids = snap_part["particle_id"].values
                
                if arrow == 'direct':#direct_arrow
                    bpm.ax.quiver(xy_init[ids,0],xy_init[ids,1],xy_final[ids,0]-xy_init[ids,0],xy_final[ids,1]-xy_init[ids,1],color=arrow_color,angles='xy', scale_units='xy', scale=1)
                elif arrow == 'annotate':#annotate_arrow
                    dxy,uv,ids = self.__annotate_arrow(xy_init,xy_final,ids)
                    # move arrow a little away from in situ points.
                    bpm.ax.quiver(xy_init[ids,0]+dxy[:,0],xy_init[ids,1]+dxy[:,1],uv[:,0],uv[:,1],color=arrow_color,angles='xy', scale_units='xy', scale=1)

                bpm.ax.scatter(xy_final[ids,0],xy_final[ids,1],facecolors='none',edgecolors=circle_color,marker='o',s=circle_size,linewidths=lw)#circle, facecolors='none',fillstyle='none'[x]
                """
                ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
                ...             arrowprops=dict(arrowstyle="->"))
                https://matplotlib.org/stable/tutorials/text/annotations.html#sphx-glr-tutorials-text-annotations-py
                """
        else:# force ax to draw scatter using given ids
            bpm.ax.scatter(xy_final[ids,0],xy_final[ids,1],facecolors='none',edgecolors=circle_color,marker='o',s=circle_size,linewidths=lw)

        #draw traps
        if not trap_filename is None:
            bpm.plot_traps(trap_filename,trap_lcr,'map')
            

        png_filename2 = prefix +'string_like_motion'+str_index+'_'+str(int(frame_init))+'.png'#'.pdf'
        bpm.save_figure(png_filename2)

        return ids

    def plot_string_like_motion(self,frame_init,frame_final,prefix='',
                                    data_name='default_exp',ids=None,
                                    bond_cut_off=None,
                                    trap_filename=None,trap_lcr=None):
        R"""
        return 
            ids: particles whose displacemnt is large
        """
        str_index=data_name
        txyz = np.load(prefix+'txyz.npy')

        xy_init = self.txyz_stable[frame_init,:,:2]
        xy_final = self.txyz_stable[frame_final,:,:2]
        particle_size=30
        circle_size = particle_size
        lw = 2#circle linewidths
        circle_color = 'limegreen'#'orange'
        arrow_color = 'limegreen'

        
        bpm = bond_plot_module()
        bpm.restrict_axis_property_relative(xy_init,'($\sigma$)')  
        #draw bonds
        a_frame = static_points_analysis_2d(points=xy_init)
        a_frame.get_first_minima_bond_length_distribution(lattice_constant=1,hist_cutoff=bond_cut_off)#,png_filename=png_filename1
        check=[0.4, a_frame.bond_first_minima_left]
        bpm.draw_points_with_given_bonds(xy_init,a_frame.bond_length,check,particle_size=particle_size)  
        #draw arrows
        df2 = displacemnt_field_2D(self.txyz_stable,bpm.ax,bpm.fig)  
        ids_in = df2.get_displacement_field_xy_id(frame_init,frame_final)
        #draw circles
        if ids is None:
            ids = ids_in
            bpm.ax.scatter(xy_final[ids,0],xy_final[ids,1],facecolors='none',
                edgecolors=circle_color,marker='o',s=circle_size,linewidths=lw,zorder =2)
        else:
            bpm.ax.scatter(xy_final[ids,0],xy_final[ids,1],facecolors='none',
                edgecolors=circle_color,marker='o',s=circle_size,linewidths=lw,zorder =2)#circle, facecolors='none',fillstyle='none'[x]
        #draw traps
        if not trap_filename is None:
            bpm.plot_traps(trap_filename,trap_lcr,'map')
            
        #save figure
        png_filename2 = prefix +'string_like_motion'+str_index+'_'+str(int(frame_init))+'_'+str(int(frame_final))+'.png'#'.pdf'
        bpm.save_figure(png_filename2)     
        return ids_in    

    def plot_string_like_motion_rank(self,frame_init,frame_final,prefix='',
                                    data_name='default_exp',nb_change=None,ids=None,
                                    bond_cut_off=None,
                                    trap_filename=None,trap_lcr=None):
        R"""
        return :
            ids: particles whose displacemnt is large
        warning:
            ids would lost partly for edge_cut ins static_points_analysis_2D
        """
        str_index=data_name
        txyz = np.load(prefix+'txyz.npy')

        xy_init = self.txyz_stable[frame_init,:,:2]
        xy_final = self.txyz_stable[frame_final,:,:2]
        particle_size=30
        circle_size = particle_size
        lw = 2#circle linewidths
        circle_color = 'limegreen'#'orange'
        arrow_color = 'limegreen'
        
        
        bpm = bond_plot_module()
        bpm.restrict_axis_property_relative(xy_init,'($\sigma$)')  
        #draw bonds
        a_frame = static_points_analysis_2d(points=xy_init)
        a_frame.get_first_minima_bond_length_distribution(lattice_constant=1,hist_cutoff=bond_cut_off)#,png_filename=png_filename1
        check=[0.4, a_frame.bond_first_minima_left]
        bpm.draw_points_with_given_bonds(xy_init,a_frame.bond_length,check,particle_size=particle_size)  
        #draw arrows
        df2 = displacemnt_field_2D(self.txyz_stable,bpm.ax,bpm.fig)  
        df2.get_displacements(frame_final,frame_init)
        uv,ids_active = df2.select_long_displacement_arrow()
        #df2.get_displacement_field_xy_id(frame_init)

        rank = df2.compare_displacement_method(uv,ids_active,frame_init,frame_final)
        #not so good, nan
        #rank = self.neighbor_change_id_method(nb_change,frame_init,frame_final,ids_active)
        
        df2.get_displacement_field_xy_rank(rank,frame_init,frame_final)

        #draw circles
        if ids is None:
            ids = ids_active
            bpm.ax.scatter(xy_final[ids,0],xy_final[ids,1],c=rank,
                marker='o',s=circle_size,linewidths=lw,zorder =2)#edgecolors=,
        else:
            bpm.ax.scatter(xy_final[ids,0],xy_final[ids,1],facecolors='none',
                edgecolors=circle_color,marker='o',s=circle_size,linewidths=lw,zorder =2)#circle, facecolors='none',fillstyle='none'[x]
        #draw traps
        if not trap_filename is None:
            bpm.plot_traps(trap_filename,trap_lcr,'map')
            
        #save figure
        png_filename2 = prefix +'string_like_motion_rank'+str_index+'_'+str(int(frame_init))+'_'+str(int(frame_final))+'.png'#'.pdf'
        bpm.save_figure(png_filename2)     
        return ids_active      
    
    def neighbor_change_id_method(self,nb_change,frame_init,frame_final,ids_active):
        list_sum_id_nb_stable = nb_change
        list_frames = (list_sum_id_nb_stable['frame']>=frame_init)&(list_sum_id_nb_stable['frame']<=frame_final)
        snap = list_sum_id_nb_stable[list_frames]
        snap_true = snap[snap['if_nb_change'] == True]
        #print(snap_true.head(10))
        
        ids_active = np.array(ids_active)
        num_ids_active = np.sum(ids_active.astype(int))
        rank = np.zeros((num_ids_active,))
        list_ids_active = np.zeros((num_ids_active,))
        #get list_ids_active
        j=0
        for i in range(len(ids_active)):
            if ids_active[i]:
                list_ids_active[j]=i
                j=j+1
                if j==num_ids_active:
                    break
        #print(list_ids_active)
        j=0
        for i in list_ids_active.astype(int):
            id_trajectory = snap_true[snap_true["particle_id"] == i]
            rank[j] = id_trajectory["frame"].max()
            j=j+1
        print(rank) #some ids are nan for edge_cut
        #or xy is in need to check displacement.
        return rank

    def __annotate_arrow(self,xy_init,xy_final,ids):
        R"""
        input:
            xy_init: from txyz_stable
            xy_final: from txyz_stable
            ids: defined by neighbor change events.
        output:
            dxy: vectors from points to places to annotate arrows, selected by ids only.
            uv: vectors of arrow, selected by ids only.
            ids: selected by displacements(>0.2)
        """
        #normalize arrow length part1
        uv = xy_final[ids]-xy_init[ids]
        dr2 = uv*uv
        dr = np.sqrt(dr2[:,0]+dr2[:,1])
        
        #select particles with long displacement
        dr_long_list = dr[:]>0.2
        ids=ids[dr_long_list]
        uv = xy_final[ids]-xy_init[ids]
        dr2 = uv*uv
        dr = np.sqrt(dr2[:,0]+dr2[:,1])

        ##normalize arrow length part2
        uv[:,0] = uv[:,0]/dr
        uv[:,1] = uv[:,1]/dr
        rotate = np.zeros((2,2))
        rotate[0,0] = np.cos(np.pi/2)
        rotate[1,1] = np.cos(np.pi/2)
        rotate[1,0] = np.sin(-np.pi/2)
        rotate[0,1] = np.sin(np.pi/2)
        # move arrow a little away from in situ points.
        scale_away = 0.5
        #rotate operator is on the right side, 
        # so rotational direction is inverted.
        #to let rotation match intuition, 
        # I inverted rotation operator too.
        dxy = np.matmul(uv,rotate)*scale_away
        #tune the length of arrow
        scale_arrow =1
        uv = uv*scale_arrow
        
        return dxy,uv,ids

class trajectory_module:
    def __init__(self):
        pass

    def trajectory_coarse_grain_general(self,txyz_stable_id=16,n_frame_to_coarse=2,tchip=None,
                                                prefix = '/home/remote/Downloads/4302_9/',
                                                mode = 'trajectory',ax=None,fig=None):
        R"""
        input:
            txyz_stable_id: int for a single particle trajectory; None for all the particle trajectories
            tchip:[fstart,fend]
            mode: 'trajectory' just plot trajectory;
                  'trajectory_displacement' plot trajectory with arrow
        introduction:
            method1: compare the i frame an the i-1 frame, 
            select parts of trajectories where chains of jump event occur.
            jump events are marked by the displacements whose length is 
            larger than the 1st minima in the distribution of displacements
            of all particles. However, the distribution of displacements
            of all particles are likely to has no minima, 
            and hence a forced rcut(e.g. 1 sigma) is in need of replacing 1st minima. 
            
            method1.2[on developing]: the difference from method1 is that 
            the 1st minima in the distribution of displacements are 
            defined by the displacements of one given particle id, 
            dr(df) = r(frame+df)-r(frame), not all the particle ids.
            
            method2[on developing]: accumulate n frames of displacements and draw the trajectory
        results:
            n_frame_to_coarse=1,plot: lines + clusters
            n_frame_to_coarse=2,plot: lines + clusters
        example:
            import points_analysis_2D as pa
            import matplotlib.pyplot as plt 
            trm = pa.trajectory_module()
            prefix = '/home/tplab/xiaotian_file/lxt_code_py/4302_9/'
            fig,axs = plt.subplots(1,3,sharey=True)
            ax0 = trm.trajectory_coarse_grain_general(None,2,[0,8],prefix,'trajectory_displacement',axs[0],fig)
            ax1 = trm.trajectory_coarse_grain_general(None,2,[8,104],prefix,'trajectory_displacement',axs[1],fig)
            ax2 = trm.trajectory_coarse_grain_general(None,2,[104,1427],prefix,'trajectory_displacement',axs[2],fig)
        
        
        colors: 
            link:https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
            color1
                pinned,interstitial;'r''orange';scale=1, width=0.005
            color2
                pinned,interstitial;'orange''royalblue';scale=1, width=0.005
                'orange''limegreen'
        
        """
        cp = 'r'   #color_pin
        ci = 'darkorange'#color_interstitial
        import points_analysis_2D as pa
        #prefix = '/home/remote/Downloads/4302_9/'
        filename_txyz_stable = prefix+'txyz_stable.npy'
        txyz_stable = np.load(filename_txyz_stable)
        if mode == "trajectory_displacement":
            file_t_pin_bool = prefix+'pin_check/t_pin_bool.npy'#'/home/tplab/xiaotian_file/lxt_code_py/4302_9/
            t_pin_bool = np.load(file_t_pin_bool).astype(bool)
        if not tchip is None:
            fstart,fend = tchip[0],tchip[1]
            txyz_stable = txyz_stable[fstart:fend,:,:2]
            strt = "_"+str(fstart)+"_"+str(fend)
            png_filename1 = prefix+"trajectory/displacement_hist_all_cut1"+strt+".png"
            png_filename2 = prefix+"trajectory/displacement_t_all_cut1"+strt+".png"
        sz = np.shape(txyz_stable)
        df = pa.dynamical_facilitation_module()
        if txyz_stable_id is None:#proceed trajectories for all particles
            dr = df.scan_displacement_t(txyz_stable[:,:,:2])
            szdr = np.shape(dr)
            dr_1d = np.reshape(dr,szdr[0]*szdr[1]) 
            self.plot_disp_hist(dr_1d,png_filename=png_filename1)
            self.displacement_length = dr_1d
            self.get_first_minima_count_of_distribution()
            list_jump_bool = self.get_jump_frame(dr,1.0)
            list_jump_bool[:].astype(int)
            count_jump = np.sum(list_jump_bool.astype(int),axis=0)#(Nparticles*1)[time_of_jumps]
            list_if_static = count_jump[:]<2
            if mode == "trajectory_displacement":
                linewidth = 2.5 #linewidth:quiverwidth = 1.5:0.005, then widths are equal
                width = 0.005*2/3*linewidth #set the same line width for plot & quiver
                size = 20#set the size of circles for scatter
                record_last_jump_xy = np.zeros((sz[1],4))#2
                list_if_jump = np.ones((sz[1],)).astype(bool)#to check if the particle(id) truly jumps
            if ax is None:
                fig,ax = plt.subplots()
            for id in range(sz[1]):
                if mode == "trajectory":
                    self.plot_trajectory(txyz_stable[list_jump_bool[:,id],id,:2],ax=ax)
                if mode == "trajectory_displacement":
                    if not ('fend' in locals()):
                        fend = -1
                    #check
                    #count_jump = np.sum(list_jump_bool[:,id].astype(int))
                    if count_jump[id]<2:#to check if the particle(id) truly jumps
                        #print(id,'_',count_jump)
                        #list_if_jump[id]=False #remove static or pinned particles
                        record_last_jump_xy[id] = [txyz_stable[-1,id,0],txyz_stable[-1,id,1],txyz_stable[-1,id,0],txyz_stable[-1,id,1]]#to show static particles
                    else:
                        last_jump_xy = self.plot_bicolor_trajectory(txyz_stable[list_jump_bool[:,id],id,:2],t_pin_bool[fend,id],ax=ax,width=linewidth,color_pin = cp,color_inte = ci)
                        record_last_jump_xy[id] = last_jump_xy 
                #simply accumulate hist will let 1st minima covered.
            #print(record_last_jump_xy[2,2:],record_last_jump_xy[2,:2])
            uv = record_last_jump_xy[:,2:]-record_last_jump_xy[:,:2]#txyz_stable[-1,:]-record_last_jump_xy
            lim = [[-18,18],[-18,18]]
            self.plot_bicolor_final_displacements(uv,record_last_jump_xy[:,0:2],t_pin_bool[fend],list_if_jump,ax,limit=lim,width=width,color_pin=cp,color_inte=ci)#txyz_stable[-1,:]
            self.plot_bicolor_final_displacements_static(record_last_jump_xy[:,0:2],t_pin_bool[fend],list_if_static,ax,limit=lim,size=size,color_pin=cp,color_inte=ci)
            #plt.show()
            
        else:#proceed trajectory for a single particle of the given id
            id = txyz_stable_id
            dr = df.scan_displacement_t(txyz_stable[:,id,:2])
            self.plot_trajectory(txyz_stable[:,id,:2])
            self.plot_disp_hist(dr)
            self.displacement_length = dr
            self.get_first_minima_count_of_distribution()
            list_jump_bool = self.get_jump_frame(dr)
            self.plot_trajectory(txyz_stable[list_jump_bool,id,:2])
            plt.show()

        if not tchip is None:
            fig.savefig(fname=png_filename2)
        
        return ax #to draw subplots

    def trajectory_coarse_grain(self,txyz_stable_id):
        R"""
        input:
            txyz_stable_id: [frame,x,y] for a single particle
        output:
            method1: compare the i-1 frame an the i+1 frame, merge trajectory in the same region.
            for the same particle id, dr(df) = r(frame+df)-r(frame),
            method2: average 3 frames of positions and draw the trajectory
        """
        import points_analysis_2D as pa
        prefix = '/home/remote/Downloads/4302_9/'
        filename_txyz_stable = prefix+'txyz_stable.npy'
        txyz_stable = np.load(filename_txyz_stable)
        sz = np.shape(txyz_stable)
        df = pa.dynamical_facilitation_module()
        for id in range(sz[1]):
            dr = df.scan_displacement_t(txyz_stable[:,id,:2],)
        pass
    
    def get_first_minima_count_of_distribution(self,method="global_compare_method"):#,hist_cutoff=2,png_filename=None,x_unit='(1)'
        R"""  
        Introduction:
            It's likely that displacements whose length are larger than 2A are not displacemented. 
            Typical displacement-lengths of honeycomb are A and 1.73A (displacement_first_neighbour & displacement_second_neighbour namely),
            hence the range=[0,2] times of lattice constant should be set.
        Parameters:
            method:"local_compare_nethod" to get the 1st minima through comparing local minima,
                suitable for continuous peak-valley hist;
                "global_compare_method" to get the 1st minima through comparing all the peaks, 
                selecting the 1st main peak(ignoring tiny peak before which), 
                finding the 1st main minima just after the 1st main peak, 
                suitable systems with powerful perturbation.
            
            hist_cutoff: plot hist of displacement_length till n times lattice_constant where n is hist_cutoff.
            
            png_filename="prefix/displacement_hist_index1512"

            displacement_first_minima_left:the upper limit of 1st neighbour displacements comming from the first minima 
                                    of displacement length distribution, with sigma as lower limit.
            
        Warning:
            [x]What if there is no minima found? 
            count_1st_max / count_1st_minima > 10 used to check the effective of minima? 

        Examples:
        """
        #locate the 1st minima of displacement-length distribution
        #plt.figure()
        count_bins=self.count_bins#plt.hist(self.displacement_length,bins=20,range=[0,hist_cutoff])
        
        self._count=count_bins[0]
        self._bins=count_bins[1]
        self.displacement_sorted=np.sort(self.displacement_length)
        #find the minimum bin, then set the left side as displacement_first_neighbour
        i_max=self._bins.size
        i=0
        if method=="local_compare_method":
            while i < i_max-3:#i start from 0, which have to -1;compares i,i+1 and i+2, which have to -2, hence -3
                #np.where( self.displacement_sorted[:]>res.bins[i] & self.displacement_sorted[:]< res.bins[i+1] ) 
                #print(self._count[i])
                if self._count[i] > self._count[i+1]:
                    if self._count[i+1] <= self._count[i+2]:
                        #if self._bins[i+1] > 1:
                        # displacement_length should be larger than sigma of particle
                        i+=1
                        break
                i+=1

        elif method=="global_compare_method":
            R"""
            scan the histogram of displacement length distribution, get the highest two columns.
            see the column whose displacement length is shorter as 'first peak'.
            search for local minima right after the 'first peak'.
            """
            self._count = np.array(self._count)
            self.count_sorted=np.sort(self._count)
            """
            print(self.count_sorted)
            print(self.count_sorted[-1])#max1
            print(self.count_sorted[-2])#max2
            """
            i_bins_for_count_peak_1 = np.where(self._count[:]==self.count_sorted[-1])
            i_bins_for_count_peak_2 = np.where(self._count[:]==self.count_sorted[-2])
            i_bins_for_count_peak_1 = i_bins_for_count_peak_1[0]
            i_bins_for_count_peak_2 = i_bins_for_count_peak_2[0]
            #if there are more than one bins share the same count, select the smallest bin number.
            if np.shape(i_bins_for_count_peak_1)[0]>1:
                i_bins_for_count_peak_1 = min(i_bins_for_count_peak_1)
            if np.shape(i_bins_for_count_peak_2)[0]>1:
                i_bins_for_count_peak_2 = min(i_bins_for_count_peak_2)
            i_bins_for_count_1st_peak = min(i_bins_for_count_peak_1,i_bins_for_count_peak_2)
            i_bins_for_count_1st_peak = int(i_bins_for_count_1st_peak)#force i be a simple data_type(int)
            """
            print(i_bins_for_count_peak_1) 
            print(i_bins_for_count_peak_2)
            print(i_bins_for_count_1st_peak) 
            """
            i = i_bins_for_count_1st_peak
            while i < i_max-3:#i start from 0, which have to -1;compares i,i+1 and i+2, which have to -2, hence -3
                if self._count[i] > self._count[i+1]:
                    if self._count[i+1] <= self._count[i+2]:
                        if self._bins[i+1] <=  self._bins[i+3]:#even if jump, i+1 is still the minimum.edit for bug in index4323_8_2000
                            #if self._bins[i+1] > 1:
                            # displacement_length should be larger than sigma of particle
                            i+=1
                            break
                i+=1
            
        else:
            print("Err: input 'local_compare_method' or 'global_compare_method', please!")
            print("x")
            
        self.displacement_first_minima_left=self._bins[i]
        self.displacement_first_neighbour=self.displacement_sorted[np.where(self.displacement_sorted[:]<self.displacement_first_minima_left)]
    
    def get_jump_frame(self,dr,rcut=None):
        R"""
        list_jump_bool: (Nframe,Nparticle). TRUE for jump, FALSE for oscillation.
                        caution, frame 0 is always seen as jump!
        """
        if rcut is None:
            list_jump_bool_1 = dr > self.displacement_first_minima_left
        else:
            list_jump_bool_1 = dr > rcut
        sp = np.shape(list_jump_bool_1)
        if len(sp)==1:
            #np.ones sets the initial position as true.
            list_jump_bool = np.ones((sp[0]+1,),dtype=np.bool8)#.astype(boolean)
            list_jump_bool[1:] = list_jump_bool_1
        else:
            #np.ones sets the initial position as true.
            list_jump_bool = np.ones((sp[0]+1,sp[1]),dtype=np.bool8)#.astype(boolean)
            list_jump_bool[1:,:] = list_jump_bool_1
        return  list_jump_bool

    def plot_trajectory(self,txy,ax=None):
        if ax is None:
            fig,ax = plt.subplots()
        #ax.scatter(pos[:,0],pos[:,1],c='k')
        ax.plot(txy[:,0],txy[:,1])#,c='k'
        ax.set_aspect('equal','box')
        plt.show()
        #png_filename = 'points.png'
        #plt.savefig(prefix_write+png_filename)
    
    def plot_disp_hist(self,disp,hist_cutoff=2,png_filename=None):
        fig,ax = plt.subplots()
        self.count_bins = ax.hist(disp,bins=20,log=True,range=[0,hist_cutoff])
        ax.set_ylabel('count(1)')
        ax.set_xlabel('$dr(\sigma)$')
        if not png_filename is None:
            fig.savefig(fname=png_filename)
        #ax.set_aspect('equal','box')
    
    def plot_bicolor_trajectory(self,txy,t_pin_bool,ax=None,width=1,color_pin = 'r',color_inte = 'orange'):#,list_jump_bool
        R"""
        input:
            list_jump_bool: part of (Nframe,Nparticle)[bool]. TRUE for jump, FALSE for oscillation.
                get from self.get_jump_frame()
            t_pin_bool: part of (Nframe,Nparticle)[bool]. TRUE for pinned, FALSE for free. 
                from dynamical_facilitation_module.get_pin_bool()
        return:
            last_jump_xy: [x,y] the final position of particle(id) jump to.
        """
        #get last two frames of 
        """
        list_jump_pin = np.logical_and(list_jump_bool,t_pin_bool)
        list_jump_free = np.logical_and(list_jump_bool,np.logical_not(t_pin_bool))
        txy[-1,t_pin_bool[-1],:2]#init
        txy[-1,t_pin_bool[-1],:2]#final
        """
        
        if ax is None:
            fig,ax = plt.subplots()
        #ax.scatter(pos[:,0],pos[:,1],c='k')
        if t_pin_bool:
            color=color_pin
        else:
            color=color_inte
        ax.plot(txy[:-1,0],txy[:-1,1],c=color,linewidth=width)#
        ax.set_aspect('equal','box')
        
        last_jump_xy = txy[-2:].reshape((4,))#may have bug when shape(txy)=null [x]
        return last_jump_xy

    def plot_bicolor_final_displacements(self,uv,xy,list_pin_true,list_if_jump,ax=None,x_unit='($\sigma$)',limit=None,width=0.01,color_pin='r',color_inte='orange'):#,png_filename=None
        list_pin_false =  np.logical_not(list_pin_true)
        list_pin_true_1 = np.logical_and(list_pin_true,list_if_jump)
        list_pin_false_1 = np.logical_and(list_pin_false,list_if_jump)
        if ax is None:
            fig,ax = plt.subplots()

        if not limit is None:
            ax.set_xlim(limit[0])
            ax.set_ylim(limit[1])
            ax.quiver(xy[list_pin_true_1,0],xy[list_pin_true_1,1],uv[list_pin_true_1,0],uv[list_pin_true_1,1],color=color_pin,angles='xy', scale_units='xy', scale=1,width=width)#,width=0.01,color='r'
            ax.quiver(xy[list_pin_false_1,0],xy[list_pin_false_1,1],uv[list_pin_false_1,0],uv[list_pin_false_1,1],color=color_inte,angles='xy', scale_units='xy', scale=1,width=width)
            #ax.scatter(final_positions[:,0],final_positions[:,1])#final_state
        else:
            ax.quiver(xy[list_pin_true_1,0],xy[list_pin_true_1,1],uv[list_pin_true_1,0],uv[list_pin_true_1,1],color=color_pin,angles='xy', scale_units='xy', scale=1,width=width)#,color='r'
            ax.quiver(xy[list_pin_false_1,0],xy[list_pin_false_1,1],uv[list_pin_false_1,0],uv[list_pin_false_1,1],color=color_inte,angles='xy', scale_units='xy', scale=1,width=width)
        #ax.scatter(xye[:,0],xye[:,1],c='k')#init_state
        ax.set_title('displacement field ')#+'index:'+str(simu_index)
        ax.set_xlabel('x'+x_unit)
        ax.set_ylabel('y'+x_unit)
        ax.set_aspect('equal','box')
    
    def plot_bicolor_final_displacements_static(self,xy,list_pin_true,list_if_static,ax=None,limit=None,size=1,color_pin='r',color_inte='orange'):#,png_filename=None
        list_pin_false =  np.logical_not(list_pin_true)
        list_pin_true_1 = np.logical_and(list_pin_true,list_if_static)
        list_pin_false_1 = np.logical_and(list_pin_false,list_if_static)
        if ax is None:
            fig,ax = plt.subplots()

        if not limit is None:
            ax.set_xlim(limit[0])
            ax.set_ylim(limit[1])
            ax.scatter(xy[list_pin_true_1,0],xy[list_pin_true_1,1],color=color_pin,s=size)
            ax.scatter(xy[list_pin_false_1,0],xy[list_pin_false_1,1],color=color_inte,s=size)
            #ax.scatter(final_positions[:,0],final_positions[:,1])#final_state
        else:
            ax.scatter(xy[list_pin_true_1,0],xy[list_pin_true_1,1],color=color_pin,s=size)
            ax.scatter(xy[list_pin_false_1,0],xy[list_pin_false_1,1],color=color_inte,s=size)

        
        
class mean_square_displacement:
    R"""
    Introduction:
        defined by wikipedia(https://en.wikipedia.org/wiki/Mean_squared_displacement)
        this function is designed to calculate mean squared displacements.
        particle index: i
        displacement: dr(i,t-0) = r(i,t) - r(i,0)
        squared displacement: [ dr(i,t-0) ]^2
        mean squared displacement: MSD(t) = < [ dr(i,t-0) ]^2 >_i
        averaged time MSD: MSD(dt) where dt = t2 -t1 to replace t -0, which is used to improve statistical performance.
        caution: averaged time MSD would hide dynamic events like collapse.

    example_experiment:
        import numpy
        import particle_tracking as ptt
        ana = ptt.particle_track()
        dir_path='/home/remote/xiaotian_file/data/20221129/DefaultVideo_5'
        traj = ana.select_stable_trajectory()
        traj_um = traj/32*3
        traj_sigma = traj_um/2
        pa = msd(traj_sigma,mode='exp')
        pa.compute_atmsd_t_chips(interval_max=0.9)
        path = '/home/remote/xiaotian_file/data/20221129/video5/'
        time_log = np.loadtxt(path+'DefaultVideo_5.txt')

        pa.plot_msd(time_log,path+'msd_chips_loglog_'+'20221129_video5_90%'+'.png')
        pa.plot_trajectory(path+'trajectory_stable_'+'20221129_video5'+'.png')

    EXAMPLE:
        print(msds.result_msd)
        print(np.shape(msds.result_msd))
    example:
        import points_analysis_2D as pa
        gsd_data = pa.proceed_gsd_file(simu_index=5208,seed=9)
        gsd_data.get_trajectory_data()

        msd_class = pa.msd(gsd_data.txyz,gsd_data.box)

        import freud
        msds = freud.msd.MSD(gsd_data.box)#the class is fault,,'direct'
        msds.compute(positions=msd_class.txyz_stable)
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.plot(msds.msd)
        plt.title("Mean Squared Displacement")
        plt.xlabel("$t$")
        plt.ylabel("MSD$(t)$")
        png_filename = 'msd_'+'index5208_9'+'.png'
        plt.savefig(png_filename)#png_filename
        plt.close()
        #print(tr.shape)
    exp-get nn msd:
        ana = ptt.particle_track()
        traj = ana.select_stable_trajectory()
        traj_um = traj/32*3
        traj_sigma = traj_um/2
        pa = dynamic_points_analysis_2d(traj_sigma,mode='exp')
        ts_id_dxy = pa.compute_nearest_neighbor_displacements(unit='sigma')
        import pandas as pd
        ts_id_dxy = pd.read_csv('ts_id_dxy.csv')
        ts_id_dxy['z'] = 0
        pa = dynamic_points_analysis_2d(ts_id_dxy,mode='exp')
        txyz_ids_stable = pa.compute_nearest_neighbor_displacements_stable(pa.txyz_stable)
        pa = dynamic_points_analysis_2d(txyz_ids_stable,mode='exp')
        pa.compute_atmsd_t_chips(0.9)
        path_to_folder = '/home/remote/xiaotian_file/data/20221129/video5/'
        time_log = path_to_folder+'DefaultVideo_5.txt'
        time_log = np.loadtxt(time_log)
        pa.plot_msd_t_chips(time_log)
    """
    def __init__(self,txyz_stable,mode):
        self.txyz_stable = txyz_stable
        self.frames,self.particles,self.dimensions=self.txyz_stable.shape
        self.mode = mode

    def compute_msd_normal(self):
        m_max = int(0.5*self.frames)#to ensure the robustness of statistics
        #k_max = 0
        start = 1
        self.record_msd=np.zeros((m_max-start+1,2))#[interval m, msd_m]
        
        list_m=np.arange(start,m_max+1)
        self.record_msd[:,0]=list_m
        for m in list_m:
            self.dm_xyz = self.txyz_stable[m:,:,:] - self.txyz_stable[0,:,:]
            sq_dm_xyz2 = np.square(self.dm_xyz)
            sum_sq_dm_r2 = np.sum(sq_dm_xyz2)
            self.record_msd[m-start,1]=sum_sq_dm_r2/self.dm_xyz.shape[0]/self.particles

    def compute_atmsd_scan_t(self):
        R"""
        method:
            scan t axis, 1 frame for each interval. here may exist some overlaps between two intervals who share the same length of time interval m.
        record_msd:[interval m, msd_m]
        k: the frame to start scan 
        m: time interval
        m_max: max time interval
        """
        m_max = 9000#int(0.5*self.frames)#to ensure the robustness of statistics
        #k_max = 0
        start = 1000
        self.record_msd=np.zeros((m_max-start+1,2))#[interval m, msd_m]
        
        list_m=np.arange(start,m_max+1)
        self.record_msd[:,0]=list_m
        for m in list_m:
            self.dm_xyz = self.txyz_stable[m:,:,:] - self.txyz_stable[:-m,:,:]
            sq_dm_xyz2 = np.square(self.dm_xyz)
            sum_sq_dm_r2 = np.sum(sq_dm_xyz2)
            self.record_msd[m-start,1]=sum_sq_dm_r2/self.dm_xyz.shape[0]/self.particles
    
    def compute_atmsd_scan_t_log_record(self):
        R"""
        method:
            scan t axis, 1 frame for each interval. here may exist some overlaps between two intervals who share the same length of time interval m.
        record_msd:[interval m, msd_m]
        k: the frame to start scan 
        m: time interval
        m_max: max time interval
        """
        frames = np.log10(self.frames)
        print(self.frames)
        bins = 20
        alpha = np.linspace(0,frames,bins)
        list_m = np.int32(np.power(10,alpha)) 
        print(list_m)
        self.record_msd=np.zeros((np.shape(list_m)[0],2))#[interval m, msd_m]
        
        snap_to_step = 1000
        self.record_msd[:,0]=list_m*snap_to_step#+500000-100
        i=0
        for m in list_m:
            self.dm_xyz = self.txyz_stable[m:,:,:] - self.txyz_stable[:-m,:,:]
            sq_dm_xyz2 = np.square(self.dm_xyz)
            sum_sq_dm_r2 = np.sum(sq_dm_xyz2)
            self.record_msd[i,1]=sum_sq_dm_r2/self.dm_xyz.shape[0]/self.particles
            i=i+1
        
        return self.record_msd

    def compute_atmsd_t_chips(self,interval_max=0.1,msd_per_particle=False):
        R"""
        parameters:
            interval_max: (0,1) to which the longest interval msd is computed.
            k: the frame to start scan 
            m: time interval
            m_max: max time interval
            msd_per_particle: control the program to calculate msd for each particle
        
        return:
            self.record_msd: [interval m, atmsd(m)]
            or record_msd_id

        method:
            cut t axis into several chips for each interval. there are not any two 
            intervals who share the same length of time interval m would overlap.
            Hence we get the averaged time mean square displacement(atmsd).

        """
        if not hasattr(self,'frames'):
            self.frames,self.particles,self.dimensions=self.txyz_stable.shape
        m_max = int(interval_max*self.frames)#to ensure the robustness of statistics
        start = 1
        self.record_msd=np.zeros((m_max-start+1,2))#[interval m, msd_m]
        #you can select points [log 1, 1/100*log m_max,log m_max]
        list_m=np.arange(start,m_max+1)
        self.record_msd[:,0]=list_m
        
        for m in list_m:
            chips = int((self.txyz_stable.shape[0]-1)/m)# how many chips the time axis is divided into
            list_frames = m*np.arange(0,chips+1)
            m_xyz = self.txyz_stable[list_frames,:,:]
            self.dm_xyz = m_xyz[1:,:,:] - m_xyz[:-1,:,:]
            sq_dm_xyz2 = np.square(self.dm_xyz)
            sum_sq_dm_r2 = np.sum(sq_dm_xyz2)
            self.record_msd[m-start,1]=sum_sq_dm_r2/self.dm_xyz.shape[0]/self.particles

            if msd_per_particle:
                id_sum_sq_dm_r2_chips = np.sum(sq_dm_xyz2,axis=2)
                print(id_sum_sq_dm_r2_chips.shape)
                id_sum_sq_dm_r2 = np.sum(id_sum_sq_dm_r2_chips,axis=0)
                print(id_sum_sq_dm_r2.shape)
                if 'record_msd_id' in locals():
                    record_msd_id[m-start,:]=id_sum_sq_dm_r2/self.dm_xyz.shape[0]
                else:
                    record_msd_id=np.zeros((m_max-start+1,self.particles))
                    record_msd_id[m-start,:]=id_sum_sq_dm_r2/self.dm_xyz.shape[0]

        if msd_per_particle:
            return record_msd_id

    def compute_nn_msd_X(self,unit='sigma'):
        R"""
        parameters:
            input: trajectory (unit = sigma or um)

        introduction:
            DOI: 10.1103/PhysRevE.95.022602
            
            nearest-neighbor relative mean square displacement(nnmsd): <[r(i,t)-<r(j,t)>]^2>
            where j is the neighbors of particle i.

            The dynamic Lindemann parameter: gamma_l_tau = nnmsd/2/a^2
        """
        #get frame-wise 1st neighbor bond length
        #list_1st_bond_length_framewise
        list_frames = range(self.frames)
        list_1st_bond_length_framewise = np.zeros((self.frames,))
        for frame in list_frames:
            result = static_points_analysis_2d(self.txyz_stable[frame,:,:])
            if unit == 'sigma':
                lc = 1.0#lattice constant, particle diameter(2 um) as default
            elif unit == 'um':
                lc = 2.0
            result.get_first_minima_bond_length_distribution(lattice_constant=lc,hist_cutoff=5)#here lattice_constant is just used to normalize figure, hence set 2.0 is ok
            #print('recognized bond length: '+str(result.bond_length_median*lc)+'+-'+str(result.bond_length_std*lc)+' '+unit)
            list_1st_bond_length_framewise[frame] = result.bond_length_median#unit = sigma!
            #displacement neighbor
            #result.bond_first_neighbour
            id_dxy = result.get_nearest_neighbor_dispalcement()
            t_id_dxy = np.zeros((id_dxy.shape[0],id_dxy.shape[1]+1))  
            t_id_dxy[:,0] = frame
            t_id_dxy[:,1:] = id_dxy  
            if frame == 0:
                ts_id_dxy = t_id_dxy
            else:
                ts_id_dxy = np.concatenate((ts_id_dxy,t_id_dxy), axis=0) #[frame,id,dx,dy]
            #print(ts_id_dxy.shape)
        #select stable trajectories
        print(ts_id_dxy[:,1].max)
        list_ids = np.unique(ts_id_dxy[:,1])
        id_check = np.zeros((list_ids.shape[0],),dtype=bool)
        for id in list_ids:
            ts_dxy = ts_id_dxy[ts_id_dxy[:,1]==id]

            if ts_dxy.shape[0] == self.frames:
                id_check[id] = True
        ts_id_dxy_stable = ts_id_dxy[:,id_check,:,:]
        #particle_tracking.select_stable_trajectory()is great!

    def plot_msd_t_chips(self,time_log=None,png_filename='msd_chips_long_loglog.png',sigma=False,um_sec=True,lindemann=False):
        R"""
        introduction:
            input: 'txyz_stable.csv' 
            output: msd plot
        """
        import matplotlib.pyplot as plt 
        plt.figure()
        if time_log is None:
            plt.semilogx(self.record_msd[:,0],self.record_msd[:,1])#loglog semilogx
        else:
            time_msd=time_log[:self.record_msd.shape[0]]
            plt.loglog(time_msd,self.record_msd[:,1])

        if self.mode =='simu':
            plt.title("Mean Squared Displacement")
            plt.xlabel("$t$ (steps)" )
            plt.ylabel("MSD$(t)(\sigma^2)$ ")
        elif self.mode == 'exp':
            if lindemann:
                plt.title("Mean Squared Displacement")
                plt.xlabel("$t$ (sec)")
                plt.ylabel("MSD$(t)$ (sigma^2/A^2)")
            elif sigma:
                plt.title("Mean Squared Displacement")
                plt.xlabel("$t$ (sec)")
                plt.ylabel("MSD$(t)$ (sigma^2)")
            elif um_sec:
                plt.title("Mean Squared Displacement")
                plt.xlabel("$t$ (sec)")
                plt.ylabel("MSD$(t)$ (um^2)")
        #png_filename = 'msd_chips_long_loglog_'+'index5208_9'+'.png'
        plt.savefig(png_filename)#png_filename
        plt.close()
        """
        list_dm_frames = np.arange(self.dm_xyz.shape[0])
        for i in list_dm_frames:
            #np.dot()
            self.dm_xyz_tr=np.transpose(self.dm_xyz)
            np.matmul(self.dm_xyz),self.dm_xyz_tr)
        """
        
        
        """
        m=0#frame interval
        record_disp2_1 = np.zeros((self.sp[0]-1,3))
        while m<self.sp[0]-1:#m < N frame
            k=0#frame scanning
            record_disp2_m = np.zeros((1,3))
            
            while k<self.sp[0]-1-m:
                disp_m = positions[k+m]-positions[k]#199-0 what is the problem?
                disp2_m = np.dot(np.transpose(disp_m),disp_m) 

                if disp2_m[0,0]>0:
                    print(m)

                record_disp2_m = record_disp2_m + [disp2_m[0,0],disp2_m[1,1],disp2_m[2,2]]
                k=k+1

            record_disp2_m = record_disp2_m/(self.sp[0]-1-m)#mean of all the chips with k frames interval
            record_disp2_1[m] = record_disp2_m[0]
            
            m=m+1
        self.result_msd_xyz = record_disp2_1/self.sp[1]#divided by N_particles
        self.result_msd = np.zeros((self.sp[0]-1,1))
        self.result_msd = self.result_msd_xyz[:,0]+self.result_msd_xyz[:,1]+self.result_msd_xyz[:,2]        
        """
    
    def plot_lindemann_msd(self,m_msd,average_1st_bond_length,time_log=None,png_filename='lindemann_msds_loglog.png'):
        R"""
        input: 
            m_msd: n rows of [interval m, msd(m)] . 
            
            time_log: at least n rows of [interval m], 
                here m is real seconds, frames or others 
                you want to set as time interval.

        output: msd plot

        introduction:
            
        """
        import matplotlib.pyplot as plt 
        a2 = average_1st_bond_length*average_1st_bond_length
        plt.figure()
        if time_log is None:
            plt.loglog(m_msd[:,1],m_msd[:,1]/a2/2.0)
        else:
            time_msd=time_log[:m_msd.shape[0]]
            plt.loglog(time_msd,m_msd[:,1]/a2/2.0)

        plt.title("dynamic Lindemann Mean Squared Displacement")
        plt.xlabel("$t$ (sec)")
        plt.ylabel("gamma_L$(t)$ (1)")
        #png_filename = 'msd_chips_long_loglog_'+'index5208_9'+'.png'
        plt.savefig(png_filename)#png_filename
        plt.close()

    def plot_msd_particle_wise_X(self,m_msd,time_log=None,png_filename='msds_loglog.png'):
        s = 0
        self.plot_lindemann_msd()

class displacemnt_field_2D:
    def __init__(self,txyz_stable,ax=None,fig=None):
        self.txyz_stable = txyz_stable
        if ax is None:
            fig,ax = plt.subplots()
        self.fig = fig
        self.ax = ax      

    def get_displacements(self,frame_index_end=-1,frame_index_start=0):
        init_positions = self.txyz_stable[frame_index_start]
        final_positions = self.txyz_stable[frame_index_end]
        self.displacements =  final_positions - init_positions

    def get_displacement_field_xy(self,frame_index_start=0,frame_index_end=-1,plot=False,png_filename=None,x_unit='($\sigma$)',limit=None):
        R"""
        Introduction:
            The function draws a displacement vector field from init state to final state 
            with positions at edge removed  to clean abnormal displacement vector. 
        input:
            frame_index: -1
            plot:True or False
            png_filename: 'displacement_field_xy.png'
            x_unit:'(sigma)', '(um)' or '(1)'
            limit: [[-4,12],[-5,16]]
        Example:
            4302_2,0-29,[[9,21],[-7,7]]
        """
        self.get_displacements(frame_index_end,frame_index_start)

        xy = self.txyz_stable[frame_index_start]#init_positions
        xye = self.txyz_stable[frame_index_end]
        uv = self.displacements

        if plot:
            if not limit is None:
                self.ax.set_xlim(limit[0])
                self.ax.set_ylim(limit[1])
                self.ax.quiver(xy[:,0],xy[:,1],uv[:,0],uv[:,1],color='orange',angles='xy', scale_units='xy', scale=1,width=0.01)
            #self.ax.scatter(self.final_positions[:,0],self.final_positions[:,1])#final_state
            else:
                self.ax.quiver(xy[:,0],xy[:,1],uv[:,0],uv[:,1],color='orange',angles='xy', scale_units='xy', scale=1)#,width=0.01
            self.ax.scatter(xye[:,0],xye[:,1],c='k')#init_state
            self.ax.set_title('displacement field ')#+'index:'+str(self.simu_index)
            self.ax.set_xlabel('x'+x_unit)
            self.ax.set_ylabel('y'+x_unit)
            self.ax.set_aspect('equal','box')

            if not png_filename is None:
                plt.savefig(png_filename)
            plt.close()
        return uv

    def get_bicolor_disp(self,list_pin_bool,frame_index_start=0,frame_index_end=-1,plot=False,png_filename=None,x_unit='($\sigma$)',limit=None,traps=None):
        R"""
        Introduction:
            The function draws a displacement vector field from init state to final state 
            with positions at edge removed  to clean abnormal displacement vector. 
        input:
            frame_index: -1
            plot:True or False
            png_filename: 'displacement_field_xy.png'
            x_unit:'(sigma)', '(um)' or '(1)'
            limit: [[-4,12],[-5,16]]
        Example:
            4302_2,0-29,[[9,21],[-7,7]]
        """
        self.get_displacements(frame_index_end,frame_index_start)
        xy = self.txyz_stable[frame_index_start]#init_positions
        xye = self.txyz_stable[frame_index_end]
        uv = self.displacements
        list_pin_bool = list_pin_bool.astype(bool)
        list_pin_nbool = np.logical_not(list_pin_bool)
        if plot:
            if not limit is None:
                self.ax.set_xlim(limit[0])
                self.ax.set_ylim(limit[1])
                self.ax.quiver(xy[list_pin_bool,0],xy[list_pin_bool,1],uv[list_pin_bool,0],uv[list_pin_bool,1],color='r',angles='xy', scale_units='xy', scale=1,width=0.01)
                self.ax.quiver(xy[list_pin_nbool,0],xy[list_pin_nbool,1],uv[list_pin_nbool,0],uv[list_pin_nbool,1],color='orange',angles='xy', scale_units='xy', scale=1,width=0.01)
            #self.ax.scatter(self.final_positions[:,0],self.final_positions[:,1])#final_state
            else:
                #self.ax.quiver(xy[:,0],xy[:,1],uv[:,0],uv[:,1],c=list_pin_bool,angles='xy', scale_units='xy', scale=1)#,width=0.01,color='r'
                self.ax.quiver(xy[list_pin_bool,0],xy[list_pin_bool,1],uv[list_pin_bool,0],uv[list_pin_bool,1],color='r',angles='xy', scale_units='xy', scale=1)#,width=0.01,color='r'
                self.ax.quiver(xy[list_pin_nbool,0],xy[list_pin_nbool,1],uv[list_pin_nbool,0],uv[list_pin_nbool,1],color='orange',angles='xy', scale_units='xy', scale=1)
            #self.ax.scatter(xye[list_pin_bool,0],xye[list_pin_bool,1],c='k')
            #self.ax.scatter(xye[list_pin_nbool,0],xye[list_pin_nbool,1],c='darkviolet')
            self.ax.scatter(xye[:,0],xye[:,1],c='k')
            #self.ax.scatter(traps[:,0],traps[:,1],c='r',marker = 'x',zorder=3)
            self.ax.set_title('displacement field ')#+'index:'+str(self.simu_index)
            self.ax.set_xlabel('x'+x_unit)
            self.ax.set_ylabel('y'+x_unit)
            self.ax.set_aspect('equal','box')

            if not png_filename is None:
                plt.savefig(png_filename)
            plt.close()
        return uv

    def get_displacement_field_xy_id(self,frame_index_start=0,frame_index_end=-1,x_unit='($\sigma$)',color='limegreen'):
        R"""
        Introduction:
            The function draws a displacement vector field from init state to final state 
            with positions at edge removed  to clean abnormal displacement vector. 
        input:
            frame_index: -1
            plot:True or False
            png_filename: 'displacement_field_xy.png'
            x_unit:'($\sigma$)', '(um)' or '(1)'
        return:
            ids: the ids of particles whose displacements are large enough.
        Example:
            import points_analysis_2D as pa
            gsd = pa.proceed_gsd_file(simu_index=1382)
            gsd.get_displacement_field(plot=True)
        """
        self.get_displacements(frame_index_end,frame_index_start)
        uv,ids = self.select_long_displacement_arrow()

        xy = self.txyz_stable[frame_index_start]#init_positions
        xye = self.txyz_stable[frame_index_end]


        #self.ax.scatter(self.final_positions[:,0],self.final_positions[:,1])#final_state
        self.ax.quiver(xy[ids,0],xy[ids,1],uv[:,0],uv[:,1],color=color,angles='xy', scale_units='xy', scale=1)
        """
        self.ax.scatter(xy[:,0],xy[:,1],c='k')#init_state
        self.ax.set_title('displacement field ')#+'index:'+str(self.simu_index)
        self.ax.set_xlabel('x'+x_unit)
        self.ax.set_ylabel('y'+x_unit)
        self.ax.set_aspect('equal','box')

        """
        
        if False:
            self.ax.xlim(5,20)
            self.ax.ylim(-15,0)
        return ids

    def get_displacement_field_xy_rank(self,rank,frame_index_start=0,frame_index_end=-1,x_unit='($\sigma$)'):#,color='limegreen'
        R"""
        Introduction:
            The function draws a displacement vector field from init state to final state 
            with positions at edge removed  to clean abnormal displacement vector. 
        input:
            frame_index: -1
            plot:True or False
            png_filename: 'displacement_field_xy.png'
            x_unit:'($\sigma$)', '(um)' or '(1)'
        return:
            ids: the ids of particles whose displacements are large enough.
        Example:
            import points_analysis_2D as pa
            gsd = pa.proceed_gsd_file(simu_index=1382)
            gsd.get_displacement_field(plot=True)
        """
        self.get_displacements(frame_index_end,frame_index_start)
        uv,ids = self.select_long_displacement_arrow()

        xy = self.txyz_stable[frame_index_start]#init_positions


        #self.ax.scatter(self.final_positions[:,0],self.final_positions[:,1])#final_state
        #arrow
        #plt.figure()
        #plt.quiver(xy[ids,0],xy[ids,1],uv[:,0],uv[:,1],c=rank, scale=1)
        #plt.show()
        cc = np.hypot(uv[:,0],uv[:,1])
        mapp = self.ax.quiver(xy[ids,0],xy[ids,1],uv[:,0],uv[:,1],rank,angles='xy', scale_units='xy', scale=1)#,rank
        self.fig.colorbar(mapp,ax=self.ax)#ax=self.ax

    def get_displacement_field_distribution(self,frame_index,log_mode=False,png_filename=None):
        R"""
        Introduction:
            The function draws a displacement vector field from init state to final state 
            with positions at edge removed  to clean abnormal displacement vector. 
        Example:
            import points_analysis_2D as pa
            gsd = pa.proceed_gsd_file(simu_index=1382)
            gsd.get_displacement_field(plot=True)
        """
        self.get_displacements(frame_index)

        xy = self.txyz_stable[0]#init_positions
        uv = self.displacements
        uv2 = uv*uv#let each element  i be i^2
        square_displacement = uv2[:,0]+uv2[:,1]
        rsd = np.sqrt(square_displacement)

        if log_mode:
            rsd = np.log10(square_displacement)
            plt.figure()
            count_bins=plt.hist(rsd,bins=20,range=[-2,2])#
            plt.xlabel('log(displacement)/um')
        else:
            plt.figure()
            count_bins=plt.hist(rsd,bins=20)#,range=[0,2]

        plt.savefig(png_filename)
        plt.close()

    def select_long_displacement_arrow(self):
        R"""
        output:
            uv: vectors of arrow, selected by ids only.
            ids: selected by displacements(>0.2)
        """
        #normalize arrow length part1
        uv = self.displacements
        dr2 = uv*uv
        dr = np.sqrt(dr2[:,0]+dr2[:,1])
        
        #select particles with long displacement
        dr_long_list = dr[:]>1.0#2.5(3*lcr),1.0(sigma),0.2(precise)
        ids=dr_long_list
        uv = uv[ids]

        #tune the length of arrow
        scale_arrow =1
        uv = uv*scale_arrow
        
        return uv,ids

    def compare_displacement_method(self,uv,ids,frame_index_start=0,frame_index_end=-1):
        R"""
        introduction:
            compare the displacements from start to set frame and 
            the displacements from start to end. hence a rank of frame for particles 
            who arrive destinations earlier or later
        return:
            rank_relative: an array of frame [j]. relative to the frame to start, the i-th particle arrive whose
            destination at the j-th frame.
        """
        ids_active = np.array(ids)
        num_ids_active = np.sum(ids_active.astype(int))
        rank_record = np.zeros((num_ids_active,frame_index_end-frame_index_start))
        rank = np.zeros((num_ids_active,))
        list_ids_active = np.zeros((num_ids_active,))
        #get list_ids_active
        j = 0
        for id in range(len(ids_active)):
            if ids_active[id]:
                list_ids_active[j]=id
                j=j+1
                if j==num_ids_active:
                    break
        #rank = np.zeros((np.shape(uv)[0],))

        j = 0
        for dframe in range(frame_index_end-frame_index_start):
            temp_frame_index_end = frame_index_start+dframe+1
            self.get_displacements(temp_frame_index_end,frame_index_start)
            duv = uv - self.displacements[ids,:]#
            duv2 = duv*duv
            druv = np.sqrt(duv2[:,0]+duv2[:,1])
            arrive_bool = druv[:]<1.0
            rank_record[:,j] = arrive_bool
            j=j+1
        #rank_frame = frame_index_end+1-np.sum(rank_record,1)#frame1,frame2,frame3
        rank_relative = frame_index_end-frame_index_start+1-np.sum(rank_record,1)#1,2,3
        rank_normalized = rank_relative/np.max(rank_relative)
        #print(rank_frame)
        return rank_relative#normalized

class bond_plot_module:
    def __init__(self,fig=None,ax=None):
        if ax is None:
            self.fig,self.ax = plt.subplots()
        else:
            self.fig = fig
            self.ax = ax
        
    def restrict_axis_property_relative(self,x_unit=None,hide_axis=False):#,xy
        R"""
        Parameters:
            x_unit: '($/sigma$)','(um)'
            txyz: all the particle positions, no one removed.
            bond_length: [particle_id1,particle_id2, bond_length] for txyz.
            check: limit the shortest and longest bond( in bond_length) to draw.
            png_filename: "prefix/bond_plot_index1513.png"
        weight of shapes:
            bond(blue line) < particles(black circle) < neighbor_change(orange circle) < traps(red cross)
            0   1   2   3   
        """
        #if dis is None:
        """self.points = xy
        xmax = max(self.points[:,0]) #- 3
        ymax = max(self.points[:,1]) #- 3
        xmin = min(self.points[:,0]) #+ 3
        ymin = min(self.points[:,1]) #+ 3
        dis = min(xmax-xmin,ymax-ymin)/2.0#half of the length of the system.
        dis = dis - 0 #cut the edge if necessary(eg. size & scale of images not match)

        center = [(xmax+xmin)*0.5,(ymax+ymin)*0.5]
        center_origin_distance = np.abs(np.dot(center,center))
        if  center_origin_distance < 1.0:# center is really close to origin
            center = [0,0]
        self.ax.set_xlim(-dis+center[0],dis+center[0])#plt.xlim(-dis+center[0],dis+center[0])
        self.ax.set_ylim(-dis+center[1],dis+center[1])#plt.ylim(-dis+center[1],dis+center[1])"""
        #draw a figure with edges
        if not x_unit is None:
            """
            plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
            })
            """
            self.ax.set_xlabel('x'+x_unit)  # Add an x-label to the axes.
            self.ax.set_ylabel('y'+x_unit)  # Add a y-label to the axes.
            self.ax.set_title("bond_length: vertices"+x_unit)  # Add a title to the axes
        self.ax.set_aspect('equal','box')#plt.axis('equal')
        if hide_axis:
            #hide xy-axis
            self.ax.set_xticks([])
            self.ax.set_yticks([])
        """
        the displayed image size will be the smaller one 
        between axis limitation for xlim/ylim or data itself. 
        Hence the effective xlim/ylim should be set smaller than data.

        To ensure that xlim/ylim < data region, 
        we add physically meaningless points.
        """
        #restrict ticks to show
        """
        #(let all the images share the same size)
        new_ticks = np.linspace(-dis,dis,int(2*dis+1))
        new_ticks = new_ticks.astype(int)
        #print(new_ticks.astype(str))
        #print((new_ticks))
        plt.xticks(new_ticks,new_ticks.astype(str))
        plt.yticks(new_ticks,new_ticks.astype(str))
        """
        #restrict data region to show
        self.x_unit = x_unit

    def restrict_axis_limitation(self,xlim,ylim):
        #restrict data region to show
        self.ax.set_xlim(xlim[0],xlim[1])
        self.ax.set_ylim(ylim[0],ylim[1])

    def draw_points_with_given_bonds(self,xy,list_bonds_index=None,particle_size=None,bond_color='b',particle_color='k',bond_width=None):
        R"""
        Parameters:
            xy: particle positions of a frame, with no one removed.
            bond_length: [particle_id1,particle_id2, bond_length] for txyz.
            vertex_bonds_index: [particle_id1,particle_id2]
        weight of shapes:
            bond(blue line) < particles(black circle) < neighbor_change(orange circle) < traps(red cross)
            0   1   2   3   
        Examples:
        """
        #list_short_ridge_bool = self.voronoi.ridge_length[:] <= self.ridge_first_minima_left
        #list_short_bonds = self.voronoi.ridge_points[np.logical_not(list_short_ridge_bool)]
        self.points = xy
        #add lines for edges
        for i in range(np.shape(list_bonds_index)[0]):
            bondx,bondy = [self.points[list_bonds_index[i],0],self.points[list_bonds_index[i],1]]
            line = lines.Line2D(bondx,bondy,color= bond_color,linewidth=bond_width)
            #pt1,pt2 = [self.points[list_bonds_index[i,0]],self.points[list_bonds_index[i,1]]]
            #line = plt.Polygon([pt1,pt2], closed=None, fill=None, edgecolor= bond_color,linewidth=bond_width)
            self.ax.add_line(line)#plt.gca().add_line(line)

        if not particle_size is None:
            self.ax.scatter(xy[:,0],xy[:,1],color=particle_color,zorder=1,s=particle_size)
        else:
            self.ax.scatter(xy[:,0],xy[:,1],color=particle_color,zorder=2)#,s=particle_size

    def get_bonds_with_conditional_ridge_length(self,ridge_length,ridge_points,ridge_first_minima_left):       
        R"""
        Introduction:
            In Voronoi cells, remove short ridges(local long bonds) and 
            reserve long ridges(local short bonds) to show polygons formed by particles. 
        """
        list_short_ridge_bool = ridge_length[:] <= ridge_first_minima_left
        list_bond_index = ridge_points[np.logical_not(list_short_ridge_bool)]
        return list_bond_index

    def get_bonds_with_conditional_bond_length(self,bond_length,bond_length_limmit=[0.9,2.0]):
        R"""
        Introduction:
            In Delauny triangulation, remove long bonds and 
            reserve short bonds to show polygons formed by particles. 
        """
        list_longer = bond_length[:,2] >= bond_length_limmit[0]
        list_shorter = bond_length[:,2] < bond_length_limmit[1]
        list_bond_bool = np.logical_and(list_longer,list_shorter)
        list_bond_index = bond_length[list_bond_bool,0:2].astype(int)
        return list_bond_index
    
    def get_bonds_with_longer_bond_length(self,bond_length=None,bond_length_limmit=2.0):
        R"""
        Introduction:
            In Delauny triangulation, remove long bonds and 
            reserve short bonds to show polygons formed by particles. 
        """
        list_longer = bond_length[:,2] > bond_length_limmit
        list_bond_bool = list_longer
        list_bond_index = bond_length[list_bond_bool,0:2].astype(int)
        return list_bond_index

    def get_bonds_with_machine_learning_idea(self,bond_length,ridge_length):
        R"""
        Machine learning based Clustering algorithm：
        cost function = normalized Ncluster + normalized sum(deviation of cluster i ) 
        scan r within [p(r) first minima, ] (Nparticle,σ2(particle)) as the normalizing parameters.
        （如果只是大团簇剪枝，就是HCS_clustering_algorithm）
        初始粒子团簇不应该标0，应该是0-n-1，然后每次向下取团簇编号。
        （要规定由短到长合并团簇吗？那就是PH算法）
        这样随着r-ridge增大，团簇数从n-1开始下降，总方差从0开始上升。综合方法可以处理p(r) 粗糙，1st minima难找问题吗？
        （由短到长合并团簇，不依赖分布？(Nparticle,id rank)*(Nridge,length rank)）能解决稀疏团簇和密集团簇互不干扰吗？
        （监测σ2(cluster i)随Nmerge的异常增长，让不同cluster互不干扰。）
        从0开始的团簇编号，意味着0号团簇全是开放结构，是边缘，应该被切边。（ridge法没切长bond，bond法去超长bond？）
        """

        pass

    def plot_neighbor_change(self,xy_stable,nb_change):
        R"""
        txyz_stable:array[Nframes,Nparticles,xyz]
                for simu data, 
                    ensure that particles never move across the boundary(box)!
                for exp data, 
                    unit of xyz must be um! 
                    ensure that particles are always in the vision field!
        nb_change: particle ids( in txyz_stable) which change neighbors.
        """
        self.ax.scatter(xy_stable[nb_change,0],xy_stable[nb_change,1],color='orange',zorder=2)
            
    def plot_traps(self,traps=None,trap_filename=None,LinearCompressionRatio=1.0,mode='map',trap_color='r',trap_size=1):
        R"""
        trap_filename:
                '/home/remote/hoomd-examples_0/testhoneycomb3-8-12'
                '/home/remote/hoomd-examples_0/testhoneycomb3-8-12-part1'
                '/home/remote/hoomd-examples_0/testkagome3-11-6'
                '/home/remote/hoomd-examples_0/testkagome_part3-11-6'
        mode: 'array'(scatter) or 'map'(pcolormesh)
        traps: n rows of [x,y] recording the positions of traps.
        """
        if trap_filename is None:
            if traps is None:
                print('Error: No traps info input!')
        else:
            traps=np.loadtxt(trap_filename)
            traps=np.multiply(traps,LinearCompressionRatio)
        if mode=='array':
            #x_scale = 200
            self.ax.scatter(traps[:,0], traps[:,1],c=trap_color,zorder=3,s=trap_size)#,marker = 'x',s=x_scale
        elif mode=='map':
            """
            #get points
            N = 256
            vals = np.ones((N, 4))
            vals[:, 0] = np.linspace(1, 1, N)
            vals[:, 1] = np.linspace(1, 128/256, N)
            vals[:, 2] = np.linspace(1, 128/256, N)
            newcmp = ListedColormap(vals)#LinearSegmentedColormap(vals)#ListedColormap(vals)
            
            cmp = plt.get_cmap('autumn')
            cmp.reversed('autumn_r')
            """
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

    def save_figure(self,png_filename):
        R"""
        parameter:
            png_filename: "prefix/bond_plot_index1513.png"

        if latex is used in matplolib,
        'pip install latex' is necessary for plt.savefig()
        """
        self.fig.savefig(png_filename,bbox_inches='tight')#plt.savefig(png_filename)
        plt.close('all')#self.fig,plt.close() # closes the current active figure

class bond_plot_module_for_image:
    def __init__(self,image):
        self.fig,self.ax = plt.subplots()

        self.ax.imshow(image,zorder=-1)
        
    def restrict_axis_property_relative(self,xy,x_unit='(um)'):
        R"""
        Parameters:
            txyz: all the particle positions, no one removed.
            bond_length: [particle_id1,particle_id2, bond_length] for txyz.
            check: limit the shortest and longest bond( in bond_length) to draw.
            png_filename: "prefix/bond_plot_index1513.png"
        weight of shapes:
            bond(blue line) < particles(black circle) < neighbor_change(orange circle) < traps(red cross)
            0   1   2   3   
        """
        
        #if dis is None:
        self.points = xy
        xmax = max(self.points[:,0]) #- 3
        ymax = max(self.points[:,1]) #- 3
        xmin = min(self.points[:,0]) #+ 3
        ymin = min(self.points[:,1]) #+ 3
        dis = min(xmax-xmin,ymax-ymin)/2.0#half of the length of the system.
        dis = dis - 0 #cut the edge if necessary(eg. size & scale of images not match)

        center = [(xmax+xmin)*0.5,(ymax+ymin)*0.5]
        center_origin_distance = np.abs(np.dot(center,center))
        if  center_origin_distance < 1.0:# center is really close to origin
            center = [0,0]

        #draw a figure with edges
        if x_unit == '(sigma)':
            """
            plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
            })
            """
            x_unit = '($\sigma$)'#pip install latex is necessary for plt.savefig
        self.ax.set_aspect('equal','box')#plt.axis('equal')
        self.ax.set_xlabel('x'+x_unit)  # Add an x-label to the axes.
        self.ax.set_ylabel('y'+x_unit)  # Add a y-label to the axes.
        """
        the displayed image size will be the smaller one 
        between axis limitation for xlim/ylim or data itself. 
        Hence the effective xlim/ylim should be set smaller than data.

        To ensure that xlim/ylim < data region, 
        we add physically meaningless points.
        """
        #restrict ticks to show
        """
        #(let all the images share the same size)
        new_ticks = np.linspace(-dis,dis,int(2*dis+1))
        new_ticks = new_ticks.astype(int)
        #print(new_ticks.astype(str))
        #print((new_ticks))
        plt.xticks(new_ticks,new_ticks.astype(str))
        plt.yticks(new_ticks,new_ticks.astype(str))
        """
        #restrict data region to show
        self.ax.set_xlim(-dis+center[0],dis+center[0])#plt.xlim(-dis+center[0],dis+center[0])
        self.ax.set_ylim(-dis+center[1],dis+center[1])#plt.ylim(-dis+center[1],dis+center[1])
        self.x_unit = x_unit

    def restrict_axis_limitation(self,xlim,ylim):
        #restrict data region to show
        self.ax.set_xlim(xlim[0],xlim[1])
        self.ax.set_ylim(ylim[0],ylim[1])

    def draw_points_with_conditional_bond(self,xy,bond_length=None,bond_length_limmit=[0.9,2.0],particle_size=None):
        R"""
        Parameters:
            xy: particle positions of a frame, with no one removed.
            bond_length: [particle_id1,particle_id2, bond_length] for txyz.
            bond_length_limmit: limit the shortest and longest bond( in bond_length) to draw.
        weight of shapes:
            bond(blue line) < particles(black circle) < neighbor_change(orange circle) < traps(red cross)
            0   1   2   3   
        Examples:
            import points_analysis_2D as pa
            s = "/home/tplab/Downloads/"
            index_num = 1387
            index_name = "index"+str(index_num)
            fname = s+index_name
            bb = pa.PointsAnalysis2D(filename=fname)
            oname1 = s+"bond_hist_"+index_name
            bb.draw_bond_length_distribution_and_first_minima(png_filename=oname1)
            oname2 = s+"bond_plot_"+index_name
            bb.draw_bonds_conditional_bond(check=[0.9, bb.bond_first_minima_left], png_filename=oname2)
        """
        if not (bond_length is None):
            bond_check= tuple([bond_length_limmit[0],bond_length_limmit[1]])
            #add lines for edges
            for i in range(np.shape(bond_length)[0]):
                if (bond_length[i,2] > bond_check[0])&(bond_length[i,2] < bond_check[1]) :
                    edge = tuple(bond_length[i,0:2].astype(int))
                    pt1,pt2 = [self.points[edge[0]],self.points[edge[1]]]
                    line = plt.Polygon([pt1,pt2], closed=None, fill=None, edgecolor='b',zorder=0,lw=0.5)#,lineStyle='dashed'
                    self.ax.add_line(line)
            self.ax.set_title("bond_length:"+str(np.around(bond_length_limmit,2))+self.x_unit)  # Add a title to the axes

        self.points = xy
        if not particle_size is None:
            self.ax.scatter(xy[:,0],xy[:,1],color='k',zorder=1,s=particle_size)
        else:
            pass#self.ax.scatter(xy[:,0],xy[:,1],color='k',zorder=1)
        return line

    def plot_neighbor_change(self,xy_stable,nb_change):
        R"""
        txyz_stable:array[Nframes,Nparticles,xyz]
                for simu data, 
                    ensure that particles never move across the boundary(box)!
                for exp data, 
                    unit of xyz must be um! 
                    ensure that particles are always in the vision field!
        nb_change: particle ids( in txyz_stable) which change neighbors.
        """
        self.ax.scatter(xy_stable[nb_change,0],xy_stable[nb_change,1],color='orange',zorder=2)
            
    def plot_traps(self,trap_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12-part1",LinearCompressionRatio=0.79,mode='array'):
        R"""
        trap_filename:
                '/home/remote/hoomd-examples_0/testhoneycomb3-8-12'
                '/home/remote/hoomd-examples_0/testhoneycomb3-8-12-part1'
                '/home/remote/hoomd-examples_0/testkagome3-11-6'
                '/home/remote/hoomd-examples_0/testkagome_part3-11-6'
        mode: 'array'(scatter) or 'map'(pcolormesh)
        """
        traps=np.loadtxt(trap_filename)
        traps=np.multiply(traps,LinearCompressionRatio)
        if mode=='array':
            #x_scale = 200
            self.ax.scatter(traps[:,0], traps[:,1],c='r',marker = 'x',zorder=3)#,s=x_scale
        elif mode=='map':
            """
            #get points
            N = 256
            vals = np.ones((N, 4))
            vals[:, 0] = np.linspace(1, 1, N)
            vals[:, 1] = np.linspace(1, 128/256, N)
            vals[:, 2] = np.linspace(1, 128/256, N)
            newcmp = ListedColormap(vals)#LinearSegmentedColormap(vals)#ListedColormap(vals)
            
            cmp = plt.get_cmap('autumn')
            cmp.reversed('autumn_r')
            """
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

    def save_figure(self,png_filename):
        R"""
        parameter:
            png_filename: "prefix/bond_plot_index1513.png"

        if latex is used in matplolib,
        'pip install latex' is necessary for plt.savefig()
        """
        self.fig.savefig(png_filename)#plt.savefig(png_filename)
        plt.close() # closes the current active figure
        #del self.ax,self.fig

    def draw_bonds_conditional_bond(self,xy,bond_length,bond_length_limmit=[0.9,2.0],x_unit='(um)'):#compatible module
        R"""
        Parameters:
            txyz: all the particle positions, no one removed.
            bond_length: [particle_id1,particle_id2, bond_length] for txyz.
            check: limit the shortest and longest bond( in bond_length) to draw.
            png_filename: "prefix/bond_plot_index1513.png"
        weight of shapes:
            bond(blue line) < particles(black circle) < neighbor_change(orange circle) < traps(red cross)
            0   1   2   3   
        Examples:
            import points_analysis_2D as pa
            s = "/home/tplab/Downloads/"
            index_num = 1387
            index_name = "index"+str(index_num)
            fname = s+index_name
            bb = pa.PointsAnalysis2D(filename=fname)
            oname1 = s+"bond_hist_"+index_name
            bb.draw_bond_length_distribution_and_first_minima(png_filename=oname1)
            oname2 = s+"bond_plot_"+index_name
            bb.draw_bonds_conditional_bond(check=[0.9, bb.bond_first_minima_left], png_filename=oname2)
        """
        bond_check= tuple([bond_length_limmit[0],bond_length_limmit[1]])
        
        #if dis is None:
        self.points = xy
        xmax = max(self.points[:,0]) #- 3
        ymax = max(self.points[:,1]) #- 3
        xmin = min(self.points[:,0]) #+ 3
        ymin = min(self.points[:,1]) #+ 3
        dis = min(xmax-xmin,ymax-ymin)/2.0#half of the length of the system.
        dis = dis - 0 #cut the edge if necessary(eg. size & scale of images not match)

        center = [(xmax+xmin)*0.5,(ymax+ymin)*0.5]
        center_origin_distance = np.abs(np.dot(center,center))
        if  center_origin_distance < 1.0:# center is really close to origin
            center = [0,0]

        #draw a figure with edges
        if x_unit == '(sigma)':
            """
            plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Helvetica"
            })
            """
            x_unit = '($\sigma$)'#pip install latex is necessary for plt.savefig
        self.ax.set_aspect('equal','box')#plt.axis('equal')
        self.ax.set_xlabel('x'+x_unit)  # Add an x-label to the axes.
        self.ax.set_ylabel('y'+x_unit)  # Add a y-label to the axes.
        self.ax.set_title("bond_length:"+str(np.around(bond_length_limmit,2))+x_unit)  # Add a title to the axes
        """
        the displayed image size will be the smaller one 
        between axis limitation for xlim/ylim or data itself. 
        Hence the effective xlim/ylim should be set smaller than data.

        To ensure that xlim/ylim < data region, 
        we add physically meaningless points.
        """
        #restrict ticks to show
        """
        #(let all the images share the same size)
        new_ticks = np.linspace(-dis,dis,int(2*dis+1))
        new_ticks = new_ticks.astype(int)
        #print(new_ticks.astype(str))
        #print((new_ticks))
        plt.xticks(new_ticks,new_ticks.astype(str))
        plt.yticks(new_ticks,new_ticks.astype(str))
        """
        #restrict data region to show
        self.ax.set_xlim(-dis+center[0],dis+center[0])#plt.xlim(-dis+center[0],dis+center[0])
        self.ax.set_ylim(-dis+center[1],dis+center[1])#plt.ylim(-dis+center[1],dis+center[1])
        
        #add lines for edges
        for i in range(np.shape(bond_length)[0]):
            if (bond_length[i,2] > bond_check[0])&(bond_length[i,2] < bond_check[1]) :
                edge = tuple(bond_length[i,0:2].astype(int))
                pt1,pt2 = [self.points[edge[0]],self.points[edge[1]]]
                line = plt.Polygon([pt1,pt2], closed=None, fill=None, edgecolor='b',zorder=0)
                self.ax.add_line(line)
        """
        particle_ids = np.linspace(0,self.points.shape[0]-1,self.points.shape[0],dtype=int)
        particle_ids_str = particle_ids.astype(str)
        for j in particle_ids:
            plt.annotate(particle_ids_str[j],self.points[j])
        """
        self.ax.scatter(xy[:,0],xy[:,1],color='k',zorder=1)

class dynamical_facilitation_module:
    R"""
    example:
        import points_analysis_2D as pa
        df = pa.dynamical_facilitation_module()
        df.plot_displacement_t(0)
    """
    def __init__(self):
        pass
    
    def plot_displacement_t(self,trajectory,ax=None):
        R"""
        input:    
            trajectory: Nframes rows of [x,y,z], a trajectory for a particle
        """
        uv = trajectory - trajectory[0]
        dr2 = uv*uv
        dr = np.sqrt(dr2[:,0]+dr2[:,1])
        if ax is None:
            fig,ax = plt.subplots()
            ax.set_xlabel('t (k steps)')
            ax.set_ylabel('$\delta$r ($\sigma$)')
        ax.plot(dr)
        #plt.show()
        return ax
        """
        #test right
        traj = np.ones((10,3))
        t0 = [1,2,3]
        t0 = np.array(t0)
        dt = traj - t0
        print(dt)
        """
    def plot_scan_displacement_t(self,trajectory,dt=1,ax=None):
        R"""
        input:    
            trajectory: Nframes rows of [x,y,z], a trajectory for a particle
            dt: time interval to calculate displacement.
        """
        dr = self.scan_displacement_t(trajectory,dt)
        if ax is None:
            fig,ax = plt.subplots()
            ax.set_xlabel('t (k steps)')
            ax.set_ylabel('$\delta$r ($\sigma$)')
        ax.plot(dr)
        #plt.show()
        return ax
    
    def scan_displacement_t(self,trajectory,dt=1):
        R"""
        input:    
            trajectory: Nframes rows of [x,y,z], a trajectory for a particle
            dt: time interval to calculate displacement.
        """
        uv = trajectory[dt:] - trajectory[:-dt]
        dr2 = uv*uv
        sz = np.shape(trajectory)
        if len(sz)==2:    #single particle
            dr = np.sqrt(dr2[:,0]+dr2[:,1])
        elif len(sz)==3: #many particles
            dr = np.sqrt(dr2[:,:,0]+dr2[:,:,1])
        return dr

    def compute_atmsd_scan_t(self):#reference
        R"""
        method:
            scan t axis, 1 frame for each interval. here may exist some overlaps between two intervals who share the same length of time interval m.
        record_msd:[interval m, msd_m]
        k: the frame to start scan 
        m: time interval
        m_max: max time interval
        """
        m_max = 9000#int(0.5*self.frames)#to ensure the robustness of statistics
        #k_max = 0
        start = 1000
        self.record_msd=np.zeros((m_max-start+1,2))#[interval m, msd_m]
        
        list_m=np.arange(start,m_max+1)
        self.record_msd[:,0]=list_m
        for m in list_m:
            self.dm_xyz = self.txyz_stable[m:,:,:] - self.txyz_stable[:-m,:,:]
            sq_dm_xyz2 = np.square(self.dm_xyz)
            sum_sq_dm_r2 = np.sum(sq_dm_xyz2)
            self.record_msd[m-start,1]=sum_sq_dm_r2/self.dm_xyz.shape[0]/self.particles

    def plot_displacement_t_all(self,txyz_stable,dt=1,ax=None):#[x]
        R"""
        input:    
            trajectory: Nframes rows of [x,y,z], a trajectory for a particle
        """
        
        uv = txyz_stable[dt:] - txyz_stable[:-dt]#[Nframe,Nparticle,xyz]
        dr2 = uv*uv
        sz = np.shape(dr2)
        print(sz)
        dr = np.sqrt(dr2[:,0]+dr2[:,1])
    
    def coarse_grain(self):#[x]
        pass

    def get_facilitation_waiting_time_distributions_1(self,reference_occupation,prefix=None,io_only=False):#,png_filename=None
        R"""
        input:
            reference_occupation:
                (Nframe,Nreference,)[frame:int, reference_point: int, isvacancy:bool]
                Check if a reference point is a vacancy; 
                If so, check if its neighbors are occupied in future frames 
            prefix: directory to save file.
            io_only: not proceeding data, just draw a figure as result.
        output:
            txt file: 'list_waiting_time_1.txt' e.g.: 11111222334
        example:
            import points_analysis_2D as pa
            import numpy as np
            #get reference positions
            dfm = pa.dynamical_facilitation_module()
            pt_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12"
            pt = np.loadtxt(pt_filename)
            pt = pt[:,:2]*0.81
            #get box from gsd
            gsd_filename=None#"/home/remote/hoomd-examples_0/trajectory_auto4302_9.gsd"
            pgsd=pa.proceed_gsd_file(gsd_filename,'remote',4302,9)
            box = pgsd.box[:2]
            #get reference positions inside box.
            pt = dfm.cut_edge_of_positions(pt,box)


            txyz_filename="/home/remote/Downloads/4302_9/txyz_stable.npy"
            result_filename="/home/remote/Downloads/4302_9/reference_occupation_cut.npy"
            txyz = np.load(txyz_filename)
            dfm.get_reference_occupation(pt,txyz,result_filename,1)#[:1000]
            reference_occupation=np.load(result_filename)
            prefix = "/home/remote/Downloads/4302_9/reference_occupation/"
            dfm.get_reference_neighbors(pt)
            dfm.get_facilitation_waiting_time_distributions_1(reference_occupation,prefix)#,io_only=True
            #non-square figure not restrained ax.set_xlim(-dis+center[0],dis+center[0]);ax.set_ylim(-dis+center[1],dis+center[1])
            #dfm.plot_reference_occupation(pt,reference_occupation[:106],prefix)
        """
        if not io_only:
            sz = np.shape(reference_occupation)
            list_waiting_time = np.zeros(sz)
            #list_waiting_time = np.ones(sz)
            #list_waiting_time[:] = -list_waiting_time[:]

            for frame in range(sz[0]):
                for id in range(sz[1]):
                    if reference_occupation[frame,id]:
                        nb = self.search_reference_neighbors(id)#search_nb.
                        count_nb = len(nb)
                        list_waiting_time_fr_id = np.zeros((count_nb,))
                        for i in range(count_nb):
                            bool0 = bool(reference_occupation[frame,nb[i]])
                            for frame_s in range(sz[0]):
                                if frame_s > frame:
                                    bool1 = bool(reference_occupation[frame_s,nb[i]])
                                    if (bool0 != bool1):#flop event
                                        list_waiting_time_fr_id[i] = frame_s-frame
                                        break
                        
                        if np.max(list_waiting_time_fr_id)==0:
                            waiting_time_fr_id = 0
                        elif np.min(list_waiting_time_fr_id)==0:
                            #choose the time > 0
                            #print('error! waiting time is 0!')
                            list_waiting_time_fr_id_sort = np.sort(list_waiting_time_fr_id)
                            #from small to large
                            for wt in list_waiting_time_fr_id_sort:
                                if wt > 0:
                                    waiting_time_fr_id = wt
                                    break
                        elif np.min(list_waiting_time_fr_id)>0:
                            waiting_time_fr_id = np.min(list_waiting_time_fr_id)
                        list_waiting_time[frame,id] = waiting_time_fr_id
            lwt_filename = prefix + 'list_waiting_time_frame_id_1.npy'
            np.save(lwt_filename,list_waiting_time)
            sz = np.shape(list_waiting_time)
            list_waiting_time_1d = np.reshape(list_waiting_time,sz[0]*sz[1])
            list_waiting_time_sort = np.sort(list_waiting_time_1d)
            count_wt_max = sz[0]*sz[1]
            print(count_wt_max)
            for i in range(count_wt_max):
                if list_waiting_time_sort[i]>0:#The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                    #count_wt_1 = i
                    list_waiting_time_eff = list_waiting_time_sort[i:]
                    break
            #count_wt = count_wt_max - count_wt_1
            txt_filename = prefix + 'list_waiting_time_1.txt'
            np.savetxt(txt_filename,list_waiting_time_eff)
        else:
            txt_filename = prefix + 'list_waiting_time_1.txt'
            list_waiting_time_eff = np.loadtxt(txt_filename)

        if not prefix is None:
            fig,ax = plt.subplots()
            count_bins = ax.hist (list_waiting_time_eff,bins=200,log=True)#bins=20,range=[0,100]
            #plt.show()
            _count=count_bins[0]
            _bins=count_bins[1]
            fig2,ax2 = plt.subplots()
            ax2.loglog(_bins[1:],_count)#semilogy,loglog
            ax2.set_xlabel('waiting time (k steps)')
            ax2.set_ylabel('count (1)')
            png_filename = prefix + 'list_waiting_time_1_bin.png'
            plt.savefig(png_filename)
            txt_filename = prefix + 'list_waiting_time_1_bin.txt'
            cb = np.zeros((_count.size,2))
            cb[:,0] = _bins[1:]
            cb[:,1] = _count
            np.savetxt(txt_filename,cb)

    def get_facilitation_waiting_time_distributions_0(self,search_nb,reference_occupation,prefix=None):#[x]
        R"""
        flopping simultaneously (coarse grained step) is counted in waiting time
        """
        pass

    def cut_facilitation_waiting_time(self,list_waiting_time_frame_id,wait_max,prefix=None):#[x]
        R"""
        flopping simultaneously (coarse grained step) is counted in waiting time
        """
        sz = np.shape(list_waiting_time_frame_id)

        for frame in range(sz[0]):
            for id in range(sz[1]):
                if list_waiting_time_frame_id[frame,id]>=wait_max:
                    list_waiting_time_frame_id[frame,id]=0
        return list_waiting_time_frame_id

    def get_reference_neighbors(self,reference_positions):
        R"""
        Input: 
            reference_positions(Nreference,2)[x,y]
        Output: 
            activate search_reference_neighbors()
            List_reference_neighbors (Nreference,)[list_neighbors_id]

        example:
            import numpy as np
            import data_analysis_cycle as dac
            trap_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12-part1"
            trap = np.loadtxt(trap_filename)
            pt_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12"
            pt = np.loadtxt(pt_filename)
        """
        self.get_nb = static_points_analysis_2d(points=reference_positions)
        self.get_nb.get_first_minima_bond_length_distribution(lattice_constant=3)
        #get_nb.draw_bonds_conditional_bond()
        self.get_nb.get_coordination_number_conditional()#__coordination_bond recorded
        
    def search_reference_neighbors(self,id):
        R"""
        input:
            id
        output:
            neighobrs
        caution: function get_reference_neighbors has been runned.
        """
        neighbors = self.get_nb.search_neighbor_id(id)
        return neighbors
        
    def get_reference_occupation(self,reference_positions,txyz,result_filename=None,r_trap=1):#,txyz
        R"""
        Input: 
            txyz(Nframe,Nparticle,2)[x,y]
        Output: 
            reference_occupation: save .npy file recording list_reference_occupation
        Variables:
            list_reference_occupation (Nframe,Nreference,)[frame:int, reference_point: int, occupied:bool]        
            bond_length: (Nref,Nxy)[Length_ij]
            ref_dis: (Nreference,)
        example:
            import numpy as np
            import points_analysis_2D as pa
            pt_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12"
            pt = np.loadtxt(pt_filename)
            dfm = pa.dynamical_facilitation_module()
            dfm.get_reference_neighbors(pt)`
        """
        txy = txyz[:,:,:2]
        reference_positions = reference_positions[:,:2]
        sz_txy = np.shape(txy)
        sz_ref = np.shape(reference_positions)
        list_reference_occupation = np.zeros((sz_txy[0],sz_ref[0]))
        for frame in range(sz_txy[0]):
            xy = txy[frame]
            self.bond_length_temp=distance.cdist(reference_positions,xy,'euclidean')
            #print(self.bond_length_temp)
            
            ref_dis = np.zeros((sz_ref[0],))
            for irow in range(sz_ref[0]):
                min_irow = np.min(self.bond_length_temp[irow])
                ref_dis[irow] = min_irow
            #print(ref_dis)
            ref_dis[:] = ref_dis[:]>r_trap# 0 for committment; 1 for active
            #print(ref_dis)
            list_reference_occupation[frame] = ref_dis
        print('size of list_reference_occupation:')
        print(np.shape(list_reference_occupation))
        if not result_filename is None:
            np.save(result_filename,list_reference_occupation)
        #np.min()
        #print()
        """
        shp=np.shape(self.voronoi.ridge_points)
        self.bond_length = np.zeros((shp[0],shp[1]+1))#(start_point,end_point,length)
        self.bond_length_temp=distance.cdist(self.voronoi.points[self.voronoi.ridge_points[:,0],:], self.voronoi.points[self.voronoi.ridge_points[:,1],:], 'euclidean')
        self.bond_length[:,0:2]=self.voronoi.ridge_points
        self.bond_length[:,2]=self.bond_length_temp.diagonal()
        """
    
    def get_pin_bool(self,reference_positions,txyz,result_prefix=None,r_trap=1,ispin=True):
        R"""
        Input: 
            reference_positions: (Ntrap,2)[x,y]
            txyz:(Nframe,Nparticle,2)[x,y]
        Output: 
            list_pin_bool: save .npy file recording list_pin_bool
                (Nframe,Nparticle)[bool] means particle_id at frame_i is pinned at reference_position or not.
        Variables:
            bond_length_temp: (Nframe,Nparticle,)[frame:int, reference_point: int, occupied:bool]        

        example:
            import points_analysis_2D as pa
            import numpy as np
            dfm = pa.dynamical_facilitation_module()
            lcr = 0.81
            file_txyz = '/home/remote/Downloads/4302_9/txyz.npy'
            file_reference_points = '/home/remote/hoomd-examples_0/testhoneycomb3-8-12'
            txyz = np.load(file_txyz)
            reference_points = np.loadtxt(file_reference_points)
            reference_points_lcr = np.multiply(lcr,reference_points) 
            dfm.get_pin_bool(reference_points_lcr,txyz)
        """
        txy = txyz[:,:,:2]
        reference_positions = reference_positions[:,:2]
        sz_txy = np.shape(txy)
        sz_ref = np.shape(reference_positions)
        list_pin_bool = np.zeros((sz_txy[0],sz_txy[1]))
        for frame in range(sz_txy[0]):
            xy = txy[frame]
            bond_length_temp = distance.cdist(reference_positions,xy,'euclidean')
            bond_length_sort = np.sort(bond_length_temp,axis=0)
            #print(self.bond_length_temp)
            list_pin_bool_1 = bond_length_sort[0]<r_trap# 0 for free; 1 for pin
            #print(ref_dis)
            list_pin_bool[frame] = list_pin_bool_1
        print('size of list_reference_occupation:')
        print(np.shape(list_pin_bool))
        if not result_prefix is None:
            if ispin:
                result_filename = result_prefix+'t_pin_bool'
            else:
                result_filename = result_prefix+'t_interstitial_bool'
            np.save(result_filename,list_pin_bool)
        return list_pin_bool
    
    def get_pin_id_bool(self,reference_positions,txyz,result_prefix=None,r_trap=0.5):
        R"""
        Input: 
            reference_positions: (Ntrap,2)[x,y]
            txyz:(Nframe,Nparticle,2)[x,y]
        Output: 
            list_pin_bool: save .npy file recording list_pin_bool
                (Nframe,Nparticle)[bool] means particle_id at frame_i is pinned at reference_position or not.
        Variables:
            bond_length_temp: (Nframe,Nparticle,)[frame:int, reference_point: int, occupied:bool]        

        example:
        """
        txy = txyz[:,:,:2]
        reference_positions = reference_positions[:,:2]
        sz_txy = np.shape(txy)
        sz_ref = np.shape(reference_positions)
        list_pin_id_bool = np.zeros((sz_txy[0],sz_txy[1],sz_ref[0]))
        for frame in range(sz_txy[0]):
            xy = txy[frame]
            bond_length_temp = distance.cdist(xy,reference_positions,'euclidean')
            #bond_length_sort = np.sort(bond_length_temp,axis=1)
            #print(self.bond_length_temp)
            list_pin_bool_1 = bond_length_temp[:,:]<r_trap# 0 for free; 1 for pin
            #print(ref_dis)
            list_pin_id_bool[frame] = list_pin_bool_1
        print('size of list_reference_occupation:')
        print(np.shape(list_pin_id_bool))
        if not result_prefix is None:
            result_filename = result_prefix+'t_pin_id_bool'
            np.save(result_filename,list_pin_id_bool)
        return list_pin_id_bool

    def get_interstitial_bool(self,reference_positions,txyz,result_prefix=None,r_trap=1):
        R"""
            for a triangle whose edge length A is 1, the center-vertex distance Rcv is sqrt(3)/3=0.577.
            if A is 3, Rcv is sqrt(3)=1.73, Rcv-1 is 0.73. so r_trap=1 is slightly too large 
        example:
            import points_analysis_2D as pa
            import numpy as np
            dfm = pa.dynamical_facilitation_module()
            lcr = 0.81
            file_txyz = '/home/remote/Downloads/4302_9/txyz.npy'#_stable
            file_reference_points = '/home/remote/hoomd-examples_0/testhoneycomb3-8-12'
            txyz = np.load(file_txyz)
            reference_points = np.loadtxt(file_reference_points)
            reference_points_lcr = np.multiply(lcr,reference_points) 
            result_prefix='/home/remote/Downloads/4302_9/pin_check/'
            rt=0.5
            t_pin_bool = dfm.get_pin_bool(reference_points_lcr,txyz,result_prefix,r_trap=rt)
            t_interstitial_bool = dfm.get_interstitial_bool(reference_points_lcr,txyz,result_prefix,r_trap=rt)
            #t_pin_bool = np.load(result_prefix+'t_pin_bool.npy')
            #t_interstitial_bool = np.load(result_prefix+'t_interstitial_bool.npy')
            #check the overlap of pinning and interstitial particles. calculate the transformation ratio too.
            t_transition_bool = t_interstitial_bool.astype(int) + t_pin_bool.astype(int)
            if max(t_transition_bool.all())>1:
                print('overlap of pinning and interstitial particles!')
            sz_list = np.shape(t_transition_bool)
            t_trans_count = np.sum(t_transition_bool,axis=1)

            import matplotlib.pyplot as plt
            fig,ax = plt.subplots()
            ax.plot(t_trans_count/sz_list[1])
            png_filename = '/home/remote/Downloads/4302_9/pin_check/t_trans_ratio_allpart_r05.jpg'
            plt.savefig(png_filename)
            plt.close()
        """
        ref_pos = static_points_analysis_2d(reference_positions)
        dual_pt = ref_pos.voronoi.vertices
        list_interstitial_bool = self.get_pin_bool(dual_pt,txyz,result_prefix,r_trap,ispin=False)
        return list_interstitial_bool

    def get_trans_ratio(self,t_state_bool):
        sz_list = np.shape(t_state_bool)
        t_trans_count = np.sum(t_state_bool,axis=1)
        t_trans_ratio = t_trans_count/sz_list[1]
        return t_trans_ratio

    def get_activation_event(self,t_state_bool,dt=1):
        R"""
        input:
            t_state_bool: (Nframe,Nparticle) t_pin_bool or t_state_bool
        output:
            t_act_bool: (Nframe-dt,Nparticle)
            
            dt = 0
            t = 0
            sz = np.size(trajectory)#(Nframe,Ndimension)
            tt = ta/2 - dt
            uv = trajectory[t+tt] - trajectory[t-tt]
            dr2 = uv*uv
            dr = np.sqrt(dr2[:,0]+dr2[:,1])
            abs(dr[t+tt] - dr[t-tt])
        """
        """
        #method1:
        t_act_bool = t_state_bool[dt:] - t_state_bool[:-dt]
        sz_list = np.shape(t_act_bool)
        t_trans_count = np.sum(t_act_bool,axis=1)
        t_trans_ratio = t_trans_count/sz_list[1]
        #method2:xor, count any changes, including positive and negative ones.
        t_act_bool = np.logical_xor(t_state_bool[dt:],t_state_bool[:-dt]) 
        sz_list = np.shape(t_act_bool)
        t_trans_count = np.sum(t_act_bool,axis=1)
        t_trans_ratio = t_trans_count/sz_list[1]
        """
        #method3:xor, count positive changes.
        t_act_bool = t_state_bool[dt:] - t_state_bool[:-dt]
        t_act_bool = t_act_bool[:,:] == 1
        sz_list = np.shape(t_act_bool)
        t_trans_count = np.sum(t_act_bool,axis=1)
        t_trans_ratio = t_trans_count/sz_list[1]
        return t_trans_ratio

    def get_pin_depin_event(self,t_state_id_bool,dt=1):
        R"""
        input:
            t_state_bool: (Nframe,Nparticle) t_pin_bool or t_state_bool
        output:
            t_act_bool: (Nframe-dt,Nparticle)
            
            dt = 0
            t = 0
            sz = np.size(trajectory)#(Nframe,Ndimension)
            tt = ta/2 - dt
            uv = trajectory[t+tt] - trajectory[t-tt]
            dr2 = uv*uv
            dr = np.sqrt(dr2[:,0]+dr2[:,1])
            abs(dr[t+tt] - dr[t-tt])
        """
        """
        #method1:
        t_act_bool = t_state_bool[dt:] - t_state_bool[:-dt]
        sz_list = np.shape(t_act_bool)
        t_trans_count = np.sum(t_act_bool,axis=1)
        t_trans_ratio = t_trans_count/sz_list[1]
        #method2:xor, count any changes, including positive and negative ones.
        t_act_bool = np.logical_xor(t_state_bool[dt:],t_state_bool[:-dt]) 
        sz_list = np.shape(t_act_bool)
        t_trans_count = np.sum(t_act_bool,axis=1)
        t_trans_ratio = t_trans_count/sz_list[1]
        """
        #method3:xor, count positive changes.
        t_act_bool = t_state_id_bool[dt:] - t_state_id_bool[:-dt]
        sz_list = np.shape(t_act_bool)

        t_pin_id_bool = t_act_bool[:,:] == 1
        t_depin_id_bool = t_act_bool[:,:] == -1

        t_pin_bool=np.sum(t_pin_id_bool,axis=2)
        t_pin_count=np.sum(t_pin_bool,axis=1)
        t_pin_ratio=t_pin_count/sz_list[1]
        t_depin_bool=np.sum(t_depin_id_bool,axis=2)
        t_depin_count=np.sum(t_depin_bool,axis=1)
        t_depin_ratio=t_depin_count/sz_list[1]
        
        return t_pin_ratio,t_depin_ratio
    
    def plot_transition_state(self,t_trans_ratio,t_pin_ratio,t_interstitial_ratio):
        #import matplotlib.pyplot as plt
        fig,ax = plt.subplots()
        dt=0
        np.linspace(dt,2000,2001-dt)
        ax.semilogx(t_trans_ratio,label='trans_ratio')#semilogy,plot
        ax.semilogx(t_pin_ratio,label='pin_ratio')
        ax.semilogx(t_interstitial_ratio,label='inter_ratio')
        plt.legend()
        #plt.title('CN_k '+'index:'+str_index)
        plt.xlabel('time(steps)')
        plt.ylabel('trans_ratio(1)')
        png_filename = '/home/remote/Downloads/4302_9/pin_check/t_trans_pin_inter_ratio_allpart_r05.jpg'
        plt.savefig(png_filename)
        plt.close()

    def plot_reference_occupation(self,reference_positions,reference_occupation,prefix=None):
        sz = np.shape(reference_occupation)
        for frame in range(sz[0]):
            fig,ax = plt.subplots()
            circle_size=50
            vac = reference_occupation[frame,:]==1
            occu = np.logical_not(vac) 
            lw=2
            ax.scatter(reference_positions[vac,0],reference_positions[vac,1],facecolors='none',edgecolors='k',marker='o',s=circle_size,linewidths=lw)
            ax.scatter(reference_positions[occu,0],reference_positions[occu,1],c='k',marker='o',s=circle_size)
            
            png_filename = prefix+'reference_occupation'+str(int(frame))+'.png'
            #scatter size
            #axis_equal
            x_unit = '($\sigma$)'#pip install latex is necessary for plt.savefig
            ax.set_aspect('equal','box')#plt.axis('equal')
            ax.set_xlabel('x'+x_unit)  # Add an x-label to the axes.
            ax.set_ylabel('y'+x_unit)  # Add a y-label to the axes.
            plt.savefig(png_filename)
            plt.close()
    
    def plot_reference_positions_waiting_time_colored(self,reference_positions,reference_occupation,prefix=None):
        R"""
        EXAMPLE:
            #get box from gsd
            gsd_filename=None#"/home/remote/hoomd-examples_0/trajectory_auto4302_9.gsd"
            pgsd=pa.proceed_gsd_file(gsd_filename,'remote',4302,9)
            box = pgsd.box[:2]
            pt_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12"
            pt = np.loadtxt(pt_filename)
            pt = pt[:,:2]*0.81
            dfm = pa.dynamical_facilitation_module()
            pt = dfm.cut_edge_of_positions(pt,box)
            print(np.shape(pt))
            txyz_filename="/home/remote/Downloads/4302_9/txyz_stable.npy"
            result_filename="/home/remote/Downloads/4302_9/reference_occupation_cut.npy"
            txyz = np.load(txyz_filename)
            #dfm.get_reference_occupation(pt,txyz,result_filename,1)#[:1000]
            reference_occupation=np.load(result_filename)
            prefix = "/home/remote/Downloads/4302_9/reference_occupation/"
            #dfm.get_reference_neighbors(pt)
            #dfm.get_facilitation_waiting_time_distributions_1(reference_occupation,prefix)#,io_only=True
            #non-square figure not restrained ax.set_xlim(-dis+center[0],dis+center[0]);ax.set_ylim(-dis+center[1],dis+center[1])
            dfm.plot_reference_occupation(pt,reference_occupation[:110],prefix)

            #draw scatter reference positions waiting time colored
            lwt_filename = prefix + 'list_waiting_time_frame_id_1.npy'
            list_waiting_time_frame_id = np.load(lwt_filename)
            list_waiting_time_frame_id = dfm.cut_facilitation_waiting_time(list_waiting_time_frame_id,5,prefix)
            dfm.plot_reference_positions_waiting_time_colored(pt,list_waiting_time_frame_id,prefix)
        """
        sz = np.shape(reference_occupation)
        for frame in range(sz[0]):
            if frame<=200:
                fig,ax = plt.subplots()
                circle_size=50
                mapp = ax.scatter(reference_positions[:,0],reference_positions[:,1],c=reference_occupation[frame],marker='o',s=circle_size)
                png_filename = prefix+'reference_waiting_time_color'+str(int(frame))+'.png'
                #scatter size
                #axis_equal
                x_unit = '($\sigma$)'#pip install latex is necessary for plt.savefig
                ax.set_aspect('equal','box')#plt.axis('equal')
                ax.set_xlabel('x'+x_unit)  # Add an x-label to the axes.
                ax.set_ylabel('y'+x_unit)  # Add a y-label to the axes.
                fig.colorbar(mapp,ax=ax)#plt.cm.ScalarMappable(norm=plt.autoscale()),
                #plt.colorbar()
                plt.savefig(png_filename)
                plt.close()
        
    def cut_edge_of_positions(self,points,box):
        R"""
        input:
            points: [x,y]
            box: [Lx,Ly] get from gsd file
        return:
            points_selected
        Variables:
            effective_lattice_constant： the distance between a particle and its neighbor
            edge_cut_positions_list: list the rows of particles' positions at given snapshot.
        example:
            gsd_filename="/home/remote/hoomd-examples_0/trajectory_auto4302_9.gsd"
            pgsd=pa.proceed_gsd_file(gsd_filename,'remote',4302,9)
            box = pgsd.box[:2]
        """
        xmax = box[0]/2
        ymax = box[1]/2
        xmin = -box[0]/2
        ymin = -box[1]/2
        
        #That directly using 'and' has been banned, so 'logical_and' is necessary
        list_xmin = points[:,0] > xmin
        list_xmax = points[:,0] < xmax
        list_ymin = points[:,1] > ymin
        list_ymax = points[:,1] < ymax
        list_x = np.logical_and(list_xmin,list_xmax)
        list_y = np.logical_and(list_ymin,list_ymax)
        list_xy = np.logical_and(list_x,list_y)

        #edge_cut_positions_list = np.where(list_xy)
        #edge_cut_positions_bool = list_xy 
        points_selected = points[list_xy]
        print('size of traps:')
        print(np.shape(points_selected))
        return points_selected

class polygon_analyzer_ml:
    def __init__(self,voronoi,ridge_first_minima_left,ridge_length_cutoff=5.0):
        R"""
        description:
            reorganize voronoi from static_points_analysis_2d, 
            remove ridges whose length are too long to to reasonable,
            and rearrange the indices to relink the ridges and vertices reasonable.
        parameters:
            self.voronoi: from static_points_analysis_2d, not tuned.
            self.voronoi.vertices[self.voronoi.ridge_vertices[:,0],:] #position of vertices
            self.voronoi.ridge_length#ridge_row -> [ridge_length]
            self.voronoi.ridge_points#ridge_row -> [start_point_row,end_point_row]
            self.voronoi.ridge_vertices#ridge_row -> [start_vertex_row,end_vertex_row]
        """
        self.voronoi = voronoi

        self.ridge_first_minima_left = ridge_first_minima_left
        #ridge cut operation, to remove those whose ridge_lengths are too to to be reasonable
        list_rational_ridge_bool = self.voronoi.ridge_length < ridge_length_cutoff
        self.ridge_length = self.voronoi.ridge_length[list_rational_ridge_bool]#value, right
        ridge_vertices = self.voronoi.ridge_vertices[list_rational_ridge_bool]#vertices index ?
        list_vertices_index = np.unique(ridge_vertices)
        self.vertices = self.voronoi.vertices[list_vertices_index]#vertices position, right
        
        #transform from old index to new index
        """
        list_rational_ridge_bool -> 0,1,2,3..len(self.ridge_length)
        list_vertices_index -> 0,1,2,3..len(self.vertices)
        ridge_vertices: row:ridge_index [vertex_index1,vertex_index2]

        vertices_index_old_new = np.zeros((len(list_vertices_index),2))
        vertices_index_old_new[:,0] = list_vertices_index
        vertices_index_old_new[:,1] = range(len(list_vertices_index))
        """
        vertices_index_old_new = np.array(list_vertices_index)
        
        #ridge_vertices_temp = np.zeros(np.shape(ridge_vertices))#vertices index ?
        ridge_vertices_temp = -np.ones(np.shape(ridge_vertices))
        #ridge_vertices_temp[:,:] = -ridge_vertices_temp[:,:]
        for j in range(2):
            for i in range(np.shape(ridge_vertices)[0]):
                row = np.where(vertices_index_old_new[:]==ridge_vertices[i,j] )
                ridge_vertices_temp[i,j] = row[0]#int(row)
        self.ridge_vertices = ridge_vertices_temp
        #self.ridge_points = self.voronoi.ridge_points[list_rational_ridge_bool]#points index ?

    def scan_conditional_bonds_and_simplices_ml(self,scan_mode='local_scan',png_filename=None,cost_mode='mean_var'):#[x]
        R"""
        input:
        ----------
            mode:
                'local_scan', scan a small range of ridge_length around self.ridge_first_minima_left,
                 forming vertices' clusters group, to check the overlap of local minima of cost and
                 ridge_first_minima_left.
                'global_scan', scan all the ridge_length to form vertices' clusters group, 
                to calculate the global minima of cost.
            png_filename:
                to save figure of ridge_length vs cost.

        return:
        ----------
            list_short_ridge_rank
            record_costs

        parameters:
        ----------
            vertex_bonds_index: n rows of [start_point_index, end_point_index]
            list_simplex_cluster: n_vertices rows of [simplex_id, cluster_id],
                where delaunay.simplices[simplex_id], 
                and cluster_id is the cluster the simplex belonging to
        method:
        ----------
            vertices cluster method: find the shortest ridges and link the related vertices.
        
        typical ridge length:
        ----------
            lattice     | bond_length   | ridge_length  | ratio
            hex         | 1             | sqrt(3)/3     | bl*lcr*rl 
            honeycomb   | 1             | sqrt(3)       |
            kagome      | 1             | 2/3*sqrt(3)   |

        """
        #organize_start
        ridge_length_sorted = np.sort(self.ridge_length)#small to large
        #ridge_length_rank#short to long, rank 0,1,2...
        list_short_ridge_bool = self.ridge_length[:] <= self.ridge_first_minima_left
        short_ridge_length_rank = np.sum(list_short_ridge_bool.astype(int)) - 1
        n_ridges = np.shape(list_short_ridge_bool)[0]
        ridge_length_rank_ratio = (short_ridge_length_rank+1)/n_ridges
        #scan a range around self.ridge_first_minima_left
        if scan_mode =='local_scan':
            scan_range_relative = 0.1
        elif scan_mode =='global_scan':
            scan_range_relative = 1.0
        scan_ridge_length_rank_min = (ridge_length_rank_ratio - scan_range_relative)*n_ridges
        scan_ridge_length_rank_min = self.check_scan_ridge_length_rank(scan_ridge_length_rank_min,n_ridges)
        scan_ridge_length_rank_max = (ridge_length_rank_ratio + scan_range_relative)*n_ridges
        scan_ridge_length_rank_max = self.check_scan_ridge_length_rank(scan_ridge_length_rank_max,n_ridges)
        list_short_ridge_rank = np.linspace(scan_ridge_length_rank_min,scan_ridge_length_rank_max-1,scan_ridge_length_rank_max-scan_ridge_length_rank_min)
        list_short_ridge_rank = list_short_ridge_rank.astype(int)
        
        record_costs = np.zeros( (np.shape(list_short_ridge_rank)[0],3) )
        ridge_rank_min_cost = 0
        min_cost = 1
        for ridge_rank in list_short_ridge_rank:
            ridge_first_minima_ml = ridge_length_sorted[ridge_rank] + 1e-5
            #use ridge_first_minima_ml replace self.ridge_first_minima_left
            #get n_clusters and sum(variance_cluster)
            list_polygon_rate  = self.get_conditional_bonds_and_simplices_ml(ridge_first_minima_ml)
            #self.vertex_bonds_index #short bonds index offered by ridge comapre method 
            #self.list_simplex_cluster #[vertex_id,cluster_id]
            index1 = int(ridge_rank-list_short_ridge_rank[0])
            if cost_mode == 'sum_var':
                record_costs[index1,0],record_costs[index1,1],record_costs[index1,2] \
                    = self.get_cost_function_cluster_ml_method_sum_var()
            elif cost_mode == 'mean_var':
                record_costs[index1,0],record_costs[index1,1],record_costs[index1,2] \
                    = self.get_cost_function_cluster_ml_method_mean_var()
            #refresh the minima of cost
            if record_costs[index1,0]<min_cost:
               ridge_rank_min_cost = ridge_rank
               min_cost = record_costs[index1,0]
        print('ridge_length = ',np.round(ridge_length_sorted[ridge_rank_min_cost],3),' is best\n')

        if not png_filename is None:
            fig,ax = plt.subplots()
            ax.plot(ridge_length_sorted,record_costs[:,0],'r',label='cost')#,legend
            ax.plot(ridge_length_sorted,record_costs[:,1],'g',label='$r(n_{cluster})$')#n_cluster_norm
            if cost_mode == 'sum_var':
                ax.plot(ridge_length_sorted,record_costs[:,2],'b',label='$r(\Sigma \sigma^2)$')#sum_var_norm
            elif cost_mode == 'mean_var':
                ax.plot(ridge_length_sorted,record_costs[:,2],'b',label='$r(\langle \sigma^2 \rangle)$')#mean_var_norm
                #'$r(\overline{\sigma^2})$' looks ugly
            ax.scatter(ridge_length_sorted[short_ridge_length_rank],record_costs[int(short_ridge_length_rank-list_short_ridge_rank[0]),0],c='k')
            #prefix = '/home/remote/Downloads/'
            #png_filename = 'costs.png'
            ax.legend()
            ax.set_xlabel('ridge_length(1)')
            ax.set_xlabel('cost(1)')
            fig.savefig(png_filename)#prefix+

        return list_short_ridge_rank,record_costs

    def get_conditional_bonds_and_simplices_ml(self,ridge_first_minima_ml):#[x]
        R"""
        return:
            vertex_bonds_index: n rows of [start_point_index, end_point_index]
            list_simplex_cluster: n_vertices rows of [simplex_id, cluster_id],
                where delaunay.simplices[simplex_id], 
                and cluster_id is the cluster the simplex belonging to
        method:
            vertices cluster method: find the shortest ridges and link the related vertices.
            pandas.array
            self.voronoi.vertices[self.voronoi.ridge_vertices[:,0],:] #position of vertices
            self.voronoi.ridge_length#ridge_row -> [ridge_length]
            self.voronoi.ridge_points#ridge_row -> [start_point_row,end_point_row]
            self.voronoi.ridge_vertices#ridge_row -> [start_vertex_row,end_vertex_row]
        """
        list_short_ridge_bool = self.ridge_length[:] <= ridge_first_minima_ml
        #list_short_bonds = self.ridge_points[np.logical_not(list_short_ridge_bool)]#[start_point_index, end_point_index]
        list_ridge_vertices_index = self.ridge_vertices[list_short_ridge_bool]
        list_vertices_index = np.unique(list_ridge_vertices_index)
        n_vertices_in_cluster = np.shape(list_vertices_index)[0]#count the vertices of short ridge to form cluster
        n_vertices_reasonable = np.shape(self.vertices)[0]#count all the rational vertices
        
        #record_vertices_cluster: n_vertices rows of [vertex_id, cluster_id],
        #where vertex_id belongs to list_vertices
        #when cluster_id=-1, the vertex has not been included by any cluster.
        record_vertices_cluster = np.ones((n_vertices_in_cluster,2))
        record_vertices_cluster[:,1]=range(n_vertices_in_cluster)#watch initial value, 0,1,2...
        record_vertices_cluster[:,0]=list_vertices_index
        #cluster_id = 0
        for i in range(n_vertices_in_cluster):
            vertex_id = list_vertices_index[i]
            #print("cluster_id\n",cluster_id)
            list_linked = np.where(list_ridge_vertices_index[:,:]==vertex_id)
            list_linked_vertices_pair = list_ridge_vertices_index[list_linked[0]]
            #print("vertices_pair\n",list_linked_vertices_pair)
            list_vertex_id_of_cluster1 = np.unique(list_linked_vertices_pair)
            cluster_id = list_vertex_id_of_cluster1[0]#set the cluster_id to merge cluster1
            for j in list_vertex_id_of_cluster1:
                record_id = record_vertices_cluster[:,0]==j#find the row_index for vertex_id
                #new cluster merge old one by refreshing a list of vertices in old cluster
                contradictory_cluster_id = record_vertices_cluster[record_id,1]
                list_bool_of_cluster_to_merge= (record_vertices_cluster[:,1]==contradictory_cluster_id)
                record_vertices_cluster[list_bool_of_cluster_to_merge,1]=cluster_id
            #cluster_id +=1
        #statistics for clusters
        list_cluster_id = np.unique(record_vertices_cluster[:,1])
        
        #if polygon_statistics:
        count_polygon = np.zeros((n_vertices_reasonable,2))#(10,2)
        for i in list_cluster_id:
            list_cluster_i = record_vertices_cluster[record_vertices_cluster[:,1]==i,0]
            n_complex_i = np.shape(list_cluster_i)[0]
            count_polygon[n_complex_i,0]=n_complex_i+2#[x]frame=1421,polygon=10;1609,12
            count_polygon[n_complex_i,1]+=1
        count_polygon[0,0]=2
        count_polygon[1,0]=3
        count_polygon[1,1]+= n_vertices_reasonable - n_vertices_in_cluster

        n_cluters_and_vertex = n_vertices_reasonable - n_vertices_in_cluster + len(list_cluster_id)#n_polygon also
        check_1 = False
        if check_1:
            print('n_cluters_and_vertex','n_polygon')
            print(n_cluters_and_vertex,sum(count_polygon[:,1]))
            np.sum(np.logical_not(list_short_ridge_bool).astype(int))#np.shape(list_short_bonds)[0]
        #print("polygon_n, count\n",count_polygon)
        count_polygon_relative = np.array(count_polygon)
        #n_polygons = sum(count_polygon[:,1])
        count_polygon_relative[:,1] = count_polygon[:,1]/n_vertices_reasonable \
                                        *(count_polygon[:,0]-2)*100#see one simplex as one weight, unit=100%
        #print("polygon_n, count%\n",count_polygon_relative)
        check_2 = True
        if check_2:
            count_max = np.max(count_polygon[:,1])
            row_domain = np.where(count_polygon[:,1]==count_max)
            print('domain polygon, domain_rate% ','n rest vertices')
            print(count_polygon_relative[row_domain],n_vertices_reasonable - n_vertices_in_cluster)
        self.count_polygon_number = count_polygon #[polygon_name, count_polygons]

        #self.vertex_bonds_index = list_short_bonds#short bonds index offered by ridge comapre method 
        self.list_simplex_cluster = record_vertices_cluster #n_vertices rows of [vertex_id, cluster_id],
        
        return count_polygon_relative#n_cluters_and_vertex#
        
    def check_scan_ridge_length_rank(self,scan_ridge_length_rank,n_ridges):
        if scan_ridge_length_rank < 0:
            scan_ridge_length_rank = 0
        elif scan_ridge_length_rank > n_ridges:
            scan_ridge_length_rank = n_ridges
        else:
            scan_ridge_length_rank = int(scan_ridge_length_rank)
        return scan_ridge_length_rank

    def get_cost_function_cluster_ml_method_sum_var(self):#,list_cluster_ids,vertex_cluster_positions[x]
        R"""
        introduction:
            cost = normalized_n_clusters + normalized_sum_variance_n_clusters
            
            self.vertex_bonds_index #short bonds index offered by ridge comapre method 
            self.list_simplex_cluster #[vertex_id,cluster_id]
        return:
            cost    
        """
        vertex_positions = np.array(self.vertices)
        #get n_clusters_norm
        n_vertices = np.shape(vertex_positions)[0]
        n_clusters_0 = n_vertices
        n_clusters = sum(self.count_polygon_number[:,1])#edges not cut ! [x]
        n_clusters_norm = n_clusters/n_clusters_0
        #get sum_var_norm
        sum_var_0 = np.var(vertex_positions[:,0])+np.var(vertex_positions[:,1])
        sum_var_0 = sum_var_0 *n_vertices#consider the contribution of each vertex
        list_cluster_ids_unique = np.unique( self.list_simplex_cluster[:,1])
        list_var_of_clusters = np.zeros((len(list_cluster_ids_unique),))#this is too large but okay anyway.
        #[here a new i array should be set to match cluster ids]
        for i in range(len(list_cluster_ids_unique)):
            cluster_id = list_cluster_ids_unique[i]
            list_vertices_indices_bool = self.list_simplex_cluster[:,1] == cluster_id#list_bool
            list_vertices_indices = np.array(self.list_simplex_cluster[list_vertices_indices_bool,0]).astype(int)
            n_vertices_in_cluster_id = len(list_vertices_indices)#np.sum .astype(int)
            list_var_of_clusters[i] = np.var(vertex_positions[list_vertices_indices,0])+np.var(vertex_positions[list_vertices_indices,1])
            list_var_of_clusters[i] = list_var_of_clusters[i] * n_vertices_in_cluster_id#consider the contribution of each vertex
        sum_var = np.sum(list_var_of_clusters)
        sum_var_norm = sum_var/sum_var_0
        cost = n_clusters_norm + sum_var_norm
        #print(n_clusters,n_clusters_norm,np.round(sum_var_norm,2))
        return cost,n_clusters_norm,sum_var_norm

    def get_cost_function_cluster_ml_method_mean_var(self):#
        vertex_positions = np.array(self.vertices)
        #get n_clusters_norm
        n_vertices = np.shape(vertex_positions)[0]
        n_clusters_0 = n_vertices
        n_clusters = sum(self.count_polygon_number[:,1])
        n_clusters_norm = n_clusters/n_clusters_0
        #get var_norm
        var_0 = np.var(vertex_positions[:,0])+np.var(vertex_positions[:,1])
        list_cluster_ids_unique = np.unique( self.list_simplex_cluster[:,1])
        list_var_of_clusters = np.zeros((len(list_cluster_ids_unique),))#this is too large but okay anyway.
        #[here a new i array should be set to match cluster ids]
        for i in range(len(list_cluster_ids_unique)):
            cluster_id = list_cluster_ids_unique[i]
            list_vertices_indices_bool = self.list_simplex_cluster[:,1] == cluster_id#list_bool
            list_vertices_indices = np.array(self.list_simplex_cluster[list_vertices_indices_bool,0]).astype(int)
            list_var_of_clusters[i] = np.var(vertex_positions[list_vertices_indices,0])+np.var(vertex_positions[list_vertices_indices,1])
        mean_var = np.sum(list_var_of_clusters)/n_clusters#mean cluster+vertex
        #np.mean(list_var_of_clusters)#mean cluster
        mean_var_norm = mean_var/var_0
        cost = n_clusters_norm + mean_var_norm
        #print(n_clusters,n_clusters_norm,np.round(sum_var_norm,2))
        return cost,n_clusters_norm,mean_var_norm


class show_cairo_order_parameter:
    def __init__(self) -> None:
        pass
    def compute_theoretical_diagram(self):
        R"""
        output:
            record: [cn3,cn4,order_parameter_for_cairo]
        example:
            import workflow_analysis as wa
            ca = wa.show_cairo_order_parameter()
            xyc = ca.compute()
            ca.plot_diagram(xyc)
        """
        num_x = 99
        list_num = np.linspace(0.01,0.99,num_x)
        num_scan = 0.5*num_x*(num_x+1)
        record = np.zeros((int(num_scan),3))
        count = 0
        for cn3 in list_num:
            for cn4 in list_num:
                p1 = cn3+cn4
                if p1<=1:
                    cn4_normal = cn4*2
                    r34 = cn3/cn4_normal
                    r43 = cn4_normal/cn3
                    p2 = np.minimum(r34,r43)
                    p = p1*p2
                    record[count] = [cn3,cn4,p]
                    count = count + 1
        return record
    
    def compute_cairo_order_parameter(self,cn3,cn4):
        R"""
        output:
            p: order_parameter_for_cairo
        introduction:
            p_cairo = (cn3 + cn4)*min(cn3/2cn4,2cn4/cn3)
        """
        p1 = cn3+cn4
        if p1<=1:
            cn4_normal = cn4*2
            r0 = cn3*cn4
            if r0>0:
                r34 = cn3/cn4_normal
                r43 = cn4_normal/cn3
                p2 = np.minimum(r34,r43)
                p = p1*p2
            else:
                p = 0
        else:
            print('error: cn3+cn4>1')
            p = -1
                    
        return p

    def plot_diagram(self,account='remote',fig=None,ax=None,save=False):
        if ax is None:
            fig,ax = plt.subplots()
        xyc = self.compute_theoretical_diagram()
        #ax.scatter(pos[:,0],pos[:,1],c='k')
        scat = ax.scatter(xyc[:,0],xyc[:,1],c=xyc[:,2])
        ax.set_aspect('equal','box')
        ax.set_xlabel('cn3(1)')
        ax.set_ylabel('cn4(1)')
        ax.set_title('order_parameter_for_cairo')
        fig.colorbar(scat,ax=ax)
        if save:
            prefix = "/home/"+account+"/Downloads/"
            png_filename=prefix+'test_order_parameter_for_cairo.png'
            plt.savefig(png_filename)
        else:
            return fig,ax

class energy_computer:
    R"""
    Introduction:
        input particle positions, substrate array, to calculate the free energy of the system.
    steps of operation:
        1. define_system_manual or define_system_from_gsd
        2. compute_energy(extended_positions)
        3. use self.energy_mechanical

    example:
        import proceed_file as pf
        import points_analysis_2D as pa
        import numpy as np
        record_energy = np.zeros((2001,3))
        filename_trap='/home/remote/hoomd-examples_0/testhoneycomb3-8-12-part1'
        pos_traps = np.loadtxt(filename_trap)
        gsd_data = pf.proceed_gsd_file(None,'remote',4302,9)
        gsd_data.get_trajectory_data()
        frames = range(2001)
        for frame_id in frames:#[0,1,10,100,1000,2000]:
            extended_positions = gsd_data.get_extended_positions(frame_id)
            points = gsd_data.txyz[frame_id,:,0:2]
            nps = len(points)
            #filename_array = "/home/remote/Downloads/index4302_9"
            en = pa.energy_computer()
            en.define_system_manual(points,300,0.25,15,pos_traps,0.81,700,1)
            en.compute_energy(extended_positions)
            record_energy[frame_id,0] = en.energy_pin
            record_energy[frame_id,1] = en.energy_interaction
            record_energy[frame_id,2] = en.energy_mechanical
            #print(frame_id)
            #print(en.energy_pin/nps)
            #print(en.energy_interaction/nps)
            #print(en.energy_mechanical/nps)#the mechanical energy is increasing?
        np.savetxt('/home/remote/Downloads/energy_per_particle_index4302_9.txt',record_energy)#energy_index4302_9.txt
        #np.savetxt('/home/remote/Downloads/energy_index4302_9.txt',record_energy)

    """
    def __init__(self):
        pass

    def define_system_manual(self,xy,a_yukawa,kappa,r_cut_interaction,traps,lcr,k,r_cut_trap):#box,
        #self.box = box
        self.xy = xy
        self.a_yukawa = a_yukawa
        self.kappa = kappa
        self.r_cut_interaction = r_cut_interaction
        self.traps = traps*lcr
        self.lcr = lcr
        self.k = k
        self.r_cut_trap = r_cut_trap

    def define_system_from_gsd(self,simu_index,seed,account):
        import proceed_file as pf
        prefix='/home/'+account+'/Downloads/'#'/home/tplab/Downloads/'
        log_prefix='/home/'+account+'/hoomd-examples_0/'#'/home/tplab/hoomd-examples_0/'
        #load time steps
        if seed is None:
            str_index=str(int(simu_index))
            gsd_data = pf.processor_gsd_file(simu_index=simu_index)
        else:
            str_index=str(int(simu_index))+'_'+str(seed)
            file_gsd = log_prefix+'trajectory_auto'+str_index+'.gsd'#+'_'+str(seed)
            gsd_data = pf.processor_gsd_file(filename_gsd_seed=file_gsd,account=account)
            
        file_log=log_prefix+'log-output_auto'+str_index+'.log'#+'_'+str(seed)
        log_data = np.genfromtxt(fname=file_log, skip_header=True)
        time_steps = log_data[:,0]
        iframes = 0
        nframes=gsd_data.num_of_frames
        af = gsd_data.trajectory.read_frame(iframes)
        pos_list = np.zeros([nframes,af.particles.N,3])#gsd_data.trajectory[0].particles.N,
        while iframes < nframes:
            af = gsd_data.trajectory.read_frame(iframes)
            pos_list[iframes] = af.particles.position
            iframes = iframes + 1
        af.configuration.box
        pass

    def compute_energy(self,extended_positions):
        R"""
        Intro:
            including two parts of energy: 
            1 of particle-particle interaction and 
            2 particle-trap interaction.
        question:
            how to define entropy and hence compute free energy?
            for orthogonal system, F = - ln(Z) / beta 
        """
        list_interaction_distance = self.compute_distance(extended_positions,self.r_cut_interaction,True) 
        list_trap_distance = self.compute_distance(self.traps,self.r_cut_trap)
        
        list_energy_interaction_step1 = self.a_yukawa*np.exp(-self.kappa*list_interaction_distance)
        list_energy_interaction_step2 = np.divide(list_energy_interaction_step1,2*list_interaction_distance)
        # the 2 at denominator are used to calculate particle-wise energy only once, no overlaps. 
        list_energy_pin = 0.5*self.k*( np.square(list_trap_distance) - self.r_cut_trap*self.r_cut_trap)

        self.energy_interaction = np.sum(list_energy_interaction_step2)
        self.energy_pin = np.sum(list_energy_pin)
        self.energy_mechanical = self.energy_interaction + self.energy_pin
    
    def compute_energy_particle_wise(self,extended_positions):#[ongoing]
        R"""
        Intro:
            including two parts of energy: 
            1 of particle-particle interaction and 
            2 particle-trap interaction.
        result:
            self.energy_inter_pin_mecha: [Nparticles,3],containing energy_interaction,energy_pin,energy_mechanical
        question:
            how to define entropy and hence compute free energy?
            for orthogonal system, F = - ln(Z) / beta 
        """
        list_interaction_distance = self.compute_distance(extended_positions,self.r_cut_interaction,True,True) 
        list_trap_distance = self.compute_distance(self.traps,self.r_cut_trap,False,True)
        
        n_particles = len(list_interaction_distance)#trap_d = len(list_trap_distance)
        self.energy_inter_pin_mecha = np.zeros((n_particles,3))
        for j in range(n_particles):
            list_energy_interaction_step1 = self.a_yukawa*np.exp(-self.kappa*list_interaction_distance[j])
            list_energy_interaction_step2 = np.divide(list_energy_interaction_step1,2*list_interaction_distance[j])
            # the 2 at denominator are used to calculate particle-wise energy only once, no overlaps. 
            list_energy_pin = 0.5*self.k*( np.square(list_trap_distance[j]) - self.r_cut_trap*self.r_cut_trap)

            self.energy_inter_pin_mecha[j,0] = np.sum(list_energy_interaction_step2)
            self.energy_inter_pin_mecha[j,1] = np.sum(list_energy_pin)
            self.energy_inter_pin_mecha[j,2] = self.energy_inter_pin_mecha[j,0] + self.energy_inter_pin_mecha[j,1]

    def compute_distance(self,reference_positions,r_cut,interact_mode=False,particle_wise_mode=False):
        R"""
        Introduction:
            compute the distance between positions and reference_positions.
            list_dis: a (1,n) array containing the distances within r_cut. 
        """
        reference_positions = reference_positions[:,0:2]#force 3d array to be 2d array
        self.list_interparticle_distance=distance.cdist(self.xy, reference_positions, 'euclidean')
        check_if_effective_interaction = self.list_interparticle_distance[:,:]<r_cut
        if interact_mode:
            d_particle = 0.99
            # the diameter of particles, 
            # check to prevent the problem of
            # f(x)/x -> f(0)/0
            check_if_effective_interaction2 = self.list_interparticle_distance > d_particle
            check_if_effective_interaction = np.logical_and(check_if_effective_interaction,check_if_effective_interaction2)

        if particle_wise_mode:
            list_dis_value = self.get_distance_particle_wise(check_if_effective_interaction)
        else:#array mode
            list_dis_value = self.get_distance_as_array(check_if_effective_interaction)
                
        return list_dis_value

    def get_distance_as_array(self,check_if_effective_interaction):
        for i in range(len(check_if_effective_interaction)):
            list_dis_value_one_row = self.list_interparticle_distance[i,check_if_effective_interaction[i]]#if it is true?
            #print(list_dis)
            if i==0:
                list_dis_value = list_dis_value_one_row
            else:
                list_dis_value = np.concatenate((list_dis_value,list_dis_value_one_row))
        return list_dis_value

    def get_distance_particle_wise(self,check_if_effective_interaction):
        
        for i in range(len(check_if_effective_interaction)):
            list_dis_value_one_row = self.list_interparticle_distance[i,check_if_effective_interaction[i]]#if it is true?
            #print(list_dis)
            if i==0:
                list_dis_value = [list_dis_value_one_row]
            else:
                list_dis_value.append(list_dis_value_one_row)
        return list_dis_value

    def plot_energy_vs_t(self,data):
        data = data[:,:]/256
        fig,ax = plt.subplots()
        times = np.linspace(0,2000,2001)
        ax.semilogx(times,data[:,0],label='energy_pin')
        #ax.semilogx(times,data[:,1],label='energy_inter')
        #ax.semilogx(times,data[:,2],label='energy_mecha')
        plt.legend()
        plt.title('energy_per_particle')
        plt.xlabel('time(steps)')
        plt.ylabel('energy($k_BT$)')
        png_filename = '/home/remote/Downloads/4302_9/energy_vs_t.png'
        plt.savefig(png_filename)
        plt.close()

    def check_case(self):
        R"""
        bugs:
        1 denominators are zero. fixed.
        2 not multiply lcr yet. fixed

        #case 1: sparse square, to test pin        
        box = np.array([18,18,1,0,0,0],'float64')
        points = np.array([[0,0],[3,0],[0,3],[3,3]],'float64')
        pos_traps = points
        gsd_data = pf.proceed_gsd_file(None,'remote',4302,9)
        extended_positions = gsd_data.get_extended_positions_from_points(box,points)
        en = pa.energy_computer()
        en.define_system_manual(points,300,0.25,15.0,pos_traps,1.0,200.0,1.0)
        en.compute_energy(extended_positions)
        print(en.energy_pin)
        print(en.energy_interaction)
        print(en.energy_mechanical)

        #case2: sparse triangle, to test far from trap and 3 typical bonds.        
        test = 100*np.exp(-0.75)
        print(test)
        box = np.array([24,24,1,0,0,0],dtype='float64')
        points = np.array([[0,0],[3,0],[1.5,1.5*np.sqrt(3)]],dtype='float64')
        pos_traps = np.array([[10,10],[13,0],[15,10*np.sqrt(3)]])
        gsd_data = pf.proceed_gsd_file(None,'remote',4302,9)
        extended_positions = gsd_data.get_extended_positions_from_points(box,points)
        en = pa.energy_computer()
        en.define_system_manual(points,300,0.25,15.0,pos_traps,1.0,200.0,1.0)
        """
        lcr=0.79
        index1=4302#5238
        seed=9
        r_trap=0.5
        simu_index = str(index1)+'_'+str(seed)
        file_txyz = '/home/remote/Downloads/'+simu_index+'/txyz.npy'#_stable
        file_reference_points = '/home/remote/hoomd-examples_0/testhoneycomb3-8-12'
        #/media/remote/32E2D4CCE2D49607/file_lxt/hoomd-examples_0/testhoneycomb3-8-12-part1
        # '/home/remote/hoomd-examples_0/testhoneycomb3-8-12'
        txyz = np.load(file_txyz)
        reference_points = np.loadtxt(file_reference_points)
        reference_points_lcr = np.multiply(lcr,reference_points) 

        #small test system
        xy=[[0,0],[3,0],[1.5,1.5*np.sqrt(3)]]
        pass