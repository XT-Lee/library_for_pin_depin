
import matplotlib.pyplot as plt
import numpy as np
import opertateOnMysql as osql


class workflow_mysql_to_data:
    R"""
    Introduction:
    table_name='depin_from_kagome_part_repeat'
    simu_index | HarmonicK | LinearCompressionRatio | CoordinationNum4Rate 
    | CoordinationNum3Rate | RandomSeed | Psi6Global
    
    FIGURE scatter  LinearCompressionRatio vs KBT, Psi6 as value 
    
    Attributes:

    Methods:

    Example:
    import getDataAndScatterOOP as sop
    workflow = sop.workflow_mysql_to_data()
    workflow.
    
    """

    def __init__(self,table_name,workflow_name='depin_from_kagome_part_random'):
        self.U_interaction=300*np.exp(-0.25)
        self.set_parameters_k()
        self.set_parameters_lcr()

        self.table_name = table_name#'depin_from_kagome_part_repeat'

        self.workflow_name = workflow_name
        self.prefix_plot='/home/tplab/Downloads/'

        self.set_plot_parameters()
    
    def set_parameters_k(self,k1=100.0,k_step=100.0,k_end=1000.0):
        #list of HarmonicK
        self.k1=k1
        self.k_step=k_step
        self.k_end=k_end
        self.num=(self.k_end-self.k1)/self.k_step+1
        self.num=round(self.num)#get the num of repeat times

    def set_parameters_lcr(self,lcr1=0.80,lcr_step=0.01,lcr_end=0.90):
        self.lcr1=lcr1
        self.lcr_step=lcr_step
        self.lcr_end=lcr_end
        self.lcr_num=(self.lcr_end-self.lcr1)/self.lcr_step+1
        self.lcr_num=round(self.lcr_num)

    def get_data_from_mysql(self):
        self.record=np.zeros((self.lcr_num*self.num,8))
        count=0
        #scatter cycle
        for i in np.linspace(1,self.num,self.num):
            for j in np.linspace(1,self.lcr_num,self.lcr_num):
                kset=self.k1+(i-1)*self.k_step
                cond1=' where HarmonicK >'+str(kset-0.5*self.k_step)+' and HarmonicK <'+str(kset+0.5*self.k_step)
                lcrset=self.lcr1+(j-1)*self.lcr_step
                cond2=' and LinearCompressionRatio > '+str(lcrset-self.lcr_step*0.5)+' and LinearCompressionRatio <'+str(lcrset+self.lcr_step*0.5)
                data=osql.getDataFromMysql(table_name=self.table_name,search_condition=cond1+cond2)
                data=np.array(data)
                m4=np.mean(data[:,3])
                std4=np.std(data[:,3])
                m3=np.mean(data[:,4])
                std3=np.std(data[:,4])
                m6=np.mean(data[:,6])
                std6=np.std(data[:,6])
                self.record[count,:]=[lcrset,kset,m4,std4,m3,std3,m6,std6]
                count+=1
                #print(data)
                #print(m)
                #print(std)
        #rename "record"
        self.data=self.record
    
    def get_data_from_txt(self,filename):
        self.data = np.loadtxt(filename)

    def save_as_txt(self,save_file_name = "cairo_diagram_1_accurate"):
        np.savetxt(save_file_name,self.data)

    def set_plot_parameters(self,xlabel='Linear Compression Ratio (1)',ylabel='U trap (kBT)[Kagome_part]'):
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(self):        
        #plot
        plt.figure()
        #plot LCR VS K, CN4 as value
        plt.scatter(self.data[:,0],self.data[:,1]*0.5,c=self.data[:,2])# LCR VS K, CN4 as value
        #plt.show()
        plt.title('LCR VS K, CN4 as value, Uparticle='+str(int(self.U_interaction)) )
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.colorbar()
        
        png_filename=self.prefix_plot+'LCR_VS_K_CN4_as_value_'+self.workflow_name
        plt.savefig(png_filename)
        plt.close()

        plt.figure()
        #plot LCR VS K, CN3 as value
        plt.scatter(self.data[:,0],self.data[:,1]*0.5,c=self.data[:,4])# LCR VS K, CN3 as value
        #plt.show()
        plt.title('LCR VS K, CN3 as value, Uparticle='+str(int(self.U_interaction)) )
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.colorbar()

        png_filename=self.prefix_plot+'LCR_VS_K_CN3_as_value_'+self.workflow_name
        plt.savefig(png_filename)
        plt.close()
        
        plt.figure()
        #plot LCR VS K, Psi6 as value
        plt.scatter(self.data[:,0],self.data[:,1]*0.5,c=self.data[:,6])# LCR VS K, Psi6 as value
        #plt.show()
        plt.title('LCR VS K, Psi6 as value, Uparticle='+str(int(self.U_interaction)) )
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.colorbar()

        png_filename=self.prefix_plot+'LCR_VS_K_Psi6_as_value_'+self.workflow_name
        plt.savefig(png_filename)
        plt.close()

        plt.figure()
        #plot LCR VS K, CN4std as value
        plt.scatter(self.data[:,0],self.data[:,1]*0.5,c=self.data[:,3])# LCR VS K, CN4std as value
        #plt.show()
        plt.title('LCR VS K, CN4std as value, Uparticle='+str(int(self.U_interaction)) )
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.colorbar()

        png_filename=self.prefix_plot+'LCR_VS_K_CN4std_as_value_'+self.workflow_name
        plt.savefig(png_filename)
        plt.close()

        plt.figure()
        #plot LCR VS K, CN3std as value
        plt.scatter(self.data[:,0],self.data[:,1]*0.5,c=self.data[:,5])# LCR VS K, CN3std as value
        #plt.show()
        plt.title('LCR VS K, CN3std as value, Uparticle='+str(int(self.U_interaction)) )
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.colorbar()

        png_filename=self.prefix_plot+'LCR_VS_K_CN3std_as_value_'+self.workflow_name
        plt.savefig(png_filename)
        plt.close()
        
        plt.figure()
        #plot LCR VS K, Psi6std as value
        plt.scatter(self.data[:,0],self.data[:,1]*0.5,c=self.data[:,7])# LCR VS K, Psi6std as value
        #plt.show()
        plt.title('LCR VS K, Psi6std as value, Uparticle='+str(int(self.U_interaction)) )
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.colorbar()

        png_filename=self.prefix_plot+'LCR_VS_K_Psi6std_as_value_'+self.workflow_name
        plt.savefig(png_filename)
        plt.close()
