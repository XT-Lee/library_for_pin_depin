import math
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import opertateOnMysql as osql
from points_analysis_2D_freud import static_points_analysis_2d
import freud
R"""
#NEW plot standard:
#https://matplotlib.org/3.6.2/tutorials/introductory/quick_start.html#sphx-glr-tutorials-introductory-quick-start-py
    import matplotlib.pyplot as plt
    import numpy as np

    points = np.random.random((100,2))#random.random () ()
    
    fig,ax = plt.subplots()
    ax.plot(points[:,0],points[:,1])
    ax.scatter(points[:,0],points[:,1],c=points[:,1])
    ax.set_xlabel('x label')  # Add an x-label to the axes.
    ax.set_ylabel('y label')  # Add a y-label to the axes.
    ax.set_title("Simple Plot")  # Add a title to the axes
    plt.show()

"""

def workflow_mysql_to_data_pin_hex_to_honeycomb():
    R"""
    simu_index | HarmonicK | LinearCompressionRatio | Psi3Global | Psi3Ratio
    """
    U_interaction=300*np.exp(-0.25)
    points=osql.getDataFromMysql(table_name="pin_hex_to_honeycomb",search_condition="where LinearCompressionRatio < 0.855")
    points=np.asarray(points)

    plt.figure()
    plt.scatter(points[:,2],0.5*points[:,1],c=points[:,3])
    plt.colorbar()
    plt.title('LCR VS K,Psi3 as value, kBT=1, Uparticle='+str(int(U_interaction))+',pin' )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb]')

    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_vs_K_Psi3_as_value_pin_hex_to_honeycomb'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    plt.scatter(points[:,2],0.5*points[:,1],c=points[:,4])
    plt.colorbar()
    plt.title('LCR VS K,Psi3Ratio as value, kBT=1, Uparticle='+str(int(U_interaction))+',pin' )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb]')
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_vs_K_Psi3Ratio_as_value_pin_hex_to_honeycomb'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_honeycomb_random():
    R"""
    Introduction:
    table_name='pin_hex_to_honeycomb_repeat' 
        #the simu_index of 'pin_hex_to_honeycomb' is complex
    simu_index | HarmonicK | LinearCompressionRatio | 
    Psi3Global | Psi6Global | RandomSeed
    
    FIGURE scatter  LinearCompressionRatio vs U_trap, Psi3 as value 
    
    Example:
    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_honeycomb_random()

    """
    #control table
    save_data_txt = True

    k1=0.0
    k_step=10.0
    k_end=90.0
    
    lcr1=0.76
    lcr_step=0.01
    lcr_end=0.80
    

    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    #list of HarmonicK
    num=(k_end-k1)/k_step+1
    num=round(num)#get the num of repeat times
    #list of LCR
    lcr_num=(lcr_end-lcr1)/lcr_step+1
    lcr_num=round(lcr_num)
    
    record=np.zeros((lcr_num*num,6))
    
    count=0
    #scatter cycle
    for i in np.linspace(1,num,num):
        for j in np.linspace(1,lcr_num,lcr_num):
            kset=k1+(i-1)*k_step
            cond1=' where HarmonicK >'+str(kset-0.5*k_step)+' and HarmonicK <'+str(kset+0.5*k_step)
            lcrset=lcr1+(j-1)*lcr_step
            cond2=' and LinearCompressionRatio > '+str(lcrset-lcr_step*0.5)+' and LinearCompressionRatio <'+str(lcrset+lcr_step*0.5)
            search_condition=cond1+cond2
            data=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_repeat',search_condition=search_condition)
            data=np.array(data)
            m3=np.mean(data[:,3])#psi3
            std3=np.std(data[:,3])
            m6=np.mean(data[:,4])#psi6
            std6=np.std(data[:,4])
            record[count,:]=[lcrset,kset,m3,std3,m6,std6]
            count+=1
            #print(data)
            #print(m)
            #print(std)
    #rename "record"
    data=record

    #save data
    if save_data_txt:
        prefix='/home/tplab/Downloads/'
        save_file_name=prefix+"honeycomb_diagram_2"
        np.savetxt(save_file_name,data)

    #plot
    plt.figure()
    #plot LCR VS K, Psi3 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,2])# LCR VS K, Psi3 as value
    #plt.show()
    plt.title('LCR VS K, Psi3 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi3_as_value_pin_hex_to_honeycomb_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,4])# LCR VS K, Psi6 as value
    #plt.show()
    plt.title('LCR VS K, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6_as_value_pin_hex_to_honeycomb_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, Psi3std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,3])# LCR VS K, Psi3std as value
    #plt.show()
    plt.title('LCR VS K, Psi3std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi3std_as_value_pin_hex_to_honeycomb_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,5])# LCR VS K, Psi6std as value
    #plt.show()
    plt.title('LCR VS K, Psi6std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6std_as_value_pin_hex_to_honeycomb_random'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_honeycomb_klt_2m(account='tplab'):
    R"""

    Note: the format of table_name='pin_hex_to_honeycomb_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    Psi3     | Psi6     | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_honeycomb_klt_2m()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    con='where HarmonicK < 61'
    data=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_klt_2m',search_condition=con)
    data=np.array(data)
    prefix='/home/'+account+'/Downloads/'
    postfix = '_pin_hex_to_honeycomb_klt_2m_weak.png'

    plt.figure()
    #plot lcr VS k, Psi3 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,4])# LCR VS K, Psi3 as value
    #plt.show()
    plt.title('lcr VS k, Psi3 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('LinearCompressionRatio(1)')
    plt.ylabel('U trap (kBT)[honeycomb]')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_Psi3_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_honeycomb_klt_2m_random(account='tplab'):
    R"""

    Note: the format of table_name='pin_hex_to_honeycomb_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    Psi3     | Psi6     | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_honeycomb_klt_2m()

    select distinct LinearCompressionRatio, HarmonicK
    from pin_hex_to_honeycomb_klt_2m 
    where SimuIndex<4686 
    order by SimuIndex, RandomSeed asc;
    """
    #import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    cont='distinct LinearCompressionRatio'
    con='where SimuIndex<4686 order by LinearCompressionRatio asc'
    lcrs=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_klt_2m',
                                search_condition=con,select_content=cont)
    #data=np.array(data)

    cont='distinct HarmonicK'
    con='where SimuIndex<4686 order by HarmonicK asc'
    ks=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_klt_2m',
                                search_condition=con,select_content=cont)

    record_lk3 = np.zeros((len(lcrs)*len(ks),3))
    cont='distinct LinearCompressionRatio,HarmonicK'
    con='where SimuIndex<4686 order by LinearCompressionRatio,HarmonicK asc'
    lks=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_klt_2m',
                                search_condition=con,select_content=cont)
    record_lk3[:,0:2]=np.array(lks)   

    count = 0
    cont='avg(Psi3)'
    lcr_step=0.01 
    for lcr in lcrs:
        for k in ks:
            lcr1 = np.array(lcr)
            lcr_min=lcr1 - 0.5*lcr_step
            lcr_max=lcr1 + 0.5*lcr_step
            con='where SimuIndex<4686 and HarmonicK='+str(int(k[0]))+\
                ' and LinearCompressionRatio >'+str(lcr_min[0])+' and LinearCompressionRatio <'+str(lcr_max[0])
            psi3_avg=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_klt_2m',
                                search_condition=con,select_content=cont)
            psi3 = np.array(psi3_avg)                    
            record_lk3[count,2] = psi3[0]
            count = count + 1
    
    #data = record_lk3
    mp = mysql_data_processor()
    title = 'lcr VS k, Psi3 as value'#, Uparticle='+str(int(U_interaction))
    xname = 'LinearCompressionRatio(1)'
    yname = 'U trap ($k_BT$)[honeycomb]'
    prefix='/home/'+account+'/Downloads/'
    postfix = '_pin_hex_to_honeycomb_klt_2m_random.png'
    record_lk3[:,1] = 0.5*record_lk3[:,1]# k -> Utrap(kT)
    mp.draw_diagram_scatter(record_lk3,title,xname,yname,prefix,postfix)
    """
    plt.figure()
    #plot lcr VS k, Psi3 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,4])# LCR VS K, Psi3 as value
    #plt.show()
    plt.title('lcr VS k, Psi3 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('LinearCompressionRatio(1)')
    plt.ylabel('U trap (kBT)[honeycomb]')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_Psi3_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()
    """
    
def workflow_mysql_to_data_depin_from_honeycomb():
    R"""
    table_name='depin_from_honeycomb'
    simu_index | HarmonicK | LinearCompressionRatio | Psi3Global | Psi6Global
    
    FIGURE scatter  HarmonicK vs LCR,  
    """
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    data=osql.getDataFromMysql(table_name='depin_from_honeycomb',search_condition=' where HarmonicK <100')
    data=np.array(data)
    U_interaction=300*np.exp(-0.25)
    #plot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap,LinearSegmentedColormap
    
    plt.figure()
    #test new color map 
    #https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html#sphx-glr-tutorials-colors-colormap-manipulation-py
    '''
    viridis = cm.get_cmap('viridis',256)
    newcolors = viridis(np.linspace(0,1,256))
    num=54
    orr = cm.get_cmap('Oranges_r',num)
    inject_color = orr(np.linspace(0,1,num))
    newcolors[0:num,:] = inject_color
    newcmp = ListedColormap(newcolors)
    '''
    num=54
    top = cm.get_cmap('Oranges_r',128)
    bottom = cm.get_cmap('Blues',128)
    newcolors =np.vstack((top(np.linspace(0,1,num)),bottom(np.linspace(0,1,256-num))))
    newcmp = ListedColormap(newcolors)

    value=data[:,3]/data[:,4]
    value=np.log10(value)
    plt.scatter(data[:,2],0.5*data[:,1],c=value,cmap=newcmp)# K VS LCR, Psi3/Psi6 as value
    plt.colorbar()
    plt.title('LCR VS K,log(Psi3/Psi6) as value, kBT=1, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb]')
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi3_over_Psi6_as_value_changecolor'
    plt.savefig(png_filename)
    plt.close()
    '''
    plt.scatter(0.5*data[:,1],data[:,2],c=value,cmap=newcmp)# K VS LCR, Psi3/Psi6 as value
    plt.colorbar()
    plt.title('K vs LCR,log(Psi3/Psi6) as value, kBT=1, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('U trap (kBT)')
    plt.ylabel('Linear Compression Ratio (1)')
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'K_vs_LCR_Psi3_over_Psi6_as_value_changecolor'
    plt.savefig(png_filename)
    plt.close()
    '''

def workflow_mysql_to_data_pin_hex_to_honeycomb_klt_2m_random_mark(account='tplab'):
    R"""
    Note: the format of table_name=
    'pin_hex_to_honeycomb_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    Psi3 | Psi6 | RandomSeed | 

    'depin_from_honeycomb_part1'
    | simu_index | HarmonicK | LinearCompressionRatio | 
    Psi3Global | Psi6Global | RandomSeed |
    
    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    "H" <_> hexagon2
    "^"     triangle_up
    "v"     triangle_down
    "d" v^  thin_diamond
    example:
    """
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    #U_interaction=300*np.exp(-0.25)
    table_name = 'pin_hex_to_honeycomb_klt_2m'

    cont='distinct LinearCompressionRatio'
    con=' order by LinearCompressionRatio asc'#where SimuIndex<4686
    lcrs=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)

    cont='distinct HarmonicK'
    con=' order by HarmonicK asc'
    ks=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)

    cont='distinct LinearCompressionRatio,HarmonicK'
    con=' order by LinearCompressionRatio,HarmonicK asc'
    lks=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)
    record_lk36 = np.zeros((len(lks),4))#(lcr,k,psi3,psi6)
    #record_lk36[:,0:2]=np.array(lks)   

    count = 0
    cont=' avg(Psi3),avg(Psi6) '
    lcr_step=0.001 
    for lcr in lcrs:
        for k in ks:
            lcr1 = np.array(lcr)
            lcr_min=lcr1 - 0.5*lcr_step
            lcr_max=lcr1 + 0.5*lcr_step
            con=' where HarmonicK='+str(int(k[0]))+\
                ' and LinearCompressionRatio >'+str(lcr_min[0])+' and LinearCompressionRatio <'+str(lcr_max[0])
            psi36_avg=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)
            psi36 = np.array(psi36_avg[0])
            #print(math.isnan(psi36[0]))#(np.nan(psi36_avg[0])
            if not psi36[0] is None:#math.isnan(psi36[0]):    
                #if not (k[0]<101 and k[0]>97):   
                #print(k)           
                #record_lk36[count,2:] = psi36[0:2]
                record_lk36[count,:] = [lcr[0],k[0],psi36[0],psi36[1]]
                count = count + 1
    record_lk36 = record_lk36[:count]
    #data = record_lk3
    
    record_lk36[:,1] = 0.5*record_lk36[:,1]# k -> Utrap(kT)
    return record_lk36
    
def workflow_mysql_to_data_depin_from_honeycomb_mark():
    R"""
    Note: the format of table_name=
    'depin_from_honeycomb'
    | simu_index | HarmonicK | LinearCompressionRatio | 
    Psi3Global | Psi6Global | RandomSeed |
    """
    data=osql.getDataFromMysql(table_name='depin_from_honeycomb')#,search_condition=' where HarmonicK <100'
    data=np.array(data)

    p36=data[:,3]/data[:,4]
    p36=np.log10(p36)

    sz = np.shape(data)
    record_lklg3 = np.zeros((sz[0],3))
    record_lklg3[:,0] = data[:,2]#lcr
    record_lklg3[:,1] = data[:,1]*0.5#k
    record_lklg3[:,2] = p36
    list_lg3 = record_lklg3[:,1] < 11#U(honeycomb pin stable)~10

    return record_lklg3[list_lg3]

def workflow_data_to_diagram_honeycomb_mark(account='tplab',save_mode="csv"):
    R"""
    Note: the format of table_name=
    'pin_hex_to_honeycomb_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    Psi3 | Psi6 | RandomSeed | 

    'depin_from_honeycomb'
    | simu_index | HarmonicK | LinearCompressionRatio | 
    Psi3Global | Psi6Global | RandomSeed |
    
    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    "H" <_> hexagon2
    "^"     triangle_up
    "v"     triangle_down
    "d" v^  thin_diamond
    example:
    """
    fig_name = 'diagram_HC_mark'
    record_lk36 = workflow_mysql_to_data_pin_hex_to_honeycomb_klt_2m_random_mark()
    record_lklg3 = workflow_mysql_to_data_depin_from_honeycomb_mark()

    mp = mysql_data_processor()
    title = 'lcr VS k, Psi3 as value'#, Uparticle='+str(int(U_interaction))
    xlabel_name = 'Linear Compression Ratio (1)'
    ylabel_name = 'Usub ($k_BT$)[Honeycomb]'#[HC]
    prefix='/home/'+account+'/Downloads/'
    
    if save_mode=="pdf":
        postfix = '_'+fig_name+'.pdf'
        fig,ax = mp.draw_diagram_scatter_mark_multi(record_lk36,title, xlabel_name,ylabel_name,prefix,postfix=None,oop_check=0.85)
        mp.draw_diagram_scatter_mark_multi(record_lklg3,title, xlabel_name,ylabel_name,prefix,postfix,0.2,fig,ax,"depin")
    
    elif save_mode=="csv":
        postfix = '_'+fig_name+'.csv'
        df0 = mp.get_diagram_data_mark_multi(record_lk36,title, xlabel_name,ylabel_name,prefix,csv_postfix=None,oop_check=0.85)
        mp.get_diagram_data_mark_multi(record_lklg3,title, xlabel_name,ylabel_name,prefix,postfix,0.2,"depin",df0)


def workflow_mysql_to_data_depin_from_honeycomb_part1(num_to_zero=64):
    R"""
    table_name='depin_from_honeycomb_part1'
    simu_index | HarmonicK | LinearCompressionRatio | Psi3Global | Psi6Global
    
    FIGURE scatter  HarmonicK vs LCR,  

    parameters:
        num_to_zero: set a num(int) to let white fixed on zero,
         then red and blue colorbar will distribute with symmetry.
    """
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    data=osql.getDataFromMysql(table_name='depin_from_honeycomb_part1')#,search_condition=' where HarmonicK <100'
    data=np.array(data)
    U_interaction=300*np.exp(-0.25)
    #plot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap,LinearSegmentedColormap
    
    plt.figure()
    #test new color map 
    #https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html#sphx-glr-tutorials-colors-colormap-manipulation-py
    '''
    viridis = cm.get_cmap('viridis',256)
    newcolors = viridis(np.linspace(0,1,256))
    num=54
    orr = cm.get_cmap('Oranges_r',num)
    inject_color = orr(np.linspace(0,1,num))
    newcolors[0:num,:] = inject_color
    newcmp = ListedColormap(newcolors)
    '''
    num=num_to_zero
    top = cm.get_cmap('Oranges_r',128)
    bottom = cm.get_cmap('Blues',128)
    newcolors =np.vstack((top(np.linspace(0,1,num)),bottom(np.linspace(0,1,256-num))))
    newcmp = ListedColormap(newcolors)

    value=data[:,3]/data[:,4]
    value=np.log10(value)
    plt.scatter(data[:,2],0.5*data[:,1],c=value,cmap=newcmp)# LCR vs K, Psi3/Psi6 as value
    plt.colorbar()
    plt.title('LCR VS K,log(Psi3/Psi6) as value, kBT=1, Uparticle='+str(int(U_interaction))+',depin' )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb part1]')
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_vs_K_Psi3_over_Psi6_as_value_changecolor_part1_depin'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_depin_from_honeycomb_part_36_kt(num_to_zero=64):
    R"""
    table_name='depin_from_honeycomb_part_klt'
    | simu_index | HarmonicK | LinearCompressionRatio | kT   | Psi3Global | Psi6Global | RandomSeed |
    
    FIGURE scatter  HarmonicK vs LCR,  

    parameters:
        num_to_zero: set a num(int) to let white fixed on zero,
         then red and blue colorbar will distribute with symmetry.
    """
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    data=osql.getDataFromMysql(table_name='depin_from_honeycomb_part_klt')#depin_from_honeycomb_part1,search_condition=' where HarmonicK <100'
    data=np.array(data)
    U_interaction=300*np.exp(-0.25)
    #plot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap,LinearSegmentedColormap
    
    plt.figure()
    #test new color map 
    #https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html#sphx-glr-tutorials-colors-colormap-manipulation-py
    '''
    viridis = cm.get_cmap('viridis',256)
    newcolors = viridis(np.linspace(0,1,256))
    num=54
    orr = cm.get_cmap('Oranges_r',num)
    inject_color = orr(np.linspace(0,1,num))
    newcolors[0:num,:] = inject_color
    newcmp = ListedColormap(newcolors)
    '''
    num=num_to_zero
    top = cm.get_cmap('Oranges_r',128)
    bottom = cm.get_cmap('Blues',128)
    newcolors =np.vstack((top(np.linspace(0,1,num)),bottom(np.linspace(0,1,256-num))))
    newcmp = ListedColormap(newcolors)

    value=data[:,4]/data[:,5]
    value=np.log10(value)
    plt.scatter(0.5*data[:,1],data[:,3],c=value,cmap=newcmp)# LCR vs K, Psi3/Psi6 as value
    plt.colorbar()
    plt.title('K VS T,log(Psi3/Psi6) as value, kBT=1, Uparticle='+str(int(U_interaction))+',depin' )
    plt.xlabel('U trap (kBT)[honeycomb part1]')
    plt.ylabel('kBT')
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'K_vs_T_Psi3_over_Psi6_as_value_changecolor_part1_depin'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_depin_from_honeycomb_part_kt():
    R"""
    table_name='depin_from_honeycomb_part_klt'
    | simu_index | HarmonicK | LinearCompressionRatio | kT   | Psi3Global | Psi6Global | RandomSeed |
    
    FIGURE scatter  HarmonicK vs LCR,  

    parameters:
        num_to_zero: set a num(int) to let white fixed on zero,
         then red and blue colorbar will distribute with symmetry.
    """
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    data=osql.getDataFromMysql(table_name='depin_from_honeycomb_part_klt')#depin_from_honeycomb_part1,search_condition=' where HarmonicK <100'
    data=np.array(data)
    U_interaction=300*np.exp(-0.25)
    #plot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap,LinearSegmentedColormap
    
    plt.figure()
    plt.scatter(0.5*data[:,1],data[:,3],c=data[:,4])# K VS T, Psi3 as value
    plt.colorbar()
    plt.title('K VS T,log(Psi3/Psi6) as value, kBT=1, Uparticle='+str(int(U_interaction))+',depin' )
    plt.xlabel('U trap (kBT)[honeycomb part1]')
    plt.ylabel('kBT')
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'K_vs_T_Psi3_as_value_honeycomb_part1_depin'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_honeycomb_part1(compare36=False,num_to_zero=64,p3=False):
    R"""
    table_name='pin_hex_to_honeycomb_part1'
    simu_index | HarmonicK | LinearCompressionRatio | Psi3Global | Psi6Global
    
    FIGURE scatter  HarmonicK vs LCR,  
    """
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    data=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_part1',search_condition="where HarmonicK > 99")#,search_condition=' where HarmonicK <100'
    data=np.array(data)
    U_interaction=300*np.exp(-0.25)
    #plot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap,LinearSegmentedColormap
    
    plt.figure()
    
    if compare36:
        #test new color map 
        #https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html#sphx-glr-tutorials-colors-colormap-manipulation-py
        num=num_to_zero
        top = cm.get_cmap('Oranges_r',128)
        bottom = cm.get_cmap('Blues',128)
        newcolors =np.vstack((top(np.linspace(0,1,num)),bottom(np.linspace(0,1,256-num))))
        newcmp = ListedColormap(newcolors)

        value=data[:,3]/data[:,4]
        value=np.log10(value)
        plt.scatter(data[:,2],0.5*data[:,1],c=value,cmap=newcmp)# LCR vs K, Psi3/Psi6 as value
        plt.colorbar()
        plt.title('LCR vs K,log(Psi3/Psi6) as value, kBT=1, Uparticle='+str(int(U_interaction))+',pin' )
        plt.xlabel('Linear Compression Ratio (1)')
        plt.ylabel('U trap (kBT)[honeycomb part1]')
        #plt.yscale('symlog') #log is not suitable
        prefix='/home/tplab/Downloads/'
        png_filename=prefix+'LCR_vs_K1k_Psi3_over_Psi6_as_value_changecolor_part1_pin'
        plt.savefig(png_filename)
        plt.close()
    elif p3:
        plt.scatter(data[:,2],0.5*data[:,1],c=data[:,3])# LCR vs K, Psi3 as value
        plt.colorbar()
        plt.title('LCR vs K,Psi3 as value, kBT=1, Uparticle='+str(int(U_interaction))+',pin' )
        plt.xlabel('Linear Compression Ratio (1)')
        plt.ylabel('U trap (kBT)[honeycomb part1]')
        #plt.yscale('symlog') #log is not suitable
        prefix='/home/tplab/Downloads/'
        png_filename=prefix+'LCR_vs_K1k_Psi3_as_value_changecolor_part1_pin'
        plt.savefig(png_filename)
        plt.close()
    else:
        plt.scatter(data[:,2],0.5*data[:,1],c=data[:,4])# LCR vs K, Psi6 as value
        plt.colorbar()
        plt.title('LCR vs K,Psi6 as value, kBT=1, Uparticle='+str(int(U_interaction))+',pin' )
        plt.xlabel('Linear Compression Ratio (1)')
        plt.ylabel('U trap (kBT)[honeycomb part1]')
        #plt.yscale('symlog') #log is not suitable
        prefix='/home/tplab/Downloads/'
        png_filename=prefix+'LCR_vs_K1k_Psi6_as_value_changecolor_part1_pin'
        plt.savefig(png_filename)
        plt.close()

def workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt(account='tplab'):
    R"""

    Note: the format of table_name='pin_hex_to_honeycomb_part_klt'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    data=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_part_klt')
    data=np.array(data)
    prefix='/home/'+account+'/Downloads/'
    postfix = '_pin_hex_to_honeycomb_part_klt.png'

    plt.figure()
    #plot k VS T, CN4 as value
    plt.scatter(data[:,1]*0.5,data[:,3],c=data[:,4])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('k VS T, CN3 as value, LCRc, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('U trap (kBT)[Honeycomb part]')
    plt.ylabel('kBT')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_CN3_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m(account='tplab'):
    R"""
    Note: the format of table_name='pin_hex_to_honeycomb_part_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    Psi3 | Psi6 | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    example:
        import getDataAndScatter as scatt
        scatt.workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m(account='remote')
    example:
        # lcr=0.77-0.78,seed=0-9
        $ select * from pin_hex_to_honeycomb_part_klt_2m where SimuIndex >4335 and SimuIndex <4356;
        #"where SimuIndex > 5298" 

        # lcr=0.79-0.8165-0.84,seed=0-9
        $ select * from pin_hex_to_honeycomb_part_klt_2m where SimuIndex >4265 and SimuIndex <4336;
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    condition="where SimuIndex > 4265 and SimuIndex<4336"
    data=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_part_klt_2m',search_condition=condition)
    data=np.array(data)
    prefix='/home/'+account+'/Downloads/'
    postfix = '_pin_hex_to_honeycomb_part_klt_2m_random.png'
    #print(data[:,4])
    plt.figure()
    #plot k VS T, Psi3 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,4])# LCR VS K, Psi3 as value
    #plt.show()
    plt.title('k VS T, Psi3 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap ($k_BT_m$)[Honeycomb part]')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_Psi3_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot k VS T, Psi6 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,5])# LCR VS K, Psi6 as value
    #plt.show()
    plt.title('k VS T, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBTm)[Honeycomb part]')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_Psi6_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m_random_oop(account='tplab'):
    R"""
    Note: the format of table_name='pin_hex_to_honeycomb_part_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    Psi3 | Psi6 | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    example:
        import getDataAndScatter as scatt
        scatt.workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m_random_oop(account='remote')

    example:
        # lcr=0.77-0.78,seed=0-9
        $ select * from pin_hex_to_honeycomb_part_klt_2m where SimuIndex >4335 and SimuIndex <4356;
        #"where SimuIndex > 5298" 

        # lcr=0.79-0.8165-0.84,seed=0-9
        $ select * from pin_hex_to_honeycomb_part_klt_2m where SimuIndex >4265 and SimuIndex <4336;
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    pr = mysql_data_processor()
    U_interaction=300*np.exp(-0.25)

    condition="where SimuIndex > 4265 and SimuIndex<4336"
    data=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_part_klt_2m',search_condition=condition)
    data=np.array(data)

    prefix='/home/'+account+'/Downloads/'
    postfix = '_pin_hex_to_honeycomb_part_klt_2m_random.png'

    list_k,k_num = pr.unique_param(data[:,1])
    list_lcr,lcr_num = pr.unique_param(data[:,2])

    record=np.zeros((lcr_num*k_num,6))
    row = 0
    for k in list_k:
         for lcr in list_lcr:
            list_row = (data[:,1]==k)&(data[:,2]==lcr)
            random_point = data[list_row,4:5+1]
            record[row, 0] = k*0.5
            record[row, 1] = lcr
            record[row, 2],record[row, 3] = pr.average_std(random_point[:,0])#psi3
            record[row, 4],record[row, 5] = pr.average_std(random_point[:,1])#psi6
            row = row+1

    xlabel_name = 'Linear Compression Ratio (1)'
    ylabel_name = 'U trap ($k_BT_m$)[Honeycomb part]'
    
    title_name = 'lcr_VS_K_Psi3_as_value'
    results = pr.record_to_results_for_scatter(record,1,0,2)
    pr.draw_diagram_scatter(results,title_name,xlabel_name,ylabel_name,prefix,postfix)

    title_name = 'lcr_VS_K_Psi3std_as_value'
    results = pr.record_to_results_for_scatter(record,1,0,3)
    pr.draw_diagram_scatter(results,title_name,xlabel_name,ylabel_name,prefix,postfix)

    title_name = 'lcr_VS_K_Psi6_as_value'
    results = pr.record_to_results_for_scatter(record,1,0,4)
    pr.draw_diagram_scatter(results,title_name,xlabel_name,ylabel_name,prefix,postfix)

    title_name = 'lcr_VS_K_Psi6std_as_value'
    results = pr.record_to_results_for_scatter(record,1,0,5)
    pr.draw_diagram_scatter(results,title_name,xlabel_name,ylabel_name,prefix,postfix)

def workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m_random_mark(account='tplab'):
    R"""
    Note: the format of table_name=
    'pin_hex_to_honeycomb_part_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    Psi3 | Psi6 | RandomSeed | 

    'depin_from_honeycomb_part1'
    | simu_index | HarmonicK | LinearCompressionRatio | 
    Psi3Global | Psi6Global | RandomSeed |
    
    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    "H" <_> hexagon2
    "^"     triangle_up
    "v"     triangle_down
    "d" v^  thin_diamond
    example:
    """
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    #U_interaction=300*np.exp(-0.25)
    table_name = 'pin_hex_to_honeycomb_part_klt_2m'

    cont='distinct LinearCompressionRatio'
    con=' order by LinearCompressionRatio asc'#where SimuIndex<4686
    lcrs=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)

    cont='distinct HarmonicK'
    con=' order by HarmonicK asc'
    ks=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)

    cont='distinct LinearCompressionRatio,HarmonicK'
    con=' order by LinearCompressionRatio,HarmonicK asc'
    lks=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)
    record_lk36 = np.zeros((len(lks),4))#(lcr,k,psi3,psi6)
    #record_lk36[:,0:2]=np.array(lks)   

    count = 0
    cont=' avg(Psi3),avg(Psi6) '
    lcr_step=0.001 
    for lcr in lcrs:
        for k in ks:
            lcr1 = np.array(lcr)
            lcr_min=lcr1 - 0.5*lcr_step
            lcr_max=lcr1 + 0.5*lcr_step
            con=' where HarmonicK='+str(int(k[0]))+\
                ' and LinearCompressionRatio >'+str(lcr_min[0])+' and LinearCompressionRatio <'+str(lcr_max[0])
            psi36_avg=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)
            psi36 = np.array(psi36_avg[0])
            #print(math.isnan(psi36[0]))#(np.nan(psi36_avg[0])
            if not psi36[0] is None:#math.isnan(psi36[0]):    
                if not (k[0]<101 and k[0]>97):   
                    #print(k)           
                    #record_lk36[count,2:] = psi36[0:2]
                    record_lk36[count,:] = [lcr[0],k[0],psi36[0],psi36[1]]
                    count = count + 1
    record_lk36 = record_lk36[:count]
    #data = record_lk3
    
    record_lk36[:,1] = 0.5*record_lk36[:,1]# k -> Utrap(kT)
    return record_lk36
    
def workflow_mysql_to_data_depin_from_honeycomb_part1_mark():
    R"""
    Note: the format of table_name=
    'depin_from_honeycomb_part1'
    | simu_index | HarmonicK | LinearCompressionRatio | 
    Psi3Global | Psi6Global | RandomSeed |
    """
    data=osql.getDataFromMysql(table_name='depin_from_honeycomb_part1')#,search_condition=' where HarmonicK <100'
    data=np.array(data)

    p36=data[:,3]/data[:,4]
    p36=np.log10(p36)

    sz = np.shape(data)
    record_lklg3 = np.zeros((sz[0],3))
    record_lklg3[:,0] = data[:,2]#lcr
    record_lklg3[:,1] = data[:,1]*0.5#k
    record_lklg3[:,2] = p36
    list_lg3 = record_lklg3[:,1] < 31#U(honeycomb_part pin stable)~30

    return record_lklg3[list_lg3]

def workflow_data_to_diagram_honeycomb_part_mark(account='tplab',save_mode="csv"):
    R"""
    Note: the format of table_name=
    'pin_hex_to_honeycomb_part_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    Psi3 | Psi6 | RandomSeed | 

    'depin_from_honeycomb_part1'
    | simu_index | HarmonicK | LinearCompressionRatio | 
    Psi3Global | Psi6Global | RandomSeed |
    
    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    "H" <_> hexagon2
    "^"     triangle_up
    "v"     triangle_down
    "d" v^  thin_diamond
    example:
    """
    fig_name = 'diagram_HCP_mark'
    record_lk36 = workflow_mysql_to_data_pin_hex_to_honeycomb_part_klt_2m_random_mark()
    record_lklg3 = workflow_mysql_to_data_depin_from_honeycomb_part1_mark()

    mp = mysql_data_processor()
    title = 'lcr VS k, Psi3 as value'#, Uparticle='+str(int(U_interaction))
    xlabel_name = 'Linear Compression Ratio (1)'
    ylabel_name = 'Usub ($k_BT$)[Honeycomb_part]'
    prefix='/home/'+account+'/Downloads/'

    if save_mode=="pdf":
        postfix = '_'+fig_name+'.pdf'
        fig,ax = mp.draw_diagram_scatter_mark_multi(record_lk36,title, xlabel_name,ylabel_name,prefix,postfix=None,oop_check=0.85)
        mp.draw_diagram_scatter_mark_multi(record_lklg3,title, xlabel_name,ylabel_name,prefix,postfix,0.3,fig,ax,"depin")
    
    elif save_mode=="csv":
        postfix = '_'+fig_name+'.csv'
        df0 = mp.get_diagram_data_mark_multi(record_lk36,title, xlabel_name,ylabel_name,prefix,csv_postfix=None,oop_check=0.85)
        mp.get_diagram_data_mark_multi(record_lklg3,title, xlabel_name,ylabel_name,prefix,postfix,0.3,"depin",df0)


class mysql_data_processor:
    def __init__(self):
        pass
    def unique_param(self,data):
        R"""
        return list_data, list_num
        """
        list_data = np.unique(data)
        list_num = np.shape(list_data)[0]
        return list_data,list_num

    def average_std(self,data):
        R"""
        return avg, std
        """
        avg = np.mean(data)
        std = np.std(data)
        return avg,std
    
    def record_to_results_for_scatter(self,data,column_x,column_y,column_value):
        R"""
        return:
            results: array[x,y,values]
        """
        results = np.zeros((np.shape(data)[0],3))
        results[:,0] = data[:,column_x]
        results[:,1] = data[:,column_y]
        results[:,2] = data[:,column_value]
        return results

    def draw_diagram_scatter(self,data,title_name,xlabel_name,ylabel_name,prefix,postfix):
        plt.figure()
        plt.scatter(data[:,0],data[:,1],c=data[:,2])
        #plt.show()
        plt.title(title_name)
        plt.xlabel(xlabel_name)
        plt.ylabel(ylabel_name)
        plt.colorbar()
        png_filename=prefix+title_name+postfix
        plt.savefig(png_filename)
        plt.close()
    
    def draw_diagram_scatter_oop(self,data,title_name,xlabel_name,ylabel_name,prefix,postfix):
        fig,ax = plt.subplots()
        ax.scatter(data[:,0],data[:,1],c=data[:,2])
        #plt.show()
        ax.set_title(title_name)
        ax.set_xlabel(xlabel_name)
        ax.set_ylabel(ylabel_name)
        #ax.set_yscale('log')
        #ax.set_xlim(xlim)
        ax.set_ylim([0,300])
        from matplotlib import cm
        from matplotlib.colors import Normalize as nm
        vmin = np.min(data[:,2])
        vmax = np.max(data[:,2])
        #print(vmin,vmax)
        nmm = nm(vmin=vmin,vmax=vmax)
        #https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar
        fig.colorbar(cm.ScalarMappable(norm=nmm),ax=ax)#.autoscale(data[:,2])
        png_filename=prefix+title_name+postfix
        fig.savefig(png_filename)
        plt.close()
    
    def draw_diagram_scatter_mark(self,data,title_name,xlabel_name,ylabel_name,prefix,postfix,oop_check = 0.88):
        R"""
        scatter:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html#matplotlib.axes.Axes.scatter
        marker:
        https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
        "H" <_> hexagon2
        "^"     triangle_up
        "v"     triangle_down
        "d" v^  thin_diamond
        colors:
        https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
        """
        from matplotlib import colors
        color_transparent = colors.to_rgba('w',alpha = 0.0)
        #color_transparent = np.array(color_transparent)
        #oop_check = 0.88
        list_HC = data[:,2]>oop_check
        list_DF = data[:,2]<oop_check
        fig,ax = plt.subplots()
        ax.scatter(data[list_HC,0],data[list_HC,1],c=None,facecolors=color_transparent,edgecolors='k',marker='H')#'w'
        ax.scatter(data[list_DF,0],data[list_DF,1],c=None,facecolors=color_transparent,edgecolors='k',marker="^")
        #plt.show()
        ax.set_title(title_name)
        ax.set_xlabel(xlabel_name)
        ax.set_ylabel(ylabel_name)
        #ax.set_yscale('log')
        #ax.set_xlim(xlim)
        ax.set_ylim([0,120])
        from matplotlib import cm
        from matplotlib.colors import Normalize as nm
        vmin = np.min(data[:,2])
        vmax = np.max(data[:,2])
        #print(vmin,vmax)
        nmm = nm(vmin=vmin,vmax=vmax)
        #https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar
        fig.colorbar(cm.ScalarMappable(norm=nmm),ax=ax)#.autoscale(data[:,2])
        png_filename=prefix+title_name+postfix
        fig.savefig(png_filename)
        plt.close()
    
    def draw_diagram_scatter_mark_multi(self,data,title_name,xlabel_name,ylabel_name,prefix,
                                        postfix=None,oop_check = 0.88,fig=None,ax=None,mode="pin"):
        R"""
        scatter:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html#matplotlib.axes.Axes.scatter
        marker:
        https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
        "H" <_> hexagon2
        "^"     triangle_up
        "v"     triangle_down
        "d" v^  thin_diamond
        colors:
        https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
        """
        from matplotlib import colors
        color_transparent = colors.to_rgba('w',alpha = 0.0)
        #color_transparent = np.array(color_transparent)
        #oop_check = 0.88
        list_HC = data[:,2]>oop_check# -> honeycomb(HC) well
        list_DF = data[:,2]<oop_check# -> honeycomb(HC) with defects
        if ax is None:
            fig,ax = plt.subplots(figsize=[10.,12.])#5:6 suit
            #fig.set_figsize()
        
        if mode=="pin":
            ax.scatter(data[list_HC,0],data[list_HC,1],c=None,facecolors=color_transparent,edgecolors='k',marker='H')#'w'
            ax.scatter(data[list_DF,0],data[list_DF,1],c=None,facecolors=color_transparent,edgecolors='k',marker="^")
            #plt.show()
            ax.set_title(title_name)
            ax.set_xlabel(xlabel_name)
            ax.set_ylabel(ylabel_name)
            #ax.set_yscale('log')
            #ax.set_xlim(xlim)
            ax.set_ylim([1,51])#[1,110]
        elif mode=="depin":
            ax.scatter(data[list_HC,0],data[list_HC,1],c=None,facecolors=color_transparent,edgecolors='k',marker='^')#'w'
            ax.scatter(data[list_DF,0],data[list_DF,1],c=None,facecolors='k',edgecolors='k',marker="H")
            #plt.show()        
        if postfix is None:
            return fig,ax
        else:
            png_filename=prefix+title_name+postfix
            fig.savefig(png_filename)
    
    def get_diagram_data_mark_multi(self,data,title_name,xlabel_name,ylabel_name,prefix,
                                        csv_postfix=None,oop_check = 0.88,mode="pin",df0=None):
        R"""
        scatter:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html#matplotlib.axes.Axes.scatter
        marker:
        https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
        "H" <_> hexagon2
        "^"     triangle_up
        "v"     triangle_down
        "d" v^  thin_diamond
        colors:
        https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
        """
        import pandas as pd
        #1 classify and mark data
        list_HC = data[:,2]>oop_check
        list_DF = data[:,2]<oop_check
        
        if mode=="pin":
            #list_HC -> honeycomb_part(HCP) well        -> 2
            #list_DF -> honeycomb_part(HCP) with defects-> 1
            data[list_HC,2] = 2
            data[list_DF,2] = 1
        elif mode=="depin":
            #list_HC -> honeycomb_part(HCP) stable   -> 1
            #list_DF -> honeycomb_part(HCP) unstable -> 0 hexagonal
            data[list_HC,2] = 1
            data[list_DF,2] = 0

        #2 reorganize data as pandas.Dataframe
        df = pd.DataFrame(data[:,:3])
        #"Psi3Class":{"hex":0,"DF":1,"HC":2}
        columns_name = [xlabel_name,ylabel_name,"Psi3Class(0Hex1DF2HC)"]
        #columns_name = ["LinearCompressionRatio(1)","Utrap(kBT)","Psi3Class(0Hex1DF2HC)"]
        df.columns = columns_name
        if df0 is None:
            df0 = df
        else:
            df0 = pd.concat([df0,df])
        
        #3 save data as .csv file
        if csv_postfix is None:
            return df0
        else:
            csv_filename=prefix+title_name+csv_postfix
            pd.DataFrame.to_csv(df0,csv_filename)
        

def workflow_mysql_to_data_pin_hex_to_honeycomb_part1_random():
    R"""
    Introduction:
    table_name='pin_hex_to_honeycomb_part1'
    simu_index | HarmonicK | LinearCompressionRatio | 
    Psi3Global | Psi6Global | RandomSeed |
    
    FIGURE scatter  LinearCompressionRatio vs KBT, Psi6 as value 
    
    Example1:
    import getDataAndScatter as scatt
    #scan lcr and kT 0.71-0.90
    scatt.workflow_mysql_to_data_pin_hex_to_honeycomb_part1_random()
    
    Example2:
    import getDataAndScatter as scatt
    #scan lcr and kT 0.78-0.82
    scatt.workflow_mysql_to_data_pin_hex_to_honeycomb_part1_random()
 
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import opertateOnMysql as osql#getDataToMysql
    U_interaction=300*np.exp(-0.25)

    #control table
    save_data_txt = True

    #list of HarmonicK
    k1=100.0
    k_step=100.0
    k_end=1000.0
    k_num=(k_end-k1)/k_step+1
    k_num=round(k_num)#get the num of repeat times
    #list of LCR
    lcr1=0.81650#0.77
    lcr_step=0.01
    lcr_end=0.81650#0.84
    lcr_num=(lcr_end-lcr1)/lcr_step+1
    lcr_num=round(lcr_num)
    
    record=np.zeros((lcr_num*k_num,6))
    
    
    count=0
    #scatter cycle
    for i in np.linspace(1,k_num,k_num):
        for j in np.linspace(1,lcr_num,lcr_num):
            kset=k1+(i-1)*k_step
            cond1=' where HarmonicK >'+str(kset-0.5*k_step)+' and HarmonicK <'+str(kset+0.5*k_step)
            lcrset=lcr1+(j-1)*lcr_step
            cond2=' and LinearCompressionRatio > '+str(lcrset-lcr_step*0.5)+' and LinearCompressionRatio <'+str(lcrset+lcr_step*0.5)
            data=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_part1',search_condition=cond1+cond2)
            data=np.array(data)
            m3=np.mean(data[:,3])#psi3
            std3=np.std(data[:,3])
            m6=np.mean(data[:,4])#psi6
            std6=np.std(data[:,4])
            record[count,:]=[lcrset,kset,m3,std3,m6,std6]
            count+=1
            #print(data)
            #print(m)
            #print(std)
    #rename "record"
    data=record

    #save data
    if save_data_txt:
        prefix='/home/tplab/Downloads/'
        save_file_name=prefix+"honeycomb_part1_diagram_c"
        np.savetxt(save_file_name,data)

    #plot
    plt.figure()
    #plot LCR VS K, Psi3 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,2])# LCR VS K, Psi3 as value
    #plt.show()
    plt.title('LCR VS K, Psi3 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_part1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi3_as_value_pin_hex_to_honeycomb_part1_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,4])# LCR VS K, Psi6 as value
    #plt.show()
    plt.title('LCR VS K, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_part1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6_as_value_pin_hex_to_honeycomb_part1_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, Psi3std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,3])# LCR VS K, Psi3std as value
    #plt.show()
    plt.title('LCR VS K, Psi3std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_part1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi3std_as_value_pin_hex_to_honeycomb_part1_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,5])# LCR VS K, Psi6std as value
    #plt.show()
    plt.title('LCR VS K, Psi6std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_part1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6std_as_value_pin_hex_to_honeycomb_part1_random'
    plt.savefig(png_filename)
    plt.close()    

def workflow_load_to_data_pin_hex_to_honeycomb_part1_random():
    R"""
    Introduction:
    table_name='pin_hex_to_honeycomb_part1'
    simu_index | HarmonicK | LinearCompressionRatio | 
    Psi3Global | Psi6Global | RandomSeed |
    
    FIGURE scatter  LinearCompressionRatio vs KBT, Psi6 as value 
    
    Example:
    import getDataAndScatter as scatt
    scatt.workflow_load_to_data_pin_hex_to_honeycomb_part1_random()
 
    """
    import matplotlib.pyplot as plt
    import numpy as np
    U_interaction=300*np.exp(-0.25)

   
    #save data
    prefix='/home/tplab/Downloads/honeycomb/pin_hex_to_honeycomb_part1_random/'
    #'/home/tplab/Downloads/pin_hex_to_honeycomb_part1_random/'
    save_file_name=prefix+"honeycomb_part1_diagram_all"
    data = np.loadtxt(save_file_name)

    #plot
    plt.figure()
    #plot LCR VS K, Psi3 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,2])# LCR VS K, Psi3 as value
    #plt.show()
    plt.title('LCR VS K, Psi3 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_part1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi3_as_value_pin_hex_to_honeycomb_part1_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,4])# LCR VS K, Psi6 as value
    #plt.show()
    plt.title('LCR VS K, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_part1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6_as_value_pin_hex_to_honeycomb_part1_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, Psi3std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,3])# LCR VS K, Psi3std as value
    #plt.show()
    plt.title('LCR VS K, Psi3std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_part1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi3std_as_value_pin_hex_to_honeycomb_part1_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,5])# LCR VS K, Psi6std as value
    #plt.show()
    plt.title('LCR VS K, Psi6std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_part1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6std_as_value_pin_hex_to_honeycomb_part1_random'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_hex_from_honeycomb():
    R"""
    table_name='hex_from_honeycomb'
    SimuIndex | KBT  | LinearCompressionRatio | Pressure | Psi6Global
    
    FIGURE scatter  LinearCompressionRatio vs Pressure,  
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    k1=0.1
    step=0.1
    k_end=2.0
    num=(k_end-k1)/step+1
    num=round(num)#get the num of repeat times

    plt.figure()

    #scatter cycle
    for i in np.linspace(1,num,num):
        kset=0.1+(i-1)*step
        cond=' where KBT >'+str(kset-0.05)+' and KBT <'+str(kset+0.05)
        data=osql.getDataFromMysql(table_name='hex_from_honeycomb',search_condition=cond)
        data=np.array(data)
        #plot
        plt.scatter(data[:,2],data[:,3])# K VS LCR, Psi3/Psi6 as value
        
    #plt.colorbar()
    plt.title('V vs P, kBT as legend [0.1,2], Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('Pressure/kBT (1)')
    #plt.legend(np.linspace(0.1,2.0,20))
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'V_vs_P_kBT_as_legend'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_melt_hex_from_honeycomb_random():
    R"""
    Introduction:

    table_name='melt_hex_from_honeycomb'
    SimuIndex | KBT  | LinearCompressionRatio | Pressure | Psi6Global| RandomSeed
    
    FIGURE scatter  LinearCompressionRatio vs Pressure,  

    Example:
    import getDataAndScatter as scatt
    #scan seed
    seed=0
    while seed<9:
        index1=307
        lcr1=0.76
        #scan lcr and kT 0.76-0.99
        while lcr1<0.995:
            tt.workflow_simu_to_mysql_kl(index1,lcr1,seed)
            index1+=20
            lcr1+=0.01
    seed+=1
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    #list of KBT
    k1=0.1
    step=0.1
    k_end=2.0
    num=(k_end-k1)/step+1
    num=round(num)#get the num of repeat times
    #list of LCR
    lcr1=0.76
    lcr_step=0.01
    lcr_end=0.99
    lcr_num=(lcr_end-lcr1)/lcr_step+1
    lcr_num=round(lcr_num)
    
    record=np.zeros((lcr_num*num,4))
    
    
    count=0
    #scatter cycle
    for i in np.linspace(1,num,num):
        for j in np.linspace(1,lcr_num,lcr_num):
            kset=k1+(i-1)*step
            cond1=' where KBT >'+str(kset-0.05)+' and KBT <'+str(kset+0.05)
            lcrset=lcr1+(j-1)*lcr_step
            cond2=' and LinearCompressionRatio > '+str(lcrset-lcr_step*0.5)+' and LinearCompressionRatio <'+str(lcrset+lcr_step*0.5)
            data=osql.getDataFromMysql(table_name='melt_hex_from_honeycomb',search_condition=cond1+cond2)
            data=np.array(data)
            m=np.mean(data[:,4])
            std=np.std(data[:,4])
            record[count,:]=[lcrset,kset,m,std]
            count+=1
            #print(data)
            #print(m)
            #print(std)
    #plot
    prefix='/home/tplab/Downloads/'
    filename=prefix+"yukawa_phase_diagram_averaged"
    #np.savetxt(filename,record)
    
    plt.figure()
    #plt.scatter(record[:,1],record[:,2],c=record[:,3])# KBT VS Psi6 std as value
    plt.scatter(record[:,0],record[:,1],c=record[:,2])# LCR VS KBT, Psi6 as value
    plt.title('LCR vs kBT, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('kBT (1)')
    plt.colorbar()
    #plt.legend(np.linspace(0.1,2.0,20))
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'lcr_vs_kBT_Psi6_as_value_Random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    plt.scatter(record[:,1],record[:,2],c=record[:,3])# KBT VS Psi6, std as value
    plt.title('kBT vs Psi6, std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('kBT (1)')
    plt.ylabel('Psi6(1)')
    plt.colorbar()
    #plt.legend(np.linspace(0.1,2.0,20))
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'kBT_vs_Psi6_std_as_value_Random'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_melt_hex_from_honeycomb_check():
    R"""
    Introduction:

    table_name='melt_hex_from_honeycomb_check'
    SimuIndex | KBT  | LinearCompressionRatio | Pressure | Psi6Global| RandomSeed

    Example:
    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_melt_hex_from_honeycomb_check()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    record=osql.getDataFromMysql(table_name='melt_hex_from_honeycomb_check')
    record=np.array(record)
    #plot
    prefix='/home/tplab/Downloads/'
    filename=prefix+"yukawa_phase_diagram_check"
    #np.savetxt(filename,record)
    
    plt.figure()
    #plt.scatter(record[:,1],record[:,2],c=record[:,3])# KBT VS Psi6 std as value
    plt.scatter(record[:,2],record[:,1],c=record[:,4])# LCR VS KBT, Psi6 as value
    plt.title('LCR vs kBT, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('kBT (1)')
    plt.colorbar()
    #plt.legend(np.linspace(0.1,2.0,20))
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'lcr_vs_kBT_Psi6_as_value_check'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    plt.scatter(record[:,1],record[:,4])# KBT VS Psi6, std as value
    plt.title('kBT vs Psi6, std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('kBT (1)')
    plt.ylabel('Psi6(1)')
    #plt.colorbar()
    #plt.legend(np.linspace(0.1,2.0,20))
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'kBT_vs_Psi6_check'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_hex_from_honeycomb_log():
    R"""
    table_name='hex_from_honeycomb'
    SimuIndex | KBT  | LinearCompressionRatio | Pressure | Psi6Global
    
    FIGURE scatter  LinearCompressionRatio vs Pressure,  
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    k1=0.1
    step=0.1
    k_end=2.0
    num=(k_end-k1)/step+1
    num=round(num)#get the num of repeat times

    plt.figure()

    #scatter cycle
    for i in np.linspace(1,num,num):
        kset=0.1+(i-1)*step
        cond=' where KBT >'+str(kset-0.05)+' and KBT <'+str(kset+0.05)
        data=osql.getDataFromMysql(table_name='hex_from_honeycomb',search_condition=cond)
        data=np.array(data)
        #plot
        plt.scatter(data[:,2],np.log10(data[:,3]))# K VS LCR, Psi3/Psi6 as value
        
    #plt.colorbar()
    plt.title('V vs log(P), kBT as legend [0.1,2], Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('Pressure/kBT (1)')
    #plt.legend(np.linspace(0.1,2.0,20))
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'V_vs_logP_kBT_as_legend'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_hex_from_honeycomb_Psi6():
    R"""
    table_name='melt_hex_from_honeycomb'
    SimuIndex | KBT  | LinearCompressionRatio | Pressure | Psi6Global
    
    FIGURE scatter  LinearCompressionRatio vs KBT, Psi6 as value 
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)
    num_list=np.linspace(0.76,0.99,24)
    #print(num_list)
    plt.figure()
    for i in num_list:
        LCRmin=i-0.005
        LCRmax=i+0.005
        condition=' where LinearCompressionRatio > '+str(LCRmin)+' && LinearCompressionRatio < '+str(LCRmax)
        data=osql.getDataFromMysql(table_name='melt_hex_from_honeycomb',search_condition=condition)
        data=np.array(data)
        #plot T vs Psi6, LCR as legend
        plt.scatter(data[:,1],data[:,4])# K VS LCR, Psi3/Psi6 as value
        #plt.show()
        
    plt.title('T vs Psi6, LCR as legend, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('kBT')
    plt.ylabel('Psi6')    
    #plt.legend(np.linspace(0.76,0.99,24))
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'T_vs_Psi6_LCR_as_legend'
    plt.savefig(png_filename)
    plt.close()
    '''
    #plot V vs T
    plt.scatter(data[:,2],data[:,1],c=data[:,4])# K VS LCR, Psi3/Psi6 as value
        
    plt.colorbar()
    plt.title('V vs T, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('kBT')
    #plt.legend(np.linspace(0.1,2.0,20))
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_vs_T_Psi6_as_value'
    plt.savefig(png_filename)
    plt.close()
    '''

def workflow_mysql_to_data_pin_hex_to_honeycomb_rectangle1():
    R"""
    Introduction:
    table_name='pin_hex_to_honeycomb_rectangle1'
    simu_index | HarmonicK | LinearCompressionRatio | 
    Psi3Global | Psi6Global | RandomSeed
    
    FIGURE scatter  LinearCompressionRatio vs KBT, Psi6 as value 
    
    Example:
    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_honeycomb_rectangle1()

    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    #scatter cycle
    data=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_rectangle1')
    data=np.array(data)

    #plot
    plt.figure()
    #plot LCR VS K, Psi3 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,3])# LCR VS K, Psi3 as value
    #plt.show()
    plt.title('LCR VS K, Psi3 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_rectangle1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi3_as_value_pin_hex_to_honeycomb_rectangle1'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,4])# LCR VS K, Psi6 as value
    #plt.show()
    plt.title('LCR VS K, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_rectangle1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6_as_value_pin_hex_to_honeycomb_rectangle1'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_honeycomb_rectangle1_random():
    R"""
    Introduction:
    table_name='pin_hex_to_honeycomb_rectangle1'
    simu_index | HarmonicK | LinearCompressionRatio | 
    Psi3Global | Psi6Global | RandomSeed
    
    FIGURE scatter  LinearCompressionRatio vs U_trap, Psi3 as value 
    
    Example:
    import getDataAndScatter as scatt
    tt.workflow_mysql_to_data_pin_hex_to_honeycomb_rectangle1_random()

    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    #list of HarmonicK
    k1=100.0
    k_step=100.0
    k_end=1000.0
    num=(k_end-k1)/k_step+1
    num=round(num)#get the num of repeat times
    #list of LCR
    lcr1=0.71
    lcr_step=0.01
    lcr_end=0.90
    lcr_num=(lcr_end-lcr1)/lcr_step+1
    lcr_num=round(lcr_num)
    
    record=np.zeros((lcr_num*num,6))
    
    
    count=0
    #scatter cycle
    for i in np.linspace(1,num,num):
        for j in np.linspace(1,lcr_num,lcr_num):
            kset=k1+(i-1)*k_step
            cond1=' where HarmonicK >'+str(kset-0.5*k_step)+' and HarmonicK <'+str(kset+0.5*k_step)
            lcrset=lcr1+(j-1)*lcr_step
            cond2=' and LinearCompressionRatio > '+str(lcrset-lcr_step*0.5)+' and LinearCompressionRatio <'+str(lcrset+lcr_step*0.5)
            data=osql.getDataFromMysql(table_name='pin_hex_to_honeycomb_rectangle1',search_condition=cond1+cond2)
            data=np.array(data)
            m3=np.mean(data[:,3])#psi3
            std3=np.std(data[:,3])
            m6=np.mean(data[:,4])#psi6
            std6=np.std(data[:,4])
            record[count,:]=[lcrset,kset,m3,std3,m6,std6]
            count+=1
            #print(data)
            #print(m)
            #print(std)
    #rename "record"
    data=record
    #plot
    plt.figure()
    #plot LCR VS K, Psi3 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,2])# LCR VS K, Psi3 as value
    #plt.show()
    plt.title('LCR VS K, Psi3 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_rectangle1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi3_as_value_pin_hex_to_honeycomb_rectangle1_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,4])# LCR VS K, Psi6 as value
    #plt.show()
    plt.title('LCR VS K, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_rectangle1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6_as_value_pin_hex_to_honeycomb_rectangle1_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, Psi3std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,3])# LCR VS K, Psi3std as value
    #plt.show()
    plt.title('LCR VS K, Psi3std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_rectangle1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi3std_as_value_pin_hex_to_honeycomb_rectangle1_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,5])# LCR VS K, Psi6std as value
    #plt.show()
    plt.title('LCR VS K, Psi6std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[honeycomb_rectangle1]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6std_as_value_pin_hex_to_honeycomb_rectangle1_random'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_depin_from_kagome():
    R"""
    table_name='depin_from_kagome'
    | simu_index | HarmonicK | LinearCompressionRatio 
    | CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed 
    | Psi6Global |
    
    FIGURE scatter  LinearCompressionRatio vs KBT, Psi6 as value 
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    data=osql.getDataFromMysql(table_name='depin_from_kagome')
    data=np.array(data)

    plt.figure()
    #plot LCR VS K, CN4 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,3])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('LCR VS K, CN4 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN4_as_value_depin_from_kagome'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, CN6 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,4])# LCR VS K, CN6 as value
    #plt.show()
    plt.title('LCR VS K, CN6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN6_as_value_depin_from_kagome'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,6])# LCR VS K, Psi6 as value
    #plt.show()
    plt.title('LCR VS K, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6_as_value_depin_from_kagome'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_kagome():
    R"""
    table_name='pin_hex_to_kagome'
    simu_index | HarmonicK | LinearCompressionRatio | CoordinationNum4Rate 
    | CoordinationNum6Rate | RandomSeed | Psi6Global
    
    FIGURE scatter  LinearCompressionRatio vs KBT, Psi6 as value 
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    data=osql.getDataFromMysql(table_name='pin_hex_to_kagome')
    data=np.array(data)
    
    plt.figure()
    #plot LCR VS K, CN4 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,3])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('LCR VS K, CN4 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN4_as_value_pin_hex_to_kagome'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, CN6 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,4])# LCR VS K, CN6 as value
    #plt.show()
    plt.title('LCR VS K, CN6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN6_as_value_pin_hex_to_kagome'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,6])# LCR VS K, Psi6 as value
    #plt.show()
    plt.title('LCR VS K, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6_as_value_pin_hex_to_kagome'
    plt.savefig(png_filename)
    plt.close()
    """
    plt.figure()
    #plot LCR VS K, logCN6/CN4 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,6])# LCR VS K, CN6/CN4 as value
    #plt.show()
    plt.title('LCR VS K, logCN6/CN4 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_logCN6-CN4_as_value'
    plt.savefig(png_filename)
    plt.close()
    """

def workflow_mysql_to_data_pin_hex_to_kagome_random():
    R"""
    Introduction:
    table_name='pin_hex_to_kagome'
    simu_index | HarmonicK | LinearCompressionRatio | CoordinationNum4Rate 
    | CoordinationNum6Rate | RandomSeed | Psi6Global
    
    FIGURE scatter  LinearCompressionRatio vs KBT, Psi6 as value 
    
    Example:
    import getDataAndScatter as scatt
    #scan seed
    seed=0
    while seed<9:
        index1=307
        lcr1=0.76
        #scan lcr and kT 0.76-0.99
        while lcr1<0.995:
            tt.workflow_mysql_to_data_pin_hex_to_kagome_random(index1,lcr1,seed)
            index1+=20
            lcr1+=0.01
    seed+=1
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    #list of HarmonicK
    k1=100.0
    k_step=100.0
    k_end=2000.0
    num=(k_end-k1)/k_step+1
    num=round(num)#get the num of repeat times
    #list of LCR
    lcr1=0.80
    lcr_step=0.01
    lcr_end=0.90
    lcr_num=(lcr_end-lcr1)/lcr_step+1
    lcr_num=round(lcr_num)
    
    record=np.zeros((lcr_num*num,8))
    
    
    count=0
    #scatter cycle
    for i in np.linspace(1,num,num):
        for j in np.linspace(1,lcr_num,lcr_num):
            kset=k1+(i-1)*k_step
            cond1=' where HarmonicK >'+str(kset-0.5*k_step)+' and HarmonicK <'+str(kset+0.5*k_step)
            lcrset=lcr1+(j-1)*lcr_step
            cond2=' and LinearCompressionRatio > '+str(lcrset-lcr_step*0.5)+' and LinearCompressionRatio <'+str(lcrset+lcr_step*0.5)
            data=osql.getDataFromMysql(table_name='pin_hex_to_kagome',search_condition=cond1+cond2)
            data=np.array(data)
            m3=np.mean(data[:,3])
            std3=np.std(data[:,3])
            m4=np.mean(data[:,4])
            std4=np.std(data[:,4])
            m6=np.mean(data[:,6])
            std6=np.std(data[:,6])
            record[count,:]=[lcrset,kset,m3,std3,m4,std4,m6,std6]
            count+=1
            #print(data)
            #print(m)
            #print(std)
    #rename "record"
    data=record
    #plot
    plt.figure()
    #plot LCR VS K, CN4 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,2])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('LCR VS K, CN4 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN4_as_value_pin_hex_to_kagome_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, CN6 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,4])# LCR VS K, CN6 as value
    #plt.show()
    plt.title('LCR VS K, CN6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN6_as_value_pin_hex_to_kagome_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,6])# LCR VS K, Psi6 as value
    #plt.show()
    plt.title('LCR VS K, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6_as_value_pin_hex_to_kagome_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, CN4std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,3])# LCR VS K, CN4std as value
    #plt.show()
    plt.title('LCR VS K, CN4std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN4std_as_value_pin_hex_to_kagome_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, CN6std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,5])# LCR VS K, CN6std as value
    #plt.show()
    plt.title('LCR VS K, CN6std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN6std_as_value_pin_hex_to_kagome_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,7])# LCR VS K, Psi6std as value
    #plt.show()
    plt.title('LCR VS K, Psi6std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6std_as_value_pin_hex_to_kagome_random'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_kagome_klt_2m(account='tplab'):
    R"""

    Note: the format of table_name='pin_hex_to_kagome_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_kagome_klt_2m()

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_kagome_klt_2m(account='remote')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    con='where SimuIndex >4917 and SimuIndex<5028'
    data=osql.getDataFromMysql(table_name='pin_hex_to_kagome_klt_2m',search_condition=con)
    data=np.array(data)
    prefix='/home/'+account+'/Downloads/'
    postfix = '_pin_hex_to_kagome_klt_2m_T01.png'

    plt.figure()
    #plot k VS T, CN4 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,5])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('k VS lcr, CN4 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('LinearCompressionRatio(1)')
    plt.ylabel('U trap (kBTm)[Kagome]')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_CN4_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_kagome_klt_2m_precise(account='tplab'):
    R"""

    Note: the format of table_name='pin_hex_to_kagome_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_kagome_klt_2m_precise(account='remote')

    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    con='where SimuIndex > 4717'#'where HarmonicK < 101'
    data=osql.getDataFromMysql(table_name='pin_hex_to_kagome_klt_2m',search_condition=con)
    data=np.array(data)
    prefix='/home/'+account+'/Downloads/'
    postfix = '_pin_hex_to_kagome_klt_2m_precise.png'

    plt.figure()
    #plot k VS T, CN4 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,5])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('k VS lcr, CN4 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('LinearCompressionRatio(1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_CN4_as_value_precise'+postfix
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_kagome_klt_2m_precise_random(account='tplab'):
    R"""

    Note: the format of table_name='pin_hex_to_kagome_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_kagome_klt_2m_precise(account='remote')

    select * from 
    pin_hex_to_kagome_klt_2m 
    where kT =1 and SimuIndex > 4717 and SimuIndex < 4778 
    order by LinearCompressionRatio, HarmonicK;

    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    index_start = 4718
    index_end = 4777
    count_index = index_end-index_start+1
    record = np.zeros((count_index,4))
    cont=' avg(LinearCompressionRatio), avg(HarmonicK), avg(CoordinationNum4Rate),std(CoordinationNum4Rate) '

    count=0
    #scatter cycle
    for i in np.linspace(index_start,index_end,count_index):
        con=' where SimuIndex = '+str(int(i))+' '
        data=osql.getDataFromMysql(table_name='pin_hex_to_kagome_klt_2m',search_condition=con,select_content=cont)
        data=np.array(data)
        record[count,:]=data#[lcrset,kset,m4,std4]
        count+=1

    prefix='/home/'+account+'/Downloads/'
    postfix = '_pin_hex_to_kagome_klt_2m_precise_random.png'
    data = record
    plt.figure()
    #plot k VS T, CN4 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,2])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('k VS lcr, CN4 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('LinearCompressionRatio(1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    png_filename=prefix+'K_VS_lcr_CN4_as_value_precise'+postfix
    #plt.show()
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot k VS T, CN4 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,3])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('k VS lcr, CN4std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('LinearCompressionRatio(1)')
    plt.ylabel('U trap (kBT)[Kagome]')
    plt.colorbar()
    png_filename=prefix+'K_VS_lcr_CN4std_as_value_precise'+postfix
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_kagome_klt_2m_random_mark(account='tplab',table_name = 'pin_hex_to_kagome'):
    R"""
    Note: the format of table_name=
    'pin_hex_to_kagome'
    | simu_index | HarmonicK | LinearCompressionRatio | CoordinationNum4Rate | 
    CoordinationNum6Rate | RandomSeed | Psi6Global |
    seed=0~9

    'pin_hex_to_kagome_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT  | CoordinationNum3Rate | 
    CoordinationNum4Rate | RandomSeed |
    where kT = 1, seed=0,9
    
    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    "H" <_> hexagon2
    "^"     triangle_up
    "v"     triangle_down
    "d" v^  thin_diamond
    example:
    """
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    #U_interaction=300*np.exp(-0.25)
    #table_name = 'pin_hex_to_kagome'

    cont='distinct LinearCompressionRatio'
    con=' order by LinearCompressionRatio asc'#where SimuIndex<4686
    lcrs=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)

    cont='distinct HarmonicK'
    con=' order by HarmonicK asc'
    ks=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)

    cont='distinct LinearCompressionRatio,HarmonicK'
    con=' order by LinearCompressionRatio,HarmonicK asc'
    lks=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)
    record_lk46 = np.zeros((len(lks),4))#(lcr,k,cn4,cn3)
    #record_lk36[:,0:2]=np.array(lks)   

    count = 0
    cont=' avg(CoordinationNum4Rate),avg(CoordinationNum6Rate) '
    lcr_step=0.001 
    for lcr in lcrs:
        for k in ks:
            lcr1 = np.array(lcr)
            lcr_min=lcr1 - 0.5*lcr_step
            lcr_max=lcr1 + 0.5*lcr_step
            con=' where HarmonicK='+str(int(k[0]))+\
                ' and LinearCompressionRatio >'+str(lcr_min[0])+' and LinearCompressionRatio <'+str(lcr_max[0])
            cn46_avg=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)
            cn46 = np.array(cn46_avg[0])
            #print(math.isnan(psi36[0]))#(np.nan(psi36_avg[0])
            if not cn46[0] is None:#math.isnan(psi36[0]):    
                #if not (k[0]<101 and k[0]>97):   
                #print(k)           
                #record_lk36[count,2:] = psi36[0:2]
                record_lk46[count,:] = [lcr[0],k[0],cn46[0],cn46[1]]
                count = count + 1
    record_lk46 = record_lk46[:count]
    #data = record_lk3
    
    record_lk46[:,1] = 0.5*record_lk46[:,1]# k -> Utrap(kT)
    return record_lk46

def workflow_mysql_to_data_pin_hex_to_kagome_klt_2m_mark(account='tplab',table_name = 'pin_hex_to_kagome_klt_2m'):
    R"""
    Note: the format of table_name=
    'pin_hex_to_kagome'
    | simu_index | HarmonicK | LinearCompressionRatio | CoordinationNum4Rate | 
    CoordinationNum6Rate | RandomSeed | Psi6Global |
    seed=0~9

    'pin_hex_to_kagome_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT  | CoordinationNum3Rate | 
    CoordinationNum4Rate | RandomSeed |
    where kT = 1, seed=0,9
    
    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    "H" <_> hexagon2
    "^"     triangle_up
    "v"     triangle_down
    "d" v^  thin_diamond
    example:
    """
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    #U_interaction=300*np.exp(-0.25)
    #table_name = 'pin_hex_to_kagome'

    cont='distinct LinearCompressionRatio,HarmonicK,CoordinationNum4Rate '
    con=' where kT=1 order by LinearCompressionRatio,HarmonicK asc'
    lks=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)
    
    record_lk4=np.array(lks)   

    record_lk4[:,1] = 0.5*record_lk4[:,1]# k -> Utrap(kT)
    return record_lk4
    
def workflow_mysql_to_data_depin_from_kagome_mark():
    R"""
    Note: the format of table_name=
    'depin_from_kagome'
    | simu_index | HarmonicK | LinearCompressionRatio 
    | CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed 
    | Psi6Global |
    """
    data=osql.getDataFromMysql(table_name='depin_from_kagome')#,search_condition=' where HarmonicK <100'
    data=np.array(data)

    p46=data[:,3]#/data[:,4]
    #p46=np.log10(p46)

    sz = np.shape(data)
    record_lklg4 = np.zeros((sz[0],3))
    record_lklg4[:,0] = data[:,2]#lcr
    record_lklg4[:,1] = data[:,1]*0.5#k
    record_lklg4[:,2] = p46
    list_lg3 = record_lklg4[:,1] < 21#U(kagome pin stable)~20
    list_lcr = record_lklg4[:,0] < 0.905
    list_r = np.logical_and(list_lg3,list_lcr)
    return record_lklg4[list_r]

def workflow_data_to_diagram_kagome_mark(account='tplab',save_mode="csv"):
    R"""
    Note: the format of table_name=
    'pin_hex_to_kagome'
    | simu_index | HarmonicK | LinearCompressionRatio | CoordinationNum4Rate | 
    CoordinationNum6Rate | RandomSeed | Psi6Global |
    seed=0~9

    'pin_hex_to_kagome_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT  | CoordinationNum3Rate | 
    CoordinationNum4Rate | RandomSeed |
    where kT = 1, seed=0,9

    'depin_from_kagome'
    | simu_index | HarmonicK | LinearCompressionRatio 
    | CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed 
    | Psi6Global |
    
    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    "H" <_> hexagon2
    "^"     triangle_up
    "v"     triangle_down
    "d" v^  thin_diamond
    example:
    """
    fig_name = 'diagram_KGM_mark'
    record_lk36 = workflow_mysql_to_data_pin_hex_to_kagome_klt_2m_random_mark()
    record_lk36_shallow = workflow_mysql_to_data_pin_hex_to_kagome_klt_2m_mark(table_name='pin_hex_to_kagome_klt_2m')
    record_lklg3 = workflow_mysql_to_data_depin_from_kagome_mark()

    mp = mysql_data_processor()
    title = 'lcr VS k, CN4 as value'#, Uparticle='+str(int(U_interaction))
    xlabel_name = 'Linear Compression Ratio (1)'
    ylabel_name = 'Usub ($k_BT$)[Kagome]'
    prefix='/home/'+account+'/Downloads/'

    pin_check = 0.7
    pin_check_s = 0.799
    depin_check = 0.95
    if save_mode=="pdf":
        postfix = '_'+fig_name+'.pdf'
        fig,ax = mp.draw_diagram_scatter_mark_multi(record_lk36,title, xlabel_name,ylabel_name,prefix,postfix=None,oop_check=pin_check)
        fig,ax = mp.draw_diagram_scatter_mark_multi(record_lk36_shallow,title, xlabel_name,ylabel_name,prefix,None,pin_check_s,fig,ax)
        mp.draw_diagram_scatter_mark_multi(record_lklg3,title, xlabel_name,ylabel_name,prefix,postfix,depin_check,fig,ax,"depin")
    
    elif save_mode=="csv":
        postfix = '_'+fig_name+'.csv'
        df0 = mp.get_diagram_data_mark_multi(record_lk36,title, xlabel_name,ylabel_name,prefix,csv_postfix=None,oop_check=pin_check)
        df0 = mp.get_diagram_data_mark_multi(record_lk36_shallow,title, xlabel_name,ylabel_name,prefix,csv_postfix=None,oop_check=pin_check_s)
        mp.get_diagram_data_mark_multi(record_lklg3,title, xlabel_name,ylabel_name,prefix,postfix,depin_check,"depin",df0)


def workflow_mysql_to_data_depin_from_kagome_part_random():#[x]
    R"""
    Introduction:
    table_name='depin_from_kagome_part_repeat'
    simu_index | HarmonicK | LinearCompressionRatio | CoordinationNum4Rate 
    | CoordinationNum3Rate | RandomSeed | Psi6Global
    
    FIGURE scatter  LinearCompressionRatio vs KBT, Psi6 as value 
    
    Example:
    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_depin_from_kagome_part_random()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    #list of HarmonicK
    k1=100.0
    k_step=100.0
    k_end=1000.0
    num=(k_end-k1)/k_step+1
    num=round(num)#get the num of repeat times
    #list of LCR
    lcr1=0.80
    lcr_step=0.01
    lcr_end=0.90
    lcr_num=(lcr_end-lcr1)/lcr_step+1
    lcr_num=round(lcr_num)
    
    record=np.zeros((lcr_num*num,8))
    
    
    count=0
    #scatter cycle
    for i in np.linspace(1,num,num):
        for j in np.linspace(1,lcr_num,lcr_num):
            kset=k1+(i-1)*k_step
            cond1=' where HarmonicK >'+str(kset-0.5*k_step)+' and HarmonicK <'+str(kset+0.5*k_step)
            lcrset=lcr1+(j-1)*lcr_step
            cond2=' and LinearCompressionRatio > '+str(lcrset-lcr_step*0.5)+' and LinearCompressionRatio <'+str(lcrset+lcr_step*0.5)
            data=osql.getDataFromMysql(table_name='depin_from_kagome_part_repeat',search_condition=cond1+cond2)
            data=np.array(data)
            m4=np.mean(data[:,3])
            std4=np.std(data[:,3])
            m3=np.mean(data[:,4])
            std3=np.std(data[:,4])
            m6=np.mean(data[:,6])
            std6=np.std(data[:,6])
            record[count,:]=[lcrset,kset,m4,std4,m3,std3,m6,std6]
            count+=1
            #print(data)
            #print(m)
            #print(std)
    #rename "record"
    data=record
    #plot
    plt.figure()
    #plot LCR VS K, CN4 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,2])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('LCR VS K, CN4 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN4_as_value_depin_from_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, CN3 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,4])# LCR VS K, CN3 as value
    #plt.show()
    plt.title('LCR VS K, CN3 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN3_as_value_depin_from_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,6])# LCR VS K, Psi6 as value
    #plt.show()
    plt.title('LCR VS K, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6_as_value_depin_from_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, CN4std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,3])# LCR VS K, CN4std as value
    #plt.show()
    plt.title('LCR VS K, CN4std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN4std_as_value_depin_from_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, CN3std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,5])# LCR VS K, CN3std as value
    #plt.show()
    plt.title('LCR VS K, CN3std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN3std_as_value_depin_from_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,7])# LCR VS K, Psi6std as value
    #plt.show()
    plt.title('LCR VS K, Psi6std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6std_as_value_depin_from_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()
    
def workflow_mysql_to_data_pin_hex_to_kagome_part():
    R"""
    Introduction:
    table_name="pin_hex_to_kagome_part_repeat"
    simu_index | HarmonicK | LinearCompressionRatio | 
    CoordinationNum4Rate | CoordinationNum3Rate | 
    RandomSeed | Psi6Global |
    
    FIGURE scatter  LinearCompressionRatio vs KBT, CN4 as value 
    
    Example:
    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_kagome_part()
    """
    U_interaction=300*np.exp(-0.25)
    points=osql.getDataFromMysql(table_name="pin_hex_to_kagome_part_repeat")#,search_condition="where LinearCompressionRatio < 0.855"
    points=np.asarray(points)

    plt.figure()
    plt.scatter(points[:,2],0.5*points[:,1],c=points[:,3])
    plt.colorbar()
    plt.title('LCR VS K,CN4 as value, kBT=1, Uparticle='+str(int(U_interaction))+',pin' )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[kagome_part]')

    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_vs_K_CN4_as_value_pin_hex_to_kagome_part'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    plt.scatter(points[:,2],0.5*points[:,1],c=points[:,4])
    plt.colorbar()
    plt.title('LCR VS K,CN3 as value, kBT=1, Uparticle='+str(int(U_interaction))+',pin' )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[kagome_part]')
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_vs_K_CN3_as_value_pin_hex_to_kagome_part'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_depin_from_kagome_part_klt(account='tplab'):
    R"""
    Note: the format of table_name='depin_from_kagome_part_klt'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_depin_from_kagome_part_klt()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    data=osql.getDataFromMysql(table_name='depin_from_kagome_part_klt')
    data=np.array(data)
    prefix='/home/'+account+'/Downloads/'
    postfix = '_depin_from_kagome_part_klt.png'

    plt.figure()
    #plot k VS T, CN4 as value
    plt.scatter(data[:,1]*0.5,data[:,3],c=data[:,5])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('k VS T, CN4 as value, LCRc, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('U trap (kBT)[Kagome part]')
    plt.ylabel('kBT')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_CN4_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_kagome_part_klt(account='tplab'):
    R"""

    Note: the format of table_name='pin_hex_to_kagome_part_klt'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_kagome_part_klt()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    data=osql.getDataFromMysql(table_name='pin_hex_to_kagome_part_klt')
    data=np.array(data)
    prefix='/home/'+account+'/Downloads/'
    postfix = '_pin_hex_to_kagome_part_klt.png'

    plt.figure()
    #plot k VS T, CN4 as value
    plt.scatter(data[:,1]*0.5,data[:,3],c=data[:,5])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('k VS T, CN4 as value, LCRc, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('U trap (kBT)[Kagome part]')
    plt.ylabel('kBT')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_CN4_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_kagome_part_klt_2m(account='tplab'):
    R"""

    Note: the format of table_name='pin_hex_to_kagome_part_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_kagome_part_klt_2m('remote')

    EXAMPLE:
        # low-T trap
        $select * from pin_hex_to_kagome_part_klt_2m where kT < 0.2;
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    table_name='pin_hex_to_kagome_part_klt_2m'
    condition = 'where kT < 0.2'
    data=osql.getDataFromMysql(table_name=table_name,search_condition=condition)
    data=np.array(data)
    prefix='/home/'+account+'/Downloads/'
    postfix = '_'+table_name+'_low_T'+'.png'
    
    plt.figure()
    #plot k VS T, CN4 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,5])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('k VS lcr, CN4 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('LinearCompressionRatio(1)')
    plt.ylabel('U trap (kBT)[Kagome part]')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_CN4_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_kagome_part_random():
    R"""
    Introduction:
    table_name="pin_hex_to_kagome_part_repeat"
    simu_index | HarmonicK | LinearCompressionRatio | 
    CoordinationNum4Rate | CoordinationNum3Rate | 
    RandomSeed | Psi6Global |
    
    FIGURE scatter  LinearCompressionRatio vs KBT, CN4 as value 
    
    Example:
    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_kagome_part_random()
    """
    #list of HarmonicK
    k1=100.0
    k_step=100.0
    k_end=2000.0
    num=(k_end-k1)/k_step+1
    num=round(num)#get the num of repeat times
    #list of LCR
    lcr1=0.80
    lcr_step=0.01
    lcr_end=0.90
    lcr_num=(lcr_end-lcr1)/lcr_step+1
    lcr_num=round(lcr_num)
    
    record=np.zeros((lcr_num*num,8))
    
    
    count=0
    #scatter cycle
    for i in np.linspace(1,num,num):
        for j in np.linspace(1,lcr_num,lcr_num):
            kset=k1+(i-1)*k_step
            cond1=' where HarmonicK >'+str(kset-0.5*k_step)+' and HarmonicK <'+str(kset+0.5*k_step)
            lcrset=lcr1+(j-1)*lcr_step
            cond2=' and LinearCompressionRatio > '+str(lcrset-lcr_step*0.5)+' and LinearCompressionRatio <'+str(lcrset+lcr_step*0.5)
            data=osql.getDataFromMysql(table_name='pin_hex_to_kagome_part_repeat',search_condition=cond1+cond2)
            data=np.array(data)
            m4=np.mean(data[:,3])
            std4=np.std(data[:,3])
            m3=np.mean(data[:,4])
            std3=np.std(data[:,4])
            m6=np.mean(data[:,6])
            std6=np.std(data[:,6])
            record[count,:]=[lcrset,kset,m4,std4,m3,std3,m6,std6]
            count+=1
            #print(data)
            #print(m)
            #print(std)
    #rename "record"
    data=record
    
    U_interaction=300*np.exp(-0.25)
    #plot
    plt.figure()
    #plot LCR VS K, CN4 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,2])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('LCR VS K, CN4 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN4_as_value_pin_hex_to_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, CN3 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,4])# LCR VS K, CN3 as value
    #plt.show()
    plt.title('LCR VS K, CN3 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN3_as_value_pin_hex_to_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6 as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,6])# LCR VS K, Psi6 as value
    #plt.show()
    plt.title('LCR VS K, Psi6 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6_as_value_pin_hex_to_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, CN4std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,3])# LCR VS K, CN4std as value
    #plt.show()
    plt.title('LCR VS K, CN4std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN4std_as_value_pin_hex_to_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, CN3std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,5])# LCR VS K, CN3std as value
    #plt.show()
    plt.title('LCR VS K, CN3std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_CN3std_as_value_pin_hex_to_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()
    
    plt.figure()
    #plot LCR VS K, Psi6std as value
    plt.scatter(data[:,0],data[:,1]*0.5,c=data[:,7])# LCR VS K, Psi6std as value
    #plt.show()
    plt.title('LCR VS K, Psi6std as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part]')
    plt.colorbar()
    prefix='/home/tplab/Downloads/'
    png_filename=prefix+'LCR_VS_K_Psi6std_as_value_depin_from_kagome_part_random'
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_kagome_part_random_oop():
    R"""
    Introduction:
    table_name="pin_hex_to_kagome_part_repeat"
    simu_index | HarmonicK | LinearCompressionRatio | 
    CoordinationNum4Rate | CoordinationNum3Rate | 
    RandomSeed | Psi6Global |
    
    FIGURE scatter  LinearCompressionRatio vs KBT, CN4 as value 
    
    Example:
    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_kagome_part_random_oop()
    """
    import getDataAndScatterOOP as sop
    table_name="pin_hex_to_kagome_part_repeat"
    workflow_name = "pin_hex_to_kagome_part_random"
    workflow = sop.workflow_mysql_to_data(table_name=table_name,workflow_name=workflow_name)
    #workflow.set_parameters_k(k_end=2000.0)
    #workflow.set_parameters_lcr(lcr1=0.86602,lcr_end=0.86602)
    workflow.get_data_from_txt("/home/tplab/Downloads/pin_hex_to_kagome_part_diagram_c")
    #workflow.save_as_txt("/home/tplab/Downloads/pin_hex_to_kagome_part_diagram_c")
    workflow.plot()

def workflow_mysql_to_data_pin_hex_to_kagome_part_klt_2m_random_mark(account='tplab'):
    R"""
    Note: the format of table_name=
    'pin_hex_to_kagome_part_klt'
    'pin_hex_to_kagome_part_repeat'
    simu_index | HarmonicK | LinearCompressionRatio | 
    CoordinationNum4Rate | CoordinationNum3Rate | 
    RandomSeed | Psi6Global | 
    'pin_hex_to_kagome_part_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed | 
    
    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    "H" <_> hexagon2
    "^"     triangle_up
    "v"     triangle_down
    "d" v^  thin_diamond
    example:
    """
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    #U_interaction=300*np.exp(-0.25)
    table_name = 'pin_hex_to_kagome_part_repeat'

    cont='distinct LinearCompressionRatio'
    con=' order by LinearCompressionRatio asc'#where SimuIndex<4686
    lcrs=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)

    cont='distinct HarmonicK'
    con=' order by HarmonicK asc'
    ks=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)

    cont='distinct LinearCompressionRatio,HarmonicK'
    con=' order by LinearCompressionRatio,HarmonicK asc'
    lks=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)
    record_lk43 = np.zeros((len(lks),4))#(lcr,k,cn4,cn3)
    #record_lk36[:,0:2]=np.array(lks)   

    count = 0
    cont=' avg(CoordinationNum4Rate),avg(CoordinationNum3Rate) '
    lcr_step=0.001 
    for lcr in lcrs:
        for k in ks:
            lcr1 = np.array(lcr)
            lcr_min=lcr1 - 0.5*lcr_step
            lcr_max=lcr1 + 0.5*lcr_step
            con=' where HarmonicK='+str(int(k[0]))+\
                ' and LinearCompressionRatio >'+str(lcr_min[0])+' and LinearCompressionRatio <'+str(lcr_max[0])
            cn43_avg=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)
            cn43 = np.array(cn43_avg[0])
            #print(math.isnan(psi36[0]))#(np.nan(psi36_avg[0])
            if not cn43[0] is None:#math.isnan(psi36[0]):    
                if k[0]>101*2:   
                    #print(k)           
                    #record_lk36[count,2:] = psi36[0:2]
                    record_lk43[count,:] = [lcr[0],k[0],cn43[0],cn43[1]]
                    count = count + 1
    record_lk43 = record_lk43[:count]
    #data = record_lk3
    
    record_lk43[:,1] = 0.5*record_lk43[:,1]# k -> Utrap(kT)
    return record_lk43
    
def workflow_mysql_to_data_depin_from_kagome_part_mark():
    R"""
    Note: the format of table_name=
    table_name='depin_from_kagome_part_repeat'
    simu_index | HarmonicK | LinearCompressionRatio | CoordinationNum4Rate 
    | CoordinationNum3Rate | RandomSeed | Psi6Global
    """
    table_name='depin_from_kagome_part_repeat'

    cont='distinct LinearCompressionRatio'
    con=' order by LinearCompressionRatio asc'#where SimuIndex<4686
    lcrs=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)

    cont='distinct HarmonicK'
    con=' order by HarmonicK asc'
    ks=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)

    cont='distinct LinearCompressionRatio,HarmonicK'
    con=' order by LinearCompressionRatio,HarmonicK asc'
    lks=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)
    record_lk43 = np.zeros((len(lks),3))#(lcr,k,cn4)
    #record_lk36[:,0:2]=np.array(lks)   

    count = 0
    cont=' avg(CoordinationNum4Rate) '#,avg(CoordinationNum3Rate) 
    lcr_step=0.001 
    for lcr in lcrs:
        for k in ks:
            lcr1 = np.array(lcr)
            lcr_min=lcr1 - 0.5*lcr_step
            lcr_max=lcr1 + 0.5*lcr_step
            con=' where HarmonicK='+str(int(k[0]))+\
                ' and LinearCompressionRatio >'+str(lcr_min[0])+' and LinearCompressionRatio <'+str(lcr_max[0])
            cn43_avg=osql.getDataFromMysql(table_name=table_name,
                                search_condition=con,select_content=cont)
            cn43 = np.array(cn43_avg[0])
            #print(math.isnan(psi36[0]))#(np.nan(psi36_avg[0])
            if not cn43[0] is None:#math.isnan(psi36[0]):    
                if k[0]<101*2:   #U(Kagome_part pin stable)~101
                    #print(k)           
                    #record_lk36[count,2:] = psi36[0:2]
                    record_lk43[count,:] = [lcr[0],k[0],cn43[0]]
                    count = count + 1
    data = record_lk43[:count]
    data[:,1] = data[:,1]*0.5#k
    return data

def workflow_data_to_diagram_kagome_part_mark(account='tplab',save_mode="csv"):
    R"""
    Note: the format of table_name=
    'pin_hex_to_kagome_part_klt'
    'pin_hex_to_kagome_part_repeat'
    simu_index | HarmonicK | LinearCompressionRatio | 
    CoordinationNum4Rate | CoordinationNum3Rate | 
    RandomSeed | Psi6Global | 
    'pin_hex_to_kagome_part_klt_2m'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed | 

    'depin_from_kagome_part'
    'depin_from_kagome_part_klt'
    'depin_from_kagome_part_repeat'
    simu_index | HarmonicK | LinearCompressionRatio | CoordinationNum4Rate 
    | CoordinationNum3Rate | RandomSeed | Psi6Global
    
    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    "H" <_> hexagon2
    "^"     triangle_up
    "v"     triangle_down
    "d" v^  thin_diamond
    example:
    """
    fig_name = 'diagram_KGP_mark'
    record_lk36 = workflow_mysql_to_data_pin_hex_to_kagome_part_klt_2m_random_mark()
    record_lk36shallow = workflow_mysql_to_data_pin_hex_to_kagome_part_klt_2m_random_mark()
    record_lklg3 = workflow_mysql_to_data_depin_from_kagome_part_mark()

    mp = mysql_data_processor()
    title = 'lcr VS k, CN4 as value'#, Uparticle='+str(int(U_interaction))
    xlabel_name = 'Linear Compression Ratio (1)'
    ylabel_name = 'Usub ($k_BT$)[Kagome_part]'
    prefix='/home/'+account+'/Downloads/'

    pin_check = 0.66
    depin_check = 0.93
    if save_mode=="pdf":
        postfix = '_'+fig_name+'.pdf'
        fig,ax = mp.draw_diagram_scatter_mark_multi(record_lk36,title, xlabel_name,ylabel_name,prefix,postfix=None,oop_check=pin_check)
        mp.draw_diagram_scatter_mark_multi(record_lklg3,title, xlabel_name,ylabel_name,prefix,postfix,depin_check,fig,ax,"depin")
    
    elif save_mode=="csv":
        postfix = '_'+fig_name+'.csv'
        df0 = mp.get_diagram_data_mark_multi(record_lk36,title, xlabel_name,ylabel_name,prefix,csv_postfix=None,oop_check=pin_check)
        mp.get_diagram_data_mark_multi(record_lklg3,title, xlabel_name,ylabel_name,prefix,postfix,depin_check,"depin",df0)


def workflow_mysql_to_data_depin_from_kagome_part_cycle(account='tplab'):
    R"""
    Note: the format of table_name='depin_from_kagome_part_cycle'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_depin_from_kagome_part_cycle()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    data=osql.getDataFromMysql(table_name='depin_from_kagome_part_cycle')
    data=np.array(data)
    prefix='/home/'+account+'/Downloads/'
    postfix = '_depin_from_kagome_c.png'

    plt.figure()
    #plot k VS T, CN4 as value
    plt.scatter(data[:,1]*0.5,data[:,3],c=data[:,5])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('k VS T, CN4 as value, LCRc, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('U trap (kBT)[Kagome]')
    plt.ylabel('kBT')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_CN4_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()
    
def workflow_txt_to_data_depin_from_kagome_part_cycle(account='tplab'):
    R"""
    Note: the format of table_name='depin_from_kagome_part_cycle'
    [ simu_index | HarmonicK | LinearCompressionRatio | 
    CoordinationNum4Rate | CoordinationNum3Rate | RandomSeed |
      Psi6Global]
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_txt_to_data_depin_from_kagome_part_cycle()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    
    prefix='/home/'+account+'/Downloads/'
    postfix = '_depin_from_kagome_c.png'

    data86=np.loadtxt(prefix+'3526-3535kl4')
    data80=np.loadtxt(prefix+'3536-3545kl4')
    data862=np.loadtxt(prefix+'3546-3555kl4')
    sz = np.shape(data86)
    data = np.zeros((sz[0]*3,sz[1]))
    data[:sz[0]]=data86
    data[sz[0]:sz[0]*2]=data80
    data[sz[0]*2:]=data862

    plt.figure()
    #plot k VS T, CN4 as value
    plt.scatter(data[:,2],data[:,1]*0.5,c=data[:,3])# LCR VS K, CN4 as value
    #plt.show()
    #plt.show()
    plt.title('LCR VS K, CN4 as value, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel('U trap (kBT)[Kagome_part_cycle]')
    plt.colorbar()
    png_filename=prefix+'LCR_VS_K_CN4_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_kagome_part_cycle(account='tplab'):
    R"""
    Note: the format of table_name='pin_hex_to_kagome_part_cycle'
    | SimuIndex | HarmonicK | LinearCompressionRatio | kT | 
    CoordinationNum3Rate | CoordinationNum4Rate | RandomSeed | 
    
    FIGURE scatter  HarmonicK vs KBT, Psi6 as value 

    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_kagome_part_cycle()
    """
    import matplotlib.pyplot as plt
    import numpy as np
    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction=300*np.exp(-0.25)

    data=osql.getDataFromMysql(table_name='pin_hex_to_kagome_part_cycle')
    data=np.array(data)
    prefix='/home/'+account+'/Downloads/'
    postfix = '_pin_hex_to_kagome_part_c.png'

    plt.figure()
    #plot k VS T, CN4 as value
    plt.scatter(data[:,1]*0.5,data[:,3],c=data[:,5])# LCR VS K, CN4 as value
    #plt.show()
    plt.title('k VS T, CN4 as value, LCR0.88, Uparticle='+str(int(U_interaction)) )
    plt.xlabel('U trap (kBT)[Kagome part cycle]')
    plt.ylabel('kBT')
    plt.colorbar()
    png_filename=prefix+'K_VS_T_CN4_as_value'+postfix
    plt.savefig(png_filename)
    plt.close()

def workflow_mysql_to_data_pin_hex_to_cairo():
    R"""
    Introduction:
    table_name='pin_hex_to_cairo'
        Simu_Index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum3Rate | CoordinationNum4Rate | 
        CoordinationNum6Rate |RandomSeed |
    
    FIGURE scatter  LinearCompressionRatio vs U_trap, CN3 as value 
    
    Example:
    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_cairo_random()

    """
    #control table
    trap_name = 'cairo'
    table_name = "pin_hex_to_cairo_egct" #"pin_hex_to_cairo"
    result_prefix='/home/tplab/Downloads/'
    save_data_txt = True
    if save_data_txt:
        data_file_name = "cairo_diagram_1_accurate"
        save_file_name=result_prefix+data_file_name
        

    k1=0.0
    k_step=10.0
    k_end=100.0
    
    lcr1=0.60
    lcr_step=0.01
    lcr_end=0.80
    

    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction = 300*np.exp(-0.25)
    U_interaction = str(int(U_interaction))
    
    #list of HarmonicK
    num=(k_end-k1)/k_step+1
    num=round(num)#get the num of repeat times
    #list of LCR
    lcr_num=(lcr_end-lcr1)/lcr_step+1
    lcr_num=round(lcr_num)
    
    #record=np.zeros((lcr_num*num,8))#[?]
    

    cond1=' where HarmonicK < 101'
    cond2=' and RandomSeed > 8.5'
    search_condition=cond1+cond2
    data=osql.getDataFromMysql(table_name=table_name,search_condition=search_condition)
    data=np.array(data)
    #[?]
    #m3=np.mean(data[:,3])#CN3
    #m4=np.mean(data[:,4])#CN4
    #m6=np.mean(data[:,5])#CN6
    data=data[:,1:6]


    #save data
    if save_data_txt:
        np.savetxt(save_file_name,data)


    #plot
    ylabel_name = 'U trap (kBT)['+trap_name+']'
    
    #plot LCR VS K, CN3 as value
    plot_diagram_value_and_std\
        (data[:,1],data[:,0],data[:,2],data[:,2],
        'CN3',U_interaction,result_prefix,table_name,ylabel_name)
    #plot LCR VS K, CN4 as value
    plot_diagram_value_and_std\
        (data[:,1],data[:,0],data[:,3],data[:,3],
        'CN4',U_interaction,result_prefix,table_name,ylabel_name)
    #plot LCR VS K, CN6 as value
    plot_diagram_value_and_std\
        (data[:,1],data[:,0],data[:,4],data[:,4],
        'CN6',U_interaction,result_prefix,table_name,ylabel_name)

def workflow_mysql_to_data_pin_hex_to_cairo_egct(account='tplab'):
    R"""
    Introduction:
    table_name='pin_hex_to_cairo'
        Simu_Index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum3Rate | CoordinationNum4Rate | 
        CoordinationNum6Rate |RandomSeed |
    
    FIGURE scatter  LinearCompressionRatio vs U_trap, PCairo as value 
    
    Example:
    import getDataAndDiagram as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_cairo_egct()

    """
    #control table
    trap_name = 'cairo'
    table_name = "pin_hex_to_cairo_egct2lcra" #"pin_hex_to_cairo"
    result_prefix='/home/'+account+'/Downloads/'
    save_data_txt = False
    if save_data_txt:
        data_file_name = "cairo_diagram_1_accurate"
        save_file_name=result_prefix+data_file_name

    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction = 300*np.exp(-0.25)
    U_interaction = str(int(U_interaction))

    cond1=' where HarmonicK>99 '#  < 101
    cond2=' and RandomSeed > 8.5'
    search_condition=cond1+cond2
    data=osql.getDataFromMysql(table_name=table_name,search_condition=search_condition)
    data=np.array(data)
    #| SimuIndex | HarmonicK | LinearCompressionRatio | CoordinationNum3Rate | 
    # CoordinationNum4Rate | CoordinationNum6Rate | PCairo   | RandomSeed |
    data=data[:,1:7]


    #save data
    if save_data_txt:
        np.savetxt(save_file_name,data)

    #plot
    ylabel_name = 'U trap (kBT)['+trap_name+']'
    
    #plot LCR VS K, PCairo as value
    plot_diagram_value\
        (data[:,1],data[:,0],data[:,5],
        'PCairo',U_interaction,result_prefix,table_name,ylabel_name)

def workflow_mysql_to_data_pin_hex_to_cairo_random():#[x]
    R"""
    Introduction:
    table_name='pin_hex_to_cairo'
        Simu_Index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum3Rate | CoordinationNum4Rate | 
        CoordinationNum6Rate |RandomSeed |
    
    FIGURE scatter  LinearCompressionRatio vs U_trap, CN3 as value 
    
    Example:
    import getDataAndScatter as scatt
    scatt.workflow_mysql_to_data_pin_hex_to_cairo_random()

    """
    #control table
    trap_name = 'cairo'
    table_name = "pin_hex_to_cairo"
    result_prefix='/home/tplab/Downloads/'
    save_data_txt = True
    if save_data_txt:
        data_file_name = "cairo_diagram_1"
        save_file_name=result_prefix+data_file_name
        

    k1=100.0
    k_step=100.0
    k_end=1000.0
    
    lcr1=0.60
    lcr_step=0.01
    lcr_end=0.80
    

    #getDataToMysql
    import opertateOnMysql as osql
    U_interaction = 300*np.exp(-0.25)
    U_interaction = str(int(U_interaction))
    
    #list of HarmonicK
    num=(k_end-k1)/k_step+1
    num=round(num)#get the num of repeat times
    #list of LCR
    lcr_num=(lcr_end-lcr1)/lcr_step+1
    lcr_num=round(lcr_num)
    
    record=np.zeros((lcr_num*num,8))#[?]
    
    count=0
    #scatter cycle
    for i in np.linspace(1,num,num):
        for j in np.linspace(1,lcr_num,lcr_num):
            kset=k1+(i-1)*k_step
            cond1=' where HarmonicK >'+str(kset-0.5*k_step)+' and HarmonicK <'+str(kset+0.5*k_step)
            lcrset=lcr1+(j-1)*lcr_step
            cond2=' and LinearCompressionRatio > '+str(lcrset-lcr_step*0.5)+' and LinearCompressionRatio <'+str(lcrset+lcr_step*0.5)
            search_condition=cond1+cond2
            data=osql.getDataFromMysql(table_name=table_name,search_condition=search_condition)
            data=np.array(data)
            #[?]
            m3=np.mean(data[:,3])#CN3
            std3=np.std(data[:,3])
            m4=np.mean(data[:,4])#CN4
            std4=np.std(data[:,4])
            m6=np.mean(data[:,5])#CN6
            std6=np.std(data[:,5])
            record[count,:]=[lcrset,kset,m3,std3,m4,std4,m6,std6]
            count+=1
            #print(data)
            #print(m)
            #print(std)
    #rename "record"
    data=record

    #save data
    if save_data_txt:
        np.savetxt(save_file_name,data)


    #plot
    ylabel_name = 'U trap (kBT)['+trap_name+']'
    
    #plot LCR VS K, CN3 as value
    plot_diagram_value_and_std\
        (data[:,0],data[:,1],data[:,2],data[:,3],
        'CN3',U_interaction,result_prefix,table_name,ylabel_name)
    #plot LCR VS K, CN4 as value
    plot_diagram_value_and_std\
        (data[:,0],data[:,1],data[:,4],data[:,5],
        'CN4',U_interaction,result_prefix,table_name,ylabel_name)
    #plot LCR VS K, CN6 as value
    plot_diagram_value_and_std\
        (data[:,0],data[:,1],data[:,6],data[:,7],
        'CN6',U_interaction,result_prefix,table_name,ylabel_name)
    
def plot_diagram_value_and_std(dx,dy,dv,dstd,value_name,U_interaction,result_prefix,table_name,ylabel_name):
    postfix = ', Uparticle='
    title1 = 'LCR VS K, '+value_name+' as value'
    title2 = 'LCR VS K, '+value_name+'std as value'
    plt.figure()
    #plot LCR VS K, Data as value
    plt.scatter(dx,dy*0.5,c=dv)# LCR VS K, Data as value
    #plt.show()
    plt.title(title1+postfix+U_interaction )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel(ylabel_name)
    plt.colorbar()
    png_filename=result_prefix+title1+'_'+table_name+'_random'
    plt.savefig(png_filename)
    plt.close()

    plt.figure()
    #plot LCR VS K, Datastd as value
    plt.scatter(dx,dy*0.5,c=dstd)# LCR VS K, Datastd as value
    #plt.show()
    plt.title(title2+postfix+U_interaction )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel(ylabel_name)
    plt.colorbar()
    png_filename=result_prefix+title2+'_'+table_name+'_random'
    plt.savefig(png_filename)
    plt.close()

def plot_diagram_value(dx,dy,dv,value_name,U_interaction,result_prefix,table_name,ylabel_name):
    postfix = ', Uparticle='
    title1 = 'LCR VS K, '+value_name+' as value'
    plt.figure()
    #plot LCR VS K, Data as value
    plt.scatter(dx,dy*0.5,c=dv)# LCR VS K, Data as value
    #plt.show()
    plt.title(title1+postfix+U_interaction )
    plt.xlabel('Linear Compression Ratio (1)')
    plt.ylabel(ylabel_name)
    plt.colorbar()
    png_filename=result_prefix+title1+'_'+table_name
    plt.savefig(png_filename)
    plt.close()

def save_image_stack(gsd_file):
    R"""
    Example:
        import getDataAndScatter as scatt
        simu_index = 1369
        prefix_gsd = '/home/tplab/hoomd-examples_0/trajectory_auto'
        postfix_gsd = '.gsd'
        filename_gsd = prefix_gsd+str(simu_index)+postfix_gsd
        scatt.save_image_stack(gsd_file=filename_gsd)
    """
    #from celluloid import camera
    import matplotlib 
    matplotlib.use(backend="agg")#Backend agg is non-interactive backend. Turning interactive mode off.
    #Backend Qt5Agg is interactive backend. Turning interactive mode on.
    import os
    import gsd.hoomd
    gsd_data=gsd.hoomd.open(gsd_file)#open a gsd file
    frame_num_max=len(gsd_data)

    frame_num=0
    dis = None
    while frame_num < frame_num_max:
        snap=gsd_data.read_frame(frame_num)#take a snapshot of the N-th frame
        positions=snap.particles.position

        plt.figure()
        if dis is None:
            xmax = max(positions[:,0]) #- 3
            ymax = max(positions[:,1]) #- 3
            xmin = min(positions[:,0]) #+ 3
            ymin = min(positions[:,1]) #+ 3
            dis = min(xmax,ymax,-xmin,-ymin)
            #dis = 5
        plt.scatter(positions[:,0],positions[:,1])
        plt.axis('equal')
        plt.xlim([-dis,dis])
        plt.ylim([-dis,dis])
        
        #plt.show()

        prefix_old="/home/tplab/hoomd-examples_0"
        folder_name=gsd_file.strip(prefix_old)
        folder_name=folder_name.strip(".gsd")#folder_name=prefix+png_filename_as_folder
        folder_name="t"+folder_name#"t" is necessary in case of "/t" deleted before
        prefix='/home/tplab/Downloads/'
        png_filename=prefix+folder_name+"/"+folder_name+"_"+str(frame_num)
        #check if the folder exists
        isExists=os.path.exists(prefix+folder_name)
        if isExists:
            plt.savefig(png_filename)
        else:
            os.makedirs(prefix+folder_name)
            plt.savefig(png_filename)
        plt.close()

        frame_num=frame_num+1

    return folder_name

    R"""
    save multiple PNGs in an file like
    "prefix/FileName/FileName_N.PNG"

    #用matplotlib和pillow保存gif图
    #https://blog.csdn.net/qq_28888837/article/details/85778395
    """


def getHoneycombPart():
    trap_prefix='/home/tplab/hoomd-examples_0/'
    trap_filename1=trap_prefix+"testhoneycomb3-8-12-part1"
    trap_filename2=trap_prefix+"testhoneycomb3-8-12-part2"

    plt.figure()
    points=np.loadtxt(trap_filename1)
    plt.scatter(points[:,1],points[:,2],c='b')
    points=np.loadtxt(trap_filename2)
    plt.scatter(points[:,0],points[:,1],c='y')
    plt.show()
    #if not png_filename==None:
    #    plt.savefig(png_filename)
    #plt.close()
    
def draw_points_and_traps_simu_result():
    R"""
    Introduction:
        draw simulated configurations using blue dots and traps using red crosses.
    Examples:
        "testhex3-16-8"
        "testhoneycomb3-8-12"
        "testkagome3-9-6"
    """    
    import hoomd
    import hoomd.azplugins
    import numpy
    from matplotlib import pyplot
    import hoomd.azplugins.sequence_generator as sg
    
    LinearCompressionRatio=0.84#LinearCompressionRatio
    HarmonicK=100#HarmonicK
    result_index = "index1773"#simu_index
    parameter="lcr"+str(LinearCompressionRatio)+"k"+str(HarmonicK)
    #filename of traps
    trap_prefix='/home/tplab/hoomd-examples_0/'
    trap_filename=trap_prefix+"testhoneycomb3-8-12_rectangle1"#data of trap position

    #generate a trap sequence
    pp = sg.sequence()
    pp.generate_honeycomb_rectangle1(a=3,n=[8,12])
    pp.save(trap_filename)

    data=numpy.loadtxt(trap_filename)
    pyplot.figure()
    pyplot.scatter(data[:,0]*LinearCompressionRatio, 
                   data[:,1]*LinearCompressionRatio,
                   c='r',marker = 'x')


    prefix= "/home/tplab/Downloads/"
    filename=prefix+result_index
    #filename of simulated results, containing n rows of [x,y,0]

    data=numpy.loadtxt(filename)
    pyplot.scatter(data[:,0], data[:,1])

    pyplot.title(result_index+parameter)
    pyplot.axis('equal')
    pyplot.xlim([-20,20])
    pyplot.ylim([-16,16])
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    pyplot.show()
    png_filename=prefix+result_index+parameter
    plt.savefig(png_filename)
    plt.close()

def draw_points_and_traps_result(simu_index,table_name,):
    R"""
    Introduction:
        draw simulated configurations using blue dots and traps using red crosses.
    Examples:
        "testhex3-16-8"
        "testhoneycomb3-8-12"
        "testkagome3-9-6"
    """    

    #retreiveDataFromMysql
    R"""
    Note: the format of table_name='pin_hex_to_kagome'
        simu_index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed |
        Psi6Global|
    """
    import opertateOnMysql as osql
    simu_index = str(simu_index)
    cond1 = ' where simu_index ='+str(simu_index)
    cond2 = ' and RandomSeed = 9'
    param=osql.getDataFromMysql(table_name=table_name,search_condition=cond1+cond2)
    param=np.array(param)
    param=param[0]
    HarmonicK = int(param[1])
    LinearCompressionRatio = param[2]
    #CN4R = param[3]

    parameter="lcr"+str(LinearCompressionRatio)+"~k"+str(HarmonicK)
    #filename of traps (filename from symmetry_transformation_auto_xx) 
    trap_prefix='/home/tplab/hoomd-examples_0/'
    trap_filename=trap_prefix+"testkagome3-9-6"#data of trap position

    traps_points=np.loadtxt(trap_filename)
    traps_points[:]=traps_points[:]*LinearCompressionRatio
    plt.figure()
    plt.scatter(traps_points[:,0], 
                   traps_points[:,1],
                   c='r',marker = 'x')
    trap_max=np.max(traps_points)
    print(max)
    trap_min=np.min(traps_points)
    print(min)
    #points
    result_prefix = "/home/tplab/Downloads/"
    points_filename = result_prefix +"index"+str(simu_index)
    #filename of simulated results, containing n rows of [x,y,0]

    particle_points=np.loadtxt(points_filename)
    plt.scatter(particle_points[:,0], particle_points[:,1])

    plt.title(simu_index+parameter)
    plt.axis('equal')
    #plt.xlim([-20,20])
    #plt.ylim([-16,16])
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.show()
    png_filename=result_prefix+"index"+simu_index+parameter+".png"
    plt.savefig(png_filename)
    plt.close()

def showTrapsMap(account):
    import hoomd.azplugins.sequence_generator as sg
    #get points
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(1, 1, N)
    vals[:, 1] = np.linspace(1, 128/256, N)
    vals[:, 2] = np.linspace(1, 128/256, N)
    newcmp = ListedColormap(vals)#LinearSegmentedColormap(vals)#ListedColormap(vals)
    
    #import matplotlib as mpl
    """
    cmp = plt.get_cmap('autumn')
    cmp.reversed('autumn_r')
    """
    cmp = plt.get_cmap('Reds')
    rcut=1.0
    cmap_name = 'Reds'#'autumn_r'#'autumn'#newcmp#'binary'#
    transparency = 0.5#0.3
    particle_size = 100

    trap_prefix='/home/'+account+'/hoomd-examples_0/'
    trap_filename=trap_prefix+"testhoneycomb3-8-12-part1"#data of trap position

    new1 = False#True#
    if new1:
        #generate a trap sequence
        pp = sg.sequence()
        pp.generate_honeycomb_part1(a=3,n=[8,12])
        pp.save(trap_filename)
    
    #set traps
    linear_compression_ratio=0.79#0.79
    traps_pos=np.loadtxt(trap_filename)
    traps_pos=np.dot(linear_compression_ratio,traps_pos)

    """
    xmax = np.maximum(traps_pos[:,0])
    ymax = np.maximum(traps_pos[:,1])
    xmin = np.minimum(traps_pos[:,0]) 
    ymin = np.minimum(traps_pos[:,1])
    length = (xmax - xmin + ymax -ymin)/2.0
    steps = length/(rcut/10.0)
    """
    max = np.max(traps_pos)
    min = np.min(traps_pos)
    length = (max - min)
    steps = length/(rcut/10.0)
    #plt.style.use('_mpl-gallery-nogrid')

    # make data
    X, Y = np.meshgrid(np.linspace(min, max, steps.astype(int)), np.linspace(min, max, steps.astype(int)))
    HarmonicK = 100
    #origin = np.zeros((1,2))
    sz = np.shape(traps_pos)
    i = 0
    Z = ( (0.50*HarmonicK*rcut*rcut-0.50*HarmonicK*((X-traps_pos[i,0])**2 + (Y-traps_pos[i,1])**2))\
        *(((X-traps_pos[i,0])**2 + (Y-traps_pos[i,1])**2) < rcut*rcut) )
    i = i+1
    while i<sz[0]:#sz[0]
        Zi = (0.50*HarmonicK*rcut*rcut-0.50*HarmonicK*((X-traps_pos[i,0])**2 + (Y-traps_pos[i,1])**2))\
            *(((X-traps_pos[i,0])**2 + (Y-traps_pos[i,1])**2) < rcut*rcut)
        Z = Z + Zi
        i = i+1
    #plt.figure()
    # plot
    #steps_coarse = (steps/100.0).astype(int)
    #axis_value = np.linspace(min, max,steps_coarse )
    #plt.imshow(Z,cmap="plasma",origin="lower",extent= axis_value)
    #ax = plt.axes(projection='3d')
    fig,ax = plt.subplots()
    ax.pcolormesh(X, Y, Z,cmap=cmap_name,zorder = 1,alpha=transparency)
    #ax.plot_surface(X,Y,Z,cmap="plasma")
    #fig.colorbar(, ax=ax)

    #plt.xticks(np.arange(0,steps.astype(int)*100,1),np.linspace(min, max, steps.astype(int)*100))
    #plt.yticks(np.arange(0,steps.astype(int)*100,1),np.linspace(min, max, steps.astype(int)*100))
    #print(num_list)
    #plt.figure()
    """
    for i in num_list:
        LCRmin=i-0.005
        LCRmax=i+0.005
        condition=' where LinearCompressionRatio > '+str(LCRmin)+' && LinearCompressionRatio < '+str(LCRmax)
        data=osql.getDataFromMysql(table_name='melt_hex_from_honeycomb',search_condition=condition)
        data=np.array(data)
        #plot T vs Psi6, LCR as legend
        plt.scatter(data[:,1],data[:,4])# K VS LCR, Psi3/Psi6 as value
        #plt.show()
    """
    particles = True#False
    depin = False#True#
    new2 = False#True
    if particles:
        #points
        filename = trap_prefix + "testhex3-16-8"#"testhoneycomb3-6-12"#"testhex3-16-8"
        if new2:
            pp = sg.sequence()
            pp.generate_hex(a=3,n=[16,8])#pp.generate_hex(a=3,n=[16,8])
            pp.save(filename)
            #filename of simulated results, containing n rows of [x,y,0]
        data=np.loadtxt(filename)

        if depin:
            data=np.dot(linear_compression_ratio,data)#init depin
        ax.scatter(data[:,0], data[:,1],color='k',zorder = 2,s=particle_size)#s=size
    ax.set_aspect('equal','box')#plt.axis('equal')
    #plt.xlim((-30,30))
    #plt.ylim((-30,30))
    #plt.title(result_index+parameter)
    #plt.axes('equal')
    plt.show()

def workflow_mysql_to_data_search_a_point(tb_name,lcrset,lcr_step,kset,k_step):
    R"""
    import getDataAndScatter as scatt
    str = scatt.search_a_point('pin_hex_to_honeycomb_rectangle1',0.76,0.01,700,100)
    print(str)
    """
    import opertateOnMysql as osql
    cond1=' where HarmonicK >'+str(kset-0.5*k_step)+' and HarmonicK <'+str(kset+0.5*k_step)
    cond2=' and LinearCompressionRatio > '+str(lcrset-lcr_step*0.5)+' and LinearCompressionRatio <'+str(lcrset+lcr_step*0.5)
    search_condition=cond1+cond2
    data=osql.getDataFromMysql(table_name=tb_name,search_condition=search_condition)
    data=np.array(data)
    m3 = np.mean(data[:,3])
    return m3
    
def merge_diagram():
    R"""
    merge two parts of data_sets into one data_set,
    and then draw a large diagram.

    file_prefix = 
    file_name1 = 
    file_name2 = 
    data1 = np.loadtxt(file_name1)
    data2 = np.loadtxt(file_name2)

    plt.xxx
    """

R"""
    Introduction:
        This module is written to ensure that our simulation can be proceeded without any fault.
    It has two parts: pre-check and post-check.

    List of functions:
        pre-check:
        post-check:
             checkIfBalance

"""

R"""
Pre-check
"""
def draw_points_and_traps_preview(draw_points=True,show=True,account='tplab'):
    R"""
    Introduction:
        draw simulated configurations using blue dots and traps using red crosses.
    Examples:
        "testhex3-16-8"
        "testhoneycomb3-8-12"
        "testkagome3-9-6" for 144 particles in hex lattice
        "testcairo3-5-5"*0.681~"hexhex3-16-8"
        "testcairo3-6-6"*0.600~"hexhex3-16-8"
        "testkagome3-11-6"
    import getDataAndScatter as scatt
    scatt.draw_points_and_traps_preview(draw_points=True,show=True,account='remote')
    
    """    
    import hoomd
    import hoomd.azplugins
    import numpy
    from matplotlib import pyplot
    import hoomd.azplugins.sequence_generator as sg
    
    LinearCompressionRatio=0.8#LinearCompressionRatio
    HarmonicK=100#HarmonicK
    result_index = "x"#simu_index
    parameter="lcr"+str(LinearCompressionRatio)+"~k"+str(HarmonicK)
    #filename of traps
    trap_prefix='/home/'+account+'/hoomd-examples_0/'
    trap_filename=trap_prefix+"testkagome3-11-6"#data of trap position

    #generate a trap sequence
    gen = True
    if gen :
        pp = sg.sequence()
        pp.generate_kagome(a=3,n=[11,6])
        pp.save(trap_filename)

    data=numpy.loadtxt(trap_filename)
    pyplot.figure()
    pyplot.scatter(data[:,0]*LinearCompressionRatio, 
                   data[:,1]*LinearCompressionRatio,
                   c='r',marker = 'x')
    pyplot.axis('equal')

    if draw_points:
        #points
        filename = trap_prefix + "testhex3-16-8"#"testhex3-16-8"##
        """
        pp = sg.sequence()
        pp.generate_kagome(a=3,n=[16,8])
        pp.save(filename)
        
        """
        
        #filename of simulated results, containing n rows of [x,y,0]

        data=numpy.loadtxt(filename)
        pyplot.scatter(data[:,0], data[:,1],marker = '.')


    pyplot.title(result_index+parameter)
    pyplot.axis('equal')
    #pyplot.xlim([-20,20])
    #pyplot.ylim([-16,16])
    pyplot.xlabel('x')
    pyplot.ylabel('y')
    if show:
        pyplot.show()
    else:
        result_prefix='/home/'+account+'/Downloads/'
        title_set="kagome3-11-6&hex3-16-8"
        png_filename=result_prefix+parameter+title_set+'.png'
        pyplot.savefig(png_filename)
        print(png_filename)
    pyplot.close()

R"""
Post-check
"""
def check_if_balance(index):
    R"""
    introduction:
        read the log file to draw step vs potential energy image, 
        and step vs temperature image to check if the system 
        has achieved thermaldynamic balance.
    """
    str_index=str(int(index))
    log_prefix='/home/tplab/hoomd-examples_0/'
    file_log=log_prefix+'log-output_auto'+str_index+'.log'
    import numpy
    from matplotlib import pyplot
    #%matplotlib inline
    data = numpy.genfromtxt(fname=file_log, skip_header=True);

    pyplot.figure(figsize=(4,2.2), dpi=140);
    pyplot.plot(data[:,0], data[:,1]);
    pyplot.xlabel('time step');
    pyplot.ylabel('potential_energy');
    pyplot.show()

    pyplot.figure(figsize=(4,2.2), dpi=140);
    pyplot.plot(data[:,0], data[:,2]);
    pyplot.xlabel('time step');
    pyplot.ylabel('temperature');
    pyplot.show()

def check_if_balance_plot(index):#[x]
    R"""
    introduction:
        read the log file to draw step vs potential energy image, 
        and step vs temperature image to check if the system 
        has achieved thermaldynamic balance.
    """
    import numpy
    from matplotlib import pyplot

    str_index=str(int(index))
    log_prefix='/home/tplab/hoomd-examples_0/'
    file_log=log_prefix+'log-output_auto'+str_index+'.log'
    
    data = numpy.genfromtxt(fname=file_log, skip_header=True);

    num_steps = numpy.shape(data)[0]
    len_slice = int(num_steps/10)
    record = numpy.zeros((10,4))# 10 rows of [step,mean,std]
    
    i=0
    while i<10:
        ist = 0 + i*len_slice #index_start
        ied = len_slice + i*len_slice #index_end
        data_slice = data[ist:ied,2]
        record[i,0] = data[ied-1,0]
        record[i,1] = numpy.mean(data_slice)
        record[i,2] = numpy.std(data_slice)
        record[i,3] = record[i,2]/record[i,1]
        #if i>0:
        #    record[i,3] = record[i,1]/record[i-1,1]
            #record[i,4] = record[i,2]/record[i-1,2]

        i=i+1

    pyplot.figure();
    pyplot.plot(record[:,0], record[:,3]);
    #pyplot.plot(record[:,0], record[:,4]);
    pyplot.xlabel('time step');
    pyplot.ylabel('fluctuation');#decrease of 'potential_energy' is really slow
    #pyplot.legend("mean","std")
    pyplot.show()

    pyplot.figure();
    pyplot.plot(data[:,0], data[:,2]);
    pyplot.xlabel('time step');
    pyplot.ylabel('temperature');
    pyplot.show()
    """
    show =False
    if show:
        pyplot.figure(figsize=(4,2.2), dpi=140);
        pyplot.plot(data[:,0], data[:,1]);
        pyplot.xlabel('time step');
        pyplot.ylabel('potential_energy');
        pyplot.show()

        pyplot.figure(figsize=(4,2.2), dpi=140);
        pyplot.plot(data[:,0], data[:,2]);
        pyplot.xlabel('time step');
        pyplot.ylabel('temperature');
        pyplot.show() 
    """                                                      
    
def check_if_balance_deciaml(index):#[x]
    R"""
    introduction:
        read the log file to draw step vs Temperature image, 
        and step vs temperature image to check if the system 
        has achieved thermaldynamic balance.
    example:
        import check_simulation as cs
        i=1513#193
        benchmark = 0.1
        while i<2042:
            fluctuation = cs.check_if_balance_deciaml(i)
            if fluctuation > benchmark :
                msg1 = "the largest fluctuation is "+str(fluctuation)
                msg2 = ", from index "+str(i)
                print(msg1+msg2)
                benchmark = fluctuation
                cs.check_if_balance_plot(i)
            i=i+1
    """
    import numpy

    str_index=str(int(index))
    log_prefix='/home/tplab/hoomd-examples_0/'
    file_log=log_prefix+'log-output_auto'+str_index+'.log'
    
    data = numpy.genfromtxt(fname=file_log, skip_header=True);

    num_steps = numpy.shape(data)[0]
    len_slice = int(num_steps/10)
    record = numpy.zeros((10,4))# 10 rows of [step,mean,std]
    
    i=9
    ist = 0 + i*len_slice #index_start
    ied = len_slice + i*len_slice #index_end
    data_slice = data[ist:ied,2]
    record[i,0] = data[ied-1,0]
    m = numpy.mean(data_slice)
    std = numpy.std(data_slice)
    record = std/m

    #if record > 0.1:
        #print("Too few steps to let the system achieve balance!")

    return record # record = std/mean, that means the fluctuation of temperature.

def merge_arrays(save_file_name1,save_file_name2,save_file_name_result):
    #prefix='/home/tplab/Downloads/honeycomb/pin_hex_to_honeycomb_part1_random/'
    #'/home/tplab/Downloads/pin_hex_to_honeycomb_part1_random/'
    #save_file_name=prefix+"honeycomb_part1_diagram_all"
    #save_file_name2=prefix+"honeycomb_part1_diagram_c"
    data = np.loadtxt(save_file_name1)
    data2 = np.loadtxt(save_file_name2)
    data = np.append(data,data2,axis=0)
    np.savetxt(save_file_name_result,data)

def get_sk():
    R"""
    Standard points:
        prefix = /home/tplab/hoomd-examples_0/
        
        testcairo3-6-6
        testkagome3-9-6
        testhex3-16-8

    """
    prefix = '/home/tplab/hoomd-examples_0/'
    name = 'testhex3-16-8'
    filename = prefix + name
    data = np.loadtxt(filename)
    data = data[:,0:2]
    res = np.fft.fft2(data)
    print(res)
    raw = False
    if raw:
        plt.figure()
        plt.scatter(data[:,0],data[:,1])
        plt.axis('equal')
        plt.show()

    ed = True
    if ed:
        plt.figure()
        plt.scatter(res[:,0],res[:,1])
        plt.axis('equal')
        plt.show()


def snap_molecule_indices(snap):
    """Find molecule index for each particle.

    Given a snapshot from a trajectory, compute clusters of bonded molecules
    and return an array of the molecule index of each particle.

    Parameters
    ----------
    snap : gsd.hoomd.Snapshot
        Trajectory snapshot.

    Returns
    -------
    numpy array (N_particles,)

    """
    system = freud.AABBQuery.from_system(snap)
    num_query_points = num_points = snap.particles.N
    query_point_indices = snap.bonds.group[:, 0]
    point_indices = snap.bonds.group[:, 1]
    distances = system.box.compute_distances(
        system.points[query_point_indices], system.points[point_indices]
    )
    nlist = freud.NeighborList.from_arrays(
        num_query_points, num_points, query_point_indices, point_indices, distances
    )
    cluster = freud.cluster.Cluster()
    cluster.compute(system=system, neighbors=nlist)
    return cluster.cluster_idx



def get_neighbour_cloud():
    R"""
    Example:
        import getDataAndScatter as scatt
        scatt.get_neighbour_cloud()
    """
    #prefix='/home/tplab/Downloads/'
    #filename = prefix +'index2572'#index2572~cairo
    prefix="/home/tplab/hoomd-examples_0/"
    filename = prefix +'testhex3-16-8'#index2572~cairo
    points = np.loadtxt(filename)

    plt.figure()
    plt.scatter(points[:,0],points[:,1]) 
    plt.show()

    import freud
    box=np.zeros(2)
    x1=min(points[:,0])
    x2=max(points[:,0])
    Lx=x2-x1#get box size in x-direction
    y1=min(points[:,1])
    y2=max(points[:,1])
    Ly=y2-y1#get box size in y-direction
    box[0]=Lx+1
    box[1]=Ly+1
    #place=np.where([:]<)
    #sp=np.shape(points)
    #pts=np.zeros((sp[0],sp[1]-1))
    #pts[:,0:2]=points[:,0:2]
    nb = freud.locality.AABBQuery(box,points)
    #NeighborList()
    nlist = nb.query(points, {'r_max': 4}).toNeighborList()
    #freud.locality.NeighborQuery()

    # Get all vectors from central particles to their neighbors
    rijs = (points[nlist.point_indices] - points[nlist.query_point_indices])
    #rijs = box.wrap(rijs)
    plt.figure()
    plt.scatter(rijs[:,0],rijs[:,1]) 
    a = 3
    lim = a*1.5
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    plt.show()

def get_neighbour_cloud2():
    import points_analysis_2D_freud as pa
    prefix="/home/tplab/hoomd-examples_0/"
    filename = prefix +'testhex3-16-8'#index2572~cairo
    points = np.loadtxt(filename)
    set = pa.static_points_analysis_2d(points = points[:,0:2])
    print(set.delaunay.neighbors)#why only triangles are listed?
    #set.delaunay.vertex_neighbor_vertices

def test_get_neighbor_cloud_bond_minima(index=1369):
    import points_analysis_2D_freud
    prefix='/home/tplab/Downloads/'
    data_filename=prefix+'index'+str(index)
    a_frame = points_analysis_2D_freud.static_points_analysis_2d(filename=data_filename,hide_figure=False)
    #a_frame.draw_bond_length_distribution_and_first_minima()
    png_filename = prefix +'neighbor_cloud_'+'index'+str(index)+'.png'
    a_frame.get_neighbor_cloud_method_1st_minima_bond(png_filename=png_filename)

def get_dual_lattice():
    R"""
    plot voronoi vertex over points.
    
    hex <-> honeycomb
    honeycomb_part(hex) -> honeycomb(interstitial) 
    kagome -> hex
    kagome_part(rect) -> rect(interstitial)
    
    example:
    import getDataAndScatter as scatt
    scatt.get_dual_lattice()
    """
    point_prefix = "/home/remote/hoomd-examples_0/"
    filename = point_prefix + "testhoneycomb3-8-12-part1"
    filename2 = point_prefix + "testhoneycomb3-8-12"
    #"testkagome_part3-11-6" "testhoneycomb3-8-12-part1"
    #"testhoneycomb3-8-12" "testkagome3-9-6" "testhex3-16-8"
    points = np.loadtxt(filename )
    points2 = np.loadtxt(filename2 )
    test = static_points_analysis_2d(points,hide_figure=False)
    
    pt = test.voronoi.vertices
    plt.figure()
    plt.scatter(test.points[:,0],test.points[:,1],c='k')#points
    plt.scatter(pt[:,0],pt[:,1],c='r')#voronoi vertices
    plt.scatter(points2[:,0],points2[:,1],c='g')#points
    plt.axis('equal')
    plt.xlim([-30,30])
    plt.ylim([-30,30])
    plt.show()

def get_xy_gr_sk(simu_index,seed,xy=False,gr=False,sk=False):
    R"""
    Example:
    import getDataAndScatter as scatt
    index=3059
    seed=9
    scatt.get_xy_gr_sk(index,seed,xy=True,gr=True,sk=True)
    """
    import freud
    import points_analysis_2D_freud
    #get data
    gsd_prefix = "/home/tplab/hoomd-examples_0/trajectory_auto"
    gsd_postfix = ".gsd"
    str_index=str(int(simu_index))+'_'+str(int(seed))
    file_gsd=gsd_prefix+str_index+gsd_postfix
    gsd_data = points_analysis_2D_freud.proceed_gsd_file(filename_gsd=file_gsd)
    snap = gsd_data.trajectory[-1]
    #filename
    prefix='/home/tplab/Downloads/'
    
    matplotlib.use(backend="agg")#Backend agg is non-interactive backend. Turning interactive mode off.
        #Backend Qt5Agg is interactive backend. Turning interactive mode on.

    #calculation
    if xy:
        positions = snap.particles.position
        plt.scatter(positions[:,0],positions[:,1])
        plt.axis('equal')
        plt.xlabel('x(sigma)')
        plt.ylabel('y(sigma)')
        fig_type = 'xy'
        data_filename=prefix+'index_'+str_index+fig_type+'.png'
        fig_type1 = 'pos'
        txt_filename=prefix+'index_'+str_index+fig_type1+'.txt'
        plt.savefig(data_filename)
        plt.close()
        np.savetxt(txt_filename,positions)
    if gr:
        rdf = freud.density.RDF(bins=200, r_max=20.0)#
        rdf.compute(system=snap)
        print(rdf.bin_centers)
        print(rdf.bin_counts)
        rdf.plot()
        fig_type = 'gr'
        data_filename=prefix+'index_'+str_index+fig_type+'.png'
        plt.savefig(data_filename)
        plt.close()
    
    if sk:
        sk = freud.diffraction.DiffractionPattern()
        sk.compute(system=snap)
        #fig,ax = plt.subplots()#figsize=(4,4),dpi=150
        sk.plot()
        fig_type = 'sk'
        data_filename=prefix+'index_'+str_index+fig_type+'.png'
        plt.savefig(data_filename)
        plt.close()

def lattice_to_xy_gr_sk(xy=True,gr=True,sk=True):
    R"""
    structure = 'hex3-16-8'
    "testkagome_part3-11-6" "testhoneycomb3-8-12-part1"
    "testhoneycomb3-8-12" "testkagome3-9-6" "testhex3-16-8"
    
    Example:
    import getDataAndScatter as scatt
    scatt.lattice_to_xy_gr_sk()
    """
    import hoomd
    import freud
    #import hoomd.azplugins.sequence_generator as sg
    #hoomd.context.initialize("");#--mode=cpu
    
    #set parameters
    import gsd.hoomd
    structure = 'kagome_depin0'+'_'
    filename_gsd='/home/tplab/hoomd-examples_0/trajectory_auto1583.gsd'
    trajectory=gsd.hoomd.open(filename_gsd)
    snap = trajectory.read_frame(-1)

    """
    structure = 'hex3-16-8'+'_'
    sys=hoomd.init.create_lattice(unitcell=hoomd.lattice.hex(a=3), n=[16,8]);
    
    structure = 'sq3-16-16'+'_'
    sys=hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=3), n=[16,16]);

    seq=sg.sequence()
    structure = 'kagome3-9-6'+'_'
    seq.generate_kagome(a=3,n=[9,6])
    sys=seq.system

    seq=sg.sequence()
    structure = 'honeycomb3-8-12'+'_'
    seq.generate_honeycomb(a=3,n=[8,12])
    sys=seq.system
    """
    #snap = sys.take_snapshot()
    #snap = trajectory[-1]

    #filename
    prefix='/home/tplab/Downloads/lattice_show/'
    
    matplotlib.use(backend="agg")#Backend agg is non-interactive backend. Turning interactive mode off.
        #Backend Qt5Agg is interactive backend. Turning interactive mode on.

    #calculation
    if xy:
        positions = snap.particles.position
        plt.scatter(positions[:,0],positions[:,1])
        plt.axis('equal')
        plt.xlabel('x(sigma)')
        plt.ylabel('y(sigma)')
        fig_type = 'xy'
        data_filename=prefix+structure+fig_type+'.png'
        fig_type1 = 'pos'
        txt_filename=prefix+structure+fig_type1+'.txt'
        plt.savefig(data_filename)
        plt.close()
        np.savetxt(txt_filename,positions)
    if gr:
        rdf = freud.density.RDF(bins=150, r_max=15.0)#
        rdf.compute(system=snap)
        #print(rdf.bin_centers) print(rdf.bin_counts)
        rdf.plot()
        fig_type = 'gr'
        data_filename=prefix+structure+fig_type+'.png'
        plt.savefig(data_filename)
        plt.close()
    
    if sk:
        sk = freud.diffraction.DiffractionPattern()
        sk.compute(system=snap)
        fig_type = 'sk'        
        data_filename=prefix+structure+fig_type+'.png'
        fig,ax = plt.subplots()
        im = ax.pcolormesh(sk.k_values,sk.k_values,sk.diffraction,cmap='afmhot')#im = 
        #ax.colorbar().remove()
        fig.colorbar(im)
        ax.axis('equal')
        """
        #method1
        ax = sk.plot()
        ax.pcolormesh(sk.k_values,sk.k_values,sk.diffraction,cmap='afmhot')
        """
        plt.savefig(data_filename)
        plt.close()
        #ax.pcolormesh(X, Y, Z,cmap="plasma",)
        """
        maybe that the loglog map is not suitable for my sk.
        linear colorbar is the right choice
        """