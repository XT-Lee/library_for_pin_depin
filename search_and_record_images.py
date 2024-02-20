import time
import os
import numpy as np
import particle_tracking as pt
import points_analysis_2D_freud as pa

R"""
parameters:
    dir_path: the directory to store images
    silent: if true, not to print time and status every second.

mode list:
    <time controller mode>:1, record the runtime of the script. if it costs too long, kill the process. 2, wait 1 sec per loop.

exp:
    import search_and_record_images as sri
    sri.script(dir_path='/home/remote/xiaotian_file/20210417')

plan:
    use file (type:dict())to compare and get new file 
    use persitence homology to calculate cluster.
"""

def script(dir_path = None,silent=False):
    #<time controller mode>
    tm1=time.localtime(time.time())
    #Warning: the script must be operated in debug mode in case running infinite time!
     #<time controller mode>
    while True:
        #<detect if new images are created>
        if 'count' in locals():
            is_new_image_existing,count,filename=detect_new_image(dir_path=dir_path, file_count_old=count)
        else:
            is_new_image_existing,count,filename=detect_new_image(dir_path=dir_path)
        # <detect if new images are created>

        if is_new_image_existing:
           #particle positions
            print(filename)
            frame=pt.particle_track()
            frame.single_frame_particle_tracking(filename,19,1800)#parameters remain undefined
            points=frame.xy
            points[:]=points[:]*3/32# transform unit from pixel to um
            """
            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(points[:,0],points[:,1])
            plt.show()
            """
            #particle density, averaged bond length
            result = pa.static_points_analysis_2d(points)
            if silent:
                png_filename1= None
                png_filename2 = None
            else:
                png_filename1= filename +'_bond_hist.png'
                png_filename2 = filename +'_bond_plot_1st_minima.png'
            
            lc = 2.0
            result.get_first_minima_bond_length_distribution(lattice_constant=lc,png_filename=png_filename1,hist_cutoff=5)#here lattice_constant is just used to normalize figure, hence set 2.0 is ok
            print('recognized bond length: '+str(result.bond_length_mean*lc)+'+-'+str(result.bond_length_std*lc)+' um')
            result.draw_bonds_conditional_bond(check=[2.0, result.bond_first_minima_left], png_filename=png_filename2)

            """
            #particle positions
            points=np.random((100,2))

            #particle density, averaged bond length
            result = pa.PointsAnalysis2D(points)
            result.get_first_minima_bond_length_distribution(lattice_constant=1.0)#here lattice_constant is just used to normalize figure, hence set 1.0 is ok
            print(result.bond_first_minima_left)

            """
            
        
        #<time controller mode>
        tm2=time.localtime(time.time())
        #calculate the time cost. If the runtime is too long, kill the loop. 
        dt_h=tm2.tm_hour-tm1.tm_hour
        if dt_h<0:
            dt_h = dt_h + 24
        if not silent:
            print(str(dt_h)+' hours passed')
        if dt_h >= 12.0 :
            print('loop stop: '+str(dt_h)+' hours passed')
            break
        #a cycle run one time per 1 second
        time.sleep(1)
        # <time controller mode>
        
    #end while

def check_particle_and_illuminiation(filename,D,minmass):
    R"""
    the particle size in pixel and the illumination distribution would vary in different experiments,
    so users have to check and input the most suitable parameters.
    """
    frame=pt.particle_track()
    frame.single_frame_particle_tracking(filename,D,minmass)

def detect_new_image(dir_path = None,postfix='.jpg',file_count_old=0):
    R"""
    Instruction:
        https://www.geeksforgeeks.org/file-searching-using-python/
    """
    # This is to get the directory that the program
    # is currently running in.
    #dir_path = '/home/tplab/Downloads/honeycomb/pin_hex_to_honeycomb_random/old'#os.path.dirname(os.path.realpath(__file__))
    file_count_new=0
    for root, dirs, files in os.walk(dir_path):#for root, dirs, files in os.walk(dir_path):
        for file in files:
            # change the extension from '.jpg' to
            # the one of your choice.
            if file.endswith(postfix):
                #print (str(file))#print (root+'/'+str(file))
                file_count_new = file_count_new+1
    #print(str(file_count_new)+' files found')
    
    is_new_image_existing = False
    if file_count_new == file_count_old+1:
        is_new_image_existing = True
        filename = root+'/'+'DefaultImage_'+str(file_count_new-1)+postfix
    else :
        filename = None
    if file_count_new == 1:
        filename = root+'/'+'DefaultImage'+postfix

    return is_new_image_existing,file_count_new,filename

def killing_loop_mode():
    #dt_s=tm2.tm_sec-tm1.tm_sec
    """
    #calculate the time cost. If the runtime is too long, kill the loop. 
    dt_h=tm2.tm_hour-tm1.tm_hour
    if dt_h > 1.0 :
        break
    """
