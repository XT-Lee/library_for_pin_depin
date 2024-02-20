
def workflow_simu_to_mysql_pin_hex_to_honeycomb(index1,lcr,seed=9):#num
    R"""
    INTRODUCTION:
        OLD_NAME:workflow_simu_to_mysql_pin_hex_to_honeycomb_repeat
    Parameters:
        initial state: hex3-16-8
        traps:honeycomb3-8-12
        safe lcr: [0.71,1.00]
    Examples1:
        #scan seed
        seed=0
        while seed<9.5:
            #pin sequence-GPU
            index1=2043
            lcr1=0.74#less than 0.71 is dangerous! some particles may not effected by trap!
            while lcr1<0.845:
                tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_repeat(index1=index1,lcr=lcr1,seed=seed)
                index1=index1+10
                lcr1=lcr1+0.01
            seed+=1
    Examples2:
        #scan seed
        seed=0
        while seed<9.5:
            #pin sequence-GPU
            index1=2363
            lcr1=0.76#less than 0.71 is dangerous! some particles may not effected by trap!
            while lcr1<0.805:
                end_index=tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_repeat(index1=index1,lcr=lcr1,seed=seed)
                index1=end_index+1#index1=index1+10
                lcr1=lcr1+0.01
            seed+=1
    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testhoneycomb3-8-12"lcr=0.71
    #particle:"testhex3-16-8"
    #"check_init_pin_hex_to_honeycomb.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    k1=0.0
    stp=10.0
    kend=90.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_honeycomb_pin as sa_k
    end_index=sa_k.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    'get file index123'

    #step3
    #get analyzed data
    import data_analysis_cycle as da
    filename_klp=da.saveIndexPsi3Psi6R(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_klp)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_honeycomb_repeat'
        simu_index | HarmonicK | LinearCompressionRatio | Psi3Global | Psi6Global | RandomSeed
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql(path_to_file_name=filename_klp,table_name="pin_hex_to_honeycomb_repeat")#"/home/tplab/Downloads/193-205kl"

    return end_index

def workflow_simu_to_mysql_pin_hex_to_honeycomb_oop(account,index1,lcr,seed=9):
    R"""
    INTRODUCTION:

    Parameters:
        initial state: hex3-16-8
        traps:honeycomb3-8-12
        safe lcr: [0.71,1.00]
    Examples1:
        #scan seed
        seed=0
        while seed<10:
            #pin sequence-GPU
            index1=2043
            lcr1=0.74#less than 0.71 is dangerous! some particles may not effected by trap!
            while lcr1<0.845:
                end_index = tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_oop(index1=index1,lcr=lcr1,seed=seed)
                index1=end_index+1
                lcr1=lcr1+0.01
            seed+=1
    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testhoneycomb3-32-48-part1"lcr=0.71
    #particle:"testhex3-64-32"
    #"4-times-size.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    trap_name = "testhoneycomb3-32-48-part1"
    #get simulation results
    import symmetry_transformation_v2_9.pin_seed_oop as pin
    """
    wk = pin.workflow_pin(index1,"remote",k1,stp,kend,lcr,seed,trap_name)
    wk.set_init_state_pin(a=3,nx=64,ny=32)
    end_index = wk.workflow()
    
    """
    end_index =3565
    #end_index=sa_k.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    'get file index123'

    #step3
    #get analyzed data
    import data_analysis_cycle as da
    filename_klp=da.saveIndexPsi3Psi6R(account=account,start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)   #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_klp)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_honeycomb_edge_cut'
        simu_index | HarmonicK | LinearCompressionRatio | Psi3Global | Psi6Global | RandomSeed
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql(path_to_file_name=filename_klp,table_name="pin_hex_to_honeycomb_edge_cut")#"/home/tplab/Downloads/193-205kl"

    #step5
    #watch kT limitation while melting
    import data_analysis_cycle as da
    i=index1
    while i<3566:
        da.save_from_gsd(simu_index=i,seed=seed,final_cut=True,
                                bond_plot =True,
                                show_traps=False,
                                trap_filename="/home/"+account+"/hoomd-examples_0/testhoneycomb3-8-12-part1",
                                trap_lcr=0.79,
                                account=account)
        i+=1

    return end_index
 
def workflow_simu_to_mysql_pin_hex_to_honeycomb_oop_klt_2m(index1,lcr,kT=1.0,seed=9,account='tplab',check=True):
    R"""
    INTRODUCTION:

    Parameters:
        initial state: hex3-16-8
        traps:honeycomb3-8-12
        safe lcr: [0.71,1.00]
    Examples1:
        index1=4686
        lcr1=0.816#less than 0.71 is dangerous! some particles may not effected by trap!
        tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_oop_klt_2m(index1=index1,lcr=lcr1,account='remote')
    example2:
        #pin sequence-GPU
        index1=4576
        lcr1=0.74#less than 0.71 is dangerous! some particles may not effected by trap!
        while lcr1<0.845:
            end_index = tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_oop_klt_2m(index1=index1,lcr=lcr1,account='remote')
            index1=end_index+1
            lcr1=lcr1+0.01    
    exp3:
        #pin sequence-GPU
        import workflow_part as tt
        import numpy
        lcr_list = numpy.linspace(0.77,0.85,9)
        lcr_list[-1] = 0.816
        #print(lcr_list)
        index1=5209
        for lcr1 in lcr_list:
            end_index = tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_oop_klt_2m(index1=index1,lcr=lcr1,account='remote')
            #print(index1,end_index,lcr1)
            index1=end_index+1 
    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.draw_points_and_traps_preview(draw_points=True,show=False,account='remote')

    #traps:"testhoneycomb3-8-12"lcr=0.71
    #particle:"testhex3-16-8"
    #"lcr0.71~k100honeycomb3-8-12&hex3-16-8.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    k1=33.0
    stp=3.0
    kend=60.0
    trap_name = "testhoneycomb3-8-12"
    #get simulation results
    if not check:
        import symmetry_transformation_v2_9.pin_seed_oop as pin
        wk = pin.workflow_uniform(index1,account,k1,stp,kend,lcr,kT,seed,trap_name,mode="--mode=gpu")
        end_index = wk.workflow()
    else:
        end_index = index1 +9#

    #end_index =3565
    #end_index=sa_k.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    'get file index123'

    #step3
    #get analyzed data
    if not check:
        import data_analysis_cycle as da
        filename_klp=da.saveIndexklTPsi36Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)   	    	#filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
        print('\n'+filename_klp)
        'get file named index1 index2 kl'
    
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_honeycomb_klt_2m'
        |simu_index | HarmonicK | LinearCompressionRatio | kT |
      Psi3Global | Psi6Global | RandomSeed | 
    """
    if not check:
        import opertateOnMysql as osql
        osql.loadTxtDataToMysql(path_to_file_name=filename_klp,table_name="pin_hex_to_honeycomb_klt_2m")#"/home/tplab/Downloads/193-205kl"
        
    #step5
    #watch kT limitation while melting
    """
    import data_analysis_cycle as da
    i=index1
    while i<3566:
        da.save_from_gsd(simu_index=i,seed=seed,final_cut=True,
                                bond_plot =True,
                                show_traps=False,
                                trap_filename="/home/"+account+"/hoomd-examples_0/testhoneycomb3-8-12",
                                trap_lcr=0.79,
                                account=account)
        i+=1
    """
    
    return end_index 

def workflow_simu_to_mysql_pin_hex_to_honeycomb_rectangle1(index1,lcr,seed=9):#num
    R"""
    Parameters:
        initial state: hex3-16-8 (12-6 in fact!)
        traps:honeycomb3-8-12_rectangle1
        safe lcr: [0.71,1.00]
    Examples:
        #scan seed
        seed=0
        while seed<9:
            #pin sequence-GPU
            index1=1843
            lcr1=0.71#less than 0.71 is dangerous! some particles may not effected by trap!
            while lcr1<0.905:
                tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_rectangle1(index1=index1,lcr=lcr1,seed=seed)
                index1=index1+10
                lcr1=lcr1+0.01
            seed+=1
    
    Example2:
        #scan seed
        seed=0
        while seed<9.5:
            #pin sequence-CPU
            index1= 2463
            lcr1=0.71#less than 0.71 is dangerous! some particles may not effected by trap!
            while lcr1<0.805:
                end_index = tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_rectangle1(index1=index1,lcr=lcr1,seed=seed)
                index1=end_index+1
                lcr1=lcr1+0.01
            seed+=1
    """
    #set parameters
    k1=1100.0
    stp=100.0
    kend=2000.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_honeycomb_rectangle1_pin as sa_hbp 
    end_index=sa_hbp.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    'get file index123'

    #get analyzed data
    import data_analysis_cycle as da
    filename_klp=da.saveIndexPsi3Psi6R(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_klp)
    'get file named index1 index2 klp'

    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_honeycomb_rectangle1'
        simu_index | HarmonicK | LinearCompressionRatio | Psi3Global | Psi6Global | RandomSeed
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_klp,
    table_name='pin_hex_to_honeycomb_rectangle1')#"/home/tplab/Downloads/193-205kl"

    return end_index

def workflow_simu_to_mysql_depin_from_honeycomb_part(index1,lcr):
    R"""
    caution: the temperature kT=0.1 !!!
    """
    #set parameters
    
    k1=0.0
    stp=10.0
    kend=90.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_honeycomb_bidispersion_depin as sa_hb
    end_index=index1+9#end_index=sa_hb.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr)
    'get file index123'

    #get analyzed data
    import data_analysis_cycle as da
    filename_klp=da.saveIndexPsi3Psi6(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    #print('\n'+filename_klp)
    'get file named index1 index2 klp'

    #loadDataToMysql
    R"""
    Note: the format of table_name='depin_from_honeycomb_part1_kt_01'
        simu_index | HarmonicK | LinearCompressionRatio | Psi3Global | Psi3Ratio
    """
    '''
    prefix='/home/tplab/Downloads/'
    start_index=index1
    end_index=index1+30
    filename_kl=prefix+str(start_index)+'-'+str(end_index)+'kl'
    '''
    
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql(path_to_file_name=filename_klp,table_name='depin_from_honeycomb_part1')#'depin_from_honeycomb_part1_kt_01'#"/home/tplab/Downloads/193-205kl"

def workflow_simu_to_mysql_pin_hex_to_honeycomb_part(index1,lcr,seed):
    R"""
    Examples1:
        #scan seed
        seed=0
        while seed<9:
            #pin sequence-GPU
            index1=1343
            lcr1=0.77#less than 0.75 is dangerous! some particles may not effected by trap!
            while lcr1<0.845:
                tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_part(index1=index1,lcr=lcr1,seed=seed)
                index1=index1+10
                lcr1=lcr1+0.01
            seed+=1
    
    Examples2:
        #scan seed
        seed=0
        while seed<9.5:
            #pin sequence-GPU
            index1=2413
            lcr1=0.78#less than 0.75 is dangerous! some particles may not effected by trap!
            while lcr1<0.825:
                end_index=tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_part(index1=index1,lcr=lcr1,seed=seed)
                index1=end_index+1
                lcr1=lcr1+0.01
            seed+=1
    Examples3:
        seed=0
        while seed<9:
            #pin sequence-GPU
            index1=3466
            lcr1=0.81650#less than 0.75 is dangerous! some particles may not effected by trap!
            tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_part(index1=index1,lcr=lcr1,seed=seed)
            seed=seed+1
    """
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_honeycomb_bidispersion_pin as sa_hbp
    end_index=sa_hbp.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    #end_index=index1+9
    'get file index123'

    #get analyzed data
    import data_analysis_cycle as da
    filename_klp=da.saveIndexPsi3Psi6R(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_klp)
    'get file named index1 index2 klp'

    #loadDataToMysql
    R"""
    Note: the format of table_name='hex_to_honeycomb_part1'
        simu_index | HarmonicK | LinearCompressionRatio | Psi3Global | Psi6Global | RandomSeed |
    """
    '''
    prefix='/home/tplab/Downloads/'
    start_index=index1
    end_index=index1+30
    filename_kl=prefix+str(start_index)+'-'+str(end_index)+'kl'
    '''
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql(path_to_file_name=filename_klp,table_name='pin_hex_to_honeycomb_part1')#"/home/tplab/Downloads/193-205kl"

    return end_index

def workflow_simu_to_mysql_depin_from_honeycomb_part_oop_kT(index1,lcr,kT=1.0,seed=9,account='tplab'):#[x]
    R"""
    INTRODUCTION:
    example4 :
        seed=9
        index1=4016
        lcr1=0.816
        kT=1.0
        while kT<10.1:
            index_end=tt.workflow_simu_to_mysql_depin_from_honeycomb_part_oop_kT(index1=index1,lcr=lcr1,kT=kT,seed=seed)
            print(index_end)
            print(kT)
            index1 = index_end + 1
            kT = kT + 1.0

    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testhoneycomb3-8-12"lcr=0.71
    #particle:"testhoneycomb3-8-12"
    """
    #step2
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    trap_name = "testhoneycomb3-8-12-part1"
    #get simulation results

    import symmetry_transformation_v2_9.pin_seed_oop as pin
    wk = pin.workflow_uniform(index1,account,k1,stp,kend,lcr,kT,seed,trap_name)
    wk.set_init_state_parameters(depin_from_honeycomb=True)
    end_index = wk.workflow()
    #end_index = index1 + 9
    'get file index123'

    #step3
    #get analyzed data

    import data_analysis_cycle as da
    filename_kl=da.saveIndexklTPsi36Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)
    print('\n'+filename_kl)
    'get file named index1 index2 klt'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='depin_from_honeycomb_part_klt'
        |simu_index|HarmonicK|LinearCompressionRatio|kT|
        Psi3Global|Psi6Global|RandomSeed|
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='depin_from_honeycomb_part_klt')
    
    
    #step5
    #watch kT limitation while cooling
    """
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,
        coordination_number=True,
        account=account)
        i+=1
    
    """
    
    return end_index

def workflow_simu_to_mysql_from_honeycomb_to_liquid(index1,lcr,kT=1.0,seed=9,account='tplab',check=False):#[x]
    R"""
    INTRODUCTION:
    example4 :
        import workflow_part as tt
        import numpy as np
        seed=9
        index1=5390
        lcr_list = np.linspace(0.82,1,10)
        kT=1.0
        for lcr1 in lcr_list:
            tt.workflow_simu_to_mysql_from_honeycomb_to_liquid(index1=index1,lcr=lcr1,kT=kT,seed=seed,account='remote')#,check=True
            print(index1,lcr1)
            index1 = index1 + 1
    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testhoneycomb3-8-12"lcr=0.71
    #particle:"testhoneycomb3-8-12"
    """
    #step2
    #set parameters
    #trap_name = "testhoneycomb3-8-12-part1"
    #get simulation results
    import symmetry_transformation_v2_9.simple_simulation as ss
    wk = ss.workflow_uniform(index1,account,lcr,kT,seed)
    wk.set_init_state_parameters(a=3,nx=8,ny=12,depin_from_honeycomb=True)
    #wk.__init_state_launch()
    if not check:#index1>5399:#
        pass#end_index = wk.workflow()
    else:
        end_index = index1
    #end_index = index1 + 9
    'get file index123'

    #step3
    #get analyzed data
    if not check:#False:#
        import data_analysis_cycle as da
        filename_kl=da.saveIndexklTPsi36Seed(start_index=index1,end_index=index1+9,k1=0,step=0,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)
        print('\n'+filename_kl)
        'get file named index1 index2 klt'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='depin_from_honeycomb_part_klt'
        |simu_index|HarmonicK|LinearCompressionRatio|kT|
        Psi3Global|Psi6Global|RandomSeed|
    """
    """
    import opertateOnMysql as osql
    osql.loadDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='depin_from_honeycomb_part_klt')
    """
    
    #step5
    #watch kT limitation while cooling
    """
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,
        coordination_number=True,
        account=account)
        i+=1
    
    """
    
    return end_index

def workflow_simu_to_mysql_pin_liquid_to_honeycomb_part(init_gsd_file,index1,lcr,kT=1.0,seed=9,account='tplab',check=False):#[x]
    R"""
    INTRODUCTION:
    example:
        import workflow_part as tt
        init_gsd_file='/home/remote/hoomd-examples_0/trajectory_auto5409_9.gsd'
        seed=9
        index1=5410
        lcr_list=list([0.79*7.44/3,0.8*7.44/3])
        #lcr_list=lcr_list
        kT=1.0
        for lcr1 in lcr_list:
            index_end=tt.workflow_simu_to_mysql_pin_liquid_to_honeycomb_part(init_gsd_file,index1=index1,lcr=lcr1,kT=kT,seed=seed,account='remote')#,check=True
            #print(index1,lcr1)
            index1 = index_end + 1
    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testhoneycomb3-8-12"lcr=0.71
    #particle:"testhoneycomb3-8-12"
    """
    #step2
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    trap_name = "testhoneycomb3-8-12-part1"
    #get simulation results

    import symmetry_transformation_v2_9.pin_seed_oop as pin
    wk = pin.workflow_uniform(index1,account,k1,stp,kend,lcr,kT,seed,trap_name,"--mode=gpu")
    wk.set_init_state_parameters(init_gsd=init_gsd_file)
    if not check:#index1>5399:#
        end_index = wk.workflow()
    else:
        end_index = index1 + 9
    'get file index123'

    #step3
    #get analyzed data
    if not check:#False:#
        import data_analysis_cycle as da
        filename_kl=da.saveIndexklTPsi36Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)
        print('\n'+filename_kl)
        'get file named index1 index2 klt'

    #step4
    #loadDataToMysql
    
    #step5
    #watch kT limitation while cooling
    
    return end_index

def workflow_simu_to_mysql_pin_hex_to_honeycomb_part_oop_klt_2m(index1,lcr,kT=1.0,seed=9,account='tplab',check=False):
    R"""
    INTRODUCTION:
    example4 :
        seed=9
        index1=3806
        lcr1=0.79
        kT=0.0
        while kT<1.01:
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_part_oop_kT(index1=index1,lcr=lcr1,kT=kT,seed=seed)
            print(index1)
            print(kT)
            index1 = index_end + 1
            kT = kT + 0.1
    exp_mega:
        index1=4271
        lcr1=0.79
        index_end=tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_part_oop_kT(index1=index1,lcr=lcr1,account='remote')
    exp_scan0.80-0.84:
        index1=4286
        lcr1=0.80
        while lcr1<0.845:
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_part_oop_kT(index1=index1,lcr=lcr1,account='remote')
            print(lcr1)
            print(index1)
            index1 = index_end+1
            lcr1 = lcr1+0.01
    exp_scan0.77-0.78:
        index1=4336
        lcr1=0.77
        while lcr1<0.785:
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_part_oop_kT(index1=index1,lcr=lcr1,account='remote')
            print(lcr1)
            print(index1)
            index1 = index_end+1
            lcr1 = lcr1+0.01
    exp low T:
        import workflow_part as tt
        seed=9
        index1=5028
        lcr1=0.77
        while lcr1<0.845:
            print(index1,lcr1,seed)
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_part_oop_klt_2m(index1=index1,lcr=lcr1,kT=0.1,seed=seed,account='remote')
            index1 = index_end+1#index1 = index1+10#
            lcr1 = lcr1+0.01
    exp precise:
        #pin sequence-GPU
        import workflow_part as tt
        import numpy
        lcr_list = numpy.linspace(0.77,0.85,9)
        lcr_list[-1] = 0.816
        #print(lcr_list)
        index1=5299
        for lcr1 in lcr_list:
            end_index = tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_part_oop_klt_2m(index1=index1,lcr=lcr1,account='remote')
            #print(index1,lcr1)
            index1=end_index+1 #index1=index1+10 #
    exp scan seed:
        #0.77-0.78
        import workflow_part as tt
        for seed in range(9):
            index1=4336
            lcr1=0.77
            while lcr1<0.785:
                index_end=tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_part_oop_klt_2m(index1=index1,lcr=lcr1,seed=seed,account='remote',check=True)
                print(lcr1,index1,seed)
                index1 = index_end+1
                lcr1 = lcr1+0.01
        #0.79-0.84,0.8165
        import numpy
        lcr_list = numpy.linspace(0.78,0.84,7)
        lcr_list[0] = 0.79
        lcr_list[1] = 0.8165
        #print(lcr_list)
        import workflow_part as tt
        for seed in range(9):
            index1=4266
            for lcr1 in lcr_list:
                index_end=tt.workflow_simu_to_mysql_pin_hex_to_honeycomb_part_oop_klt_2m(index1=index1,lcr=lcr1,seed=seed,
                                                                                        account='remote',check=True)
                print(index1,lcr1,seed)
                index1=index_end+1
    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testhoneycomb3-8-12"lcr=0.71
    #particle:"testhex3-16-8"
    #"check_init_pin_hex_to_honeycomb.png"

    size_par < size_trap checked right!
    """

    #step2
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    trap_name = "testhoneycomb3-8-12-part1"
    #get simulation results
    if not check:
        import symmetry_transformation_v2_9.pin_seed_oop as pin
        wk = pin.workflow_uniform(index1,account,k1,stp,kend,lcr,kT,seed,trap_name,mode="--mode=gpu")
        end_index = wk.workflow()#period=1000;steps=2e6+1
    else:
        end_index = index1 + 9
        'get file index123'

    #step3
    #get analyzed data
    if not check:
        import data_analysis_cycle as da
        filename_kl=da.saveIndexklTPsi36Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)
        print('\n'+filename_kl)
    
    'get file named index1 index2 klt'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_honeycomb_part_klt_2m'
        |SimuIndex|HarmonicK|LinearCompressionRatio|kT|
        Psi3|Psi6|RandomSeed|
    """
    if not check:
        import opertateOnMysql as osql
        osql.loadTxtDataToMysql\
        (path_to_file_name=filename_kl,
        table_name='pin_hex_to_honeycomb_part_klt_2m')
    
    
    
    #step5
    #watch kT limitation while cooling
    """
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,
        coordination_number=True,
        account=account)

        da.save_from_gsd(simu_index=i,seed=seed,final_cut=True,
                                    bond_plot =True,
                                    show_traps=True,
                                    trap_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12-part1",
                                    trap_lcr=lcr,
                                    account='remote')
        i+=1
    """
    
    
    """
    import data_analysis_cycle as da
    i=index1
    while i<4271:
        da.save_from_gsd(simu_index=i,seed=9,final_cut=True,
                                    bond_plot =True,
                                    show_traps=True,
                                    trap_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12-part1",
                                    trap_lcr=0.79,
                                    account='remote')
        i+=1
    
    """
    
    return end_index

def workflow_simu_to_mysql_melt_from_honeycomb_part(index1,lcr,seed=9):
    R"""
    EXAMPLE:
    seed=8
    index1=3421#1369
    lcr1=0.79
    tt.workflow_simu_to_mysql_melt_from_honeycomb_part(index1,lcr1,seed)
    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testhoneycomb3-8-12-part1"lcr=0.79
    #particle:"testhoneycomb3-6-12"lcr=0.79
    #"check_init_melt_from_honeycomb_part.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    kt1=1.0
    stp=1.0
    ktend=20.0
    kset=700
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_honeycomb_bidispersion_melt_seed as sa_h
    end_index=sa_h.workflow(index1=index1,kt1=kt1,step=stp,kt_end=ktend,linear_compression_ratio=lcr,kset=kset,seed_set=seed)
    'get file index123'
    #end_index=int(index1+19)
    
    #step3[x]
    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    filename_kl=da.saveIndexkTPsi36Seed(start_index=index1,end_index=end_index,kt1=kt1,step=stp,
                                        linear_compression_ratio=lcr,kset=kset,randomseed=seed)
    #saveIndexCN4CN6SeedPsi6Seed
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    #print('\n'+filename_kl)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='melt_from_honeycomb_part_test'
        | simu_index | HarmonicK | LinearCompressionRatio | kT |
        Psi3Global | Psi6Global | RandomSeed |
    example:
        create table melt_from_honeycomb_part_test(simu_index int unsigned not null,HarmonicK float, 
        LinearCompressionRatio float, kT float,Psi3Global float, Psi6Global float, RandomSeed int unsigned);
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='melt_from_honeycomb_part_test')

    #step5
    #watch kT limitation while melting
    import data_analysis_cycle as da
    i=3421
    while i<3441:
        da.save_from_gsd_seed(simu_index=i,seed=8,final_cut=True,
                                bond_plot =True,
                                show_traps=True,
                                trap_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12-part1",
                                trap_lcr=0.79)
        i+=1

def workflow_simu_to_mysql_heat_treat_from_honeycomb_part():
    R"""
    Example:
        tt.workflow_simu_to_mysql_heat_treat_from_honeycomb_part()
    """
    #init
    init_gsd = '/home/tplab/hoomd-examples_0/trajectory_auto34758.gsd'
    lcr=0.81650
    kset=1000
    seed=8
    #heating
    index1_heat=3476
    index_end_heat = workflow_simu_to_mysql_heat_from_honeycomb_part(init_gsd,index1_heat,lcr,kset,seed)
    #cooling
    init = index1_heat#3476
    ends = 3481 #index_end_heat#3481 
    index1_cool=ends+1#3482
    workflow_simu_to_mysql_cool_from_honeycomb_part(init,ends,index1_cool,lcr,kset,seed)

def workflow_simu_to_mysql_heat_from_honeycomb_part(init_gsd,index1,lcr,kset,seed=9):
    R"""
    EXAMPLE:
    seed=8
    index1=3441
    lcr1=0.79
    tt.workflow_simu_to_mysql_heat_from_honeycomb_part(index1,lcr1,seed)
    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testhoneycomb3-8-12-part1"lcr=0.79,0.80
    #particle: 1369_8,2433_8
    #"check_init_melt_from_honeycomb_part.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    kt1=2.0
    stp=1.0
    ktend=7.0
    #kset=1000#700
    #init_gsd = '/home/tplab/hoomd-examples_0/trajectory_auto2433_8.gsd'#init_gsd='/home/tplab/hoomd-examples_0/trajectory_auto3365_9.gsd'
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_honeycomb_bidispersion_heat_seed as sa_k
    end_index=sa_k.workflow(init_gsd,index1=index1,kt1=kt1,step=stp,kt_end=ktend,linear_compression_ratio=lcr,kset=kset,seed_set=seed)
    'get file index123'
    #end_index=int(index1+9)
    
    #step3
    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    filename_kl=da.saveIndexkTPsi36Seed(start_index=index1,end_index=end_index,kt1=kt1,step=stp,
                                        linear_compression_ratio=lcr,kset=kset,randomseed=seed)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='melt_from_honeycomb_part_test'
        | simu_index | HarmonicK | LinearCompressionRatio | kT |
        Psi3Global | Psi6Global | RandomSeed |
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='melt_from_honeycomb_part_test')

    #step5
    #watch kT limitation while heating
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,final_cut=True,
                                bond_plot =True,
                                show_traps=True,
                                trap_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12-part1",
                                trap_lcr=lcr)
        i+=1
    
    return end_index

def workflow_simu_to_mysql_cool_from_honeycomb_part(index_init,index_init_end,index1,lcr,kset,seed=9):
    R"""
    Introduction:
        cool other systems(by index_init~end) into target systems(by index1~end).
    Example:
        init = 3454
        ends = 3459
        index1=3460
        seed=8
        tt.workflow_simu_to_mysql_cool_from_honeycomb_part(init,ends,index1,seed)
    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testhoneycomb_part3-11-6"
    #particle:"testhoneycomb3-10-6"
    #"check_init_depin_from_honeycomb_part.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    #lcr=0.80
    #kset=1100
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_honeycomb_bidispersion_cool_seed as sa_k
    end_index=sa_k.workflow(index_init,index_init_end,index1,linear_compression_ratio=lcr,kset=kset,seed_set=seed)
    'get file index123'
    #end_index=int(index1+9)

    #step3
    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    kt1=1.0
    stp=1.0
    filename_kl=da.saveIndexkTPsi36Seed(start_index=index1,end_index=end_index,kt1=kt1,step=stp,
                                        linear_compression_ratio=lcr,kset=kset,randomseed=seed)
    #saveIndexCN4CN6SeedPsi6Seed
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    #print('\n'+filename_kl)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='melt_from_honeycomb_part_test'
        | simu_index | HarmonicK | LinearCompressionRatio | kT |
        Psi3Global | Psi6Global | RandomSeed |
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='melt_from_honeycomb_part_test')

    #step5
    #watch kT limitation while cooling
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,final_cut=True,
                                bond_plot =True,
                                show_traps=True,
                                trap_filename="/home/tplab/hoomd-examples_0/testhoneycomb3-8-12-part1",
                                trap_lcr=lcr)
        i+=1

def workflow_simu_to_mysql_pin_hex_to_honeycomb_part_step2(index_old,index1,lcr):#[x]
    #set parameters
    
    k1=50.0
    stp=10.0
    kend=90.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_honeycomb_bidispersion_pin_step2 as sa_hbp
    end_index=sa_hbp.workflow(index_old=index_old,index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr)
    'get file index123'

    #get analyzed data
    import data_analysis_cycle as da
    filename_klp=da.saveIndexPsi3Psi6(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_klp)
    'get file named index1 index2 klp'

def workflow_simu_to_mysql_pin_hex_to_honeycomb_part_step3(index_old,index1,lcr):#[x]
    #set parameters

    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_honeycomb_bidispersion_pin_step3 as sa_hbp
    k1=700.0
    index_new=index1
    while index_old<index1:
        sa_hbp.workflow(index_old=index_old,index1=index_new,kset=k1,linear_compression_ratio=lcr)
        'get file index123'
        index_old+=1
        index_new+=1

    #get analyzed data
    import data_analysis_cycle as da
    filename_klp=da.saveIndexPsi3Psi6(start_index=index1,end_index=1512,k1=k1,step=0.0,linear_compression_ratio=lcr)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_klp)
    'get file named index1 index2 klp'

def workflow_simu_to_mysql_depin_from_kagome(index1,lcr,seed=9):#num
    R"""
    EXP:
        import workflow_part as tt
        index1=1513
        lists = np.linspace(0.8,0.99,20)
        lists = np.insert(lists,0,0.866)

        for lcr in lists:
            tt.workflow_simu_to_mysql_depin_from_kagome(index1=index1,lcr=lcr)
            #print(index1,lcr)
            index1=index1+10
    """
    #set parameters
    k1=0.0
    stp=10.0
    kend=90.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_kagome_depin as sa_k
    end_index=index1+9#end_index=sa_k.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    'get file index123'

    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    filename_kl=da.saveIndexCN4CN6SeedPsi6(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_kl)
    'get file named index1 index2 kl'

    
    #loadDataToMysql
    R"""
    Note: the format of table_name='depin_from_kagome'
        simu_index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed |
        Psi6Global|
    """
    '''
    prefix='/home/tplab/Downloads/'
    start_index=index1
    end_index=index1+30
    filename_kl=prefix+str(start_index)+'-'+str(end_index)+'kl'
    '''
    
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='depin_from_kagome')

def workflow_simu_to_mysql_pin_hex_to_kagome(index1,lcr,seed=9,account='tplab'):#num
    #set parameters

    k1=40#100.0
    stp=6#100.0
    kend=94#1000.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_kagome_pin as sa_k
    end_index=sa_k.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed,account=account)
    'get file index123'

    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    filename_kl=da.saveIndexCN4CN6SeedPsi6(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed,account=account)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_kl)
    'get file named index1 index2 kl'

    
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_kagome'
        simu_index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed |
        Psi6Global|
    """
    '''
    prefix='/home/tplab/Downloads/'
    start_index=index1
    end_index=index1+30
    filename_kl=prefix+str(start_index)+'-'+str(end_index)+'kl'
    '''
    
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='pin_hex_to_kagome')

def workflow_simu_to_mysql_depin_from_kagome_part(index1,lcr,seed=9):#num
    R"""
    EXAMPLE:
    #scan seed
    seed=0
    while seed<9.5:
        index1=3063
        lcr1=0.80
        #scan lcr and kT 0.80-0.90
        while lcr1<0.905:
            tt.workflow_simu_to_mysql_depin_from_kagome_part(index1,lcr1,seed)
            index1+=10
            lcr1+=0.01

        seed+=1
    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_part3-11-6"
    #particle:"testkagome3-10-6"
    #"check_init_depin_from_kagome_part.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_kagome_part_depin_seed as sa_k
    end_index=sa_k.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    'get file index123'
    #end_index=int(index1+9)
    
    #step3
    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    filename_kl=da.saveIndexCN4CN3SeedPsi6(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)
    #saveIndexCN4CN6SeedPsi6Seed
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_kl)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='depin_from_kagome_part'
        simu_index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed |
        Psi6Global|
    """
    '''
    prefix='/home/tplab/Downloads/'
    start_index=index1
    end_index=index1+30
    filename_kl=prefix+str(start_index)+'-'+str(end_index)+'kl'
    '''

    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='depin_from_kagome_part_repeat')

def workflow_simu_to_mysql_depin_from_kagome_part_oop_kt(index1,lcr=0.866,kT=1.0,seed=9,account='tplab'):#num
    R"""
    EXAMPLE:
    seed=9
    index1=3916
    lcr1=0.866
    kt=10.0
    while kt>0.9:
        index_end=tt.workflow_simu_to_mysql_depin_from_kagome_part_oop_kt(index1,kT=kt)
        index1=index_end+1
        kt=kt-1.0
    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_part3-11-6"
    #particle:"testkagome3-10-6"
    #"check_init_depin_from_kagome_part.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    trap_name = "testkagome_part3-11-6"
    
    import symmetry_transformation_v2_9.pin_seed_oop as pin
    wk = pin.workflow_uniform(index1,account,k1,stp,kend,lcr,kT,seed,trap_name)
    wk.set_init_state_parameters(depin_from_kagome=True)
    end_index = wk.workflow()
    
    #end_index = index1 + 9
    'get file index123'

    #step3
    #get analyzed data
    import data_analysis_cycle as da
    filename_kl=da.saveIndexklTCN3CN4Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)
    print('\n'+filename_kl)
    'get file named index1 index2 klt'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='depin_from_kagome_part_klt'
        |SimuIndex|HarmonicK|LinearCompressionRatio|kT|
        CoordinationNum3Rate|CoordinationNum4Rate|RandomSeed|
    """
    
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='depin_from_kagome_part_klt')
    
    
    
    #step5
    #watch kT limitation while cooling
    import getDataAndDiagram as scatt
    scatt.workflow_mysql_to_data_depin_from_kagome_part_klt()
    """
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,
        coordination_number=True,
        account=account)
        i+=1
    """
    """
    import data_analysis_cycle as da
    index1 = 3925
    seed = 9
    i=index1
    da.save_from_gsd(simu_index=i,seed=seed,bond_plot=True,show_traps=True,trap_filename="/home/tplab/hoomd-examples_0/testkagome_part3-11-6",trap_lcr=0.866)
    """
    return end_index

def workflow_simu_to_mysql_pin_hex_to_kagome_part(index1,lcr,seed=9):
    R"""
    EXAMPLE:
    #scan seed
    seed=0
    while seed<9:
        index1=3173
        lcr1=0.80
        #scan lcr and kT 0.80-0.90
        while lcr1<0.905:
            tt.workflow_simu_to_mysql_pin_hex_to_kagome_part(index1,lcr1,seed)
            index1+=10
            lcr1+=0.01

        seed+=1
    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_part3-11-6"
    #particle:"testhex3-16-8"lcr=0.8
    #"check_init_pin_hex_to_kagome_part.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_kagome_part_pin_seed as sa_k
    end_index=sa_k.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    'get file index123'
    #end_index=int(index1+9)
    
    #step3
    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    filename_kl=da.saveIndexCN4CN3SeedPsi6(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_kl)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_kagome_part_repeat'
        simu_index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed |
        Psi6Global|
    """
    '''
    prefix='/home/tplab/Downloads/'
    start_index=index1
    end_index=index1+30
    filename_kl=prefix+str(start_index)+'-'+str(end_index)+'kl'
    '''
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='pin_hex_to_kagome_part_repeat')

def workflow_simu_to_mysql_pin_hex_to_kagome_part2(index1,lcr,seed=9):
    R"""
    EXAMPLE:
    #scan seed
    seed=0
    while seed<9:
        index1=3283
        lcr1=0.80
        #scan lcr and kT 0.80-0.90
        while lcr1<0.905:
            tt.workflow_simu_to_mysql_pin_hex_to_kagome_part2(index1,lcr1,seed)
            index1+=10
            lcr1+=0.01

        seed+=1
    #20220626
    #scan seed
    seed=0
    while seed<9:
        index1=3173
        lcr1=0.80
        #scan lcr and kT 0.80-0.90
        while lcr1<0.905:
            tt.workflow_simu_to_mysql_pin_hex_to_kagome_part(index1,lcr1,seed)
            index1+=10
            lcr1+=0.01

        seed+=1

    #scan seed
    seed=0
    while seed<9:
        index1=3283
        lcr1=0.80
        #scan lcr and kT 0.80-0.90
        while lcr1<0.905:
            tt.workflow_simu_to_mysql_pin_hex_to_kagome_part2(index1,lcr1,seed)
            index1+=10
            lcr1+=0.01

        seed+=1
    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_part3-11-6"
    #particle:"testhex3-16-8"lcr=0.8
    #"check_init_pin_hex_to_kagome_part.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    k1=1100.0
    stp=100.0
    kend=2000.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_kagome_part_pin_seed as sa_k
    end_index=sa_k.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    'get file index123'
    #end_index=int(index1+9)
    
    #step3
    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    filename_kl=da.saveIndexCN4CN3SeedPsi6(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_kl)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_kagome_part_repeat'
        simu_index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed |
        Psi6Global|
    """
    '''
    prefix='/home/tplab/Downloads/'
    start_index=index1
    end_index=index1+30
    filename_kl=prefix+str(start_index)+'-'+str(end_index)+'kl'
    '''
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='pin_hex_to_kagome_part_repeat')

def workflow_simu_to_mysql_pin_hex_to_kagome_part_oop_kT(index1,lcr,kT=1.0,seed=9,account='tplab'):#[x]
    R"""
    INTRODUCTION:
    example4 :
        seed=9
        index1=3696
        lcr1=0.88
        kT=0.0
        while kT<1.01:
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_part_oop_kT(index1=index1,lcr=lcr1,kT=kT,seed=seed)
            print(index1)
            print(kT)
            index1 = index_end + 1
            kT = kT + 0.1

    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_part3-11-6"
    #particle:"testhex3-16-8"lcr=0.8
    #"check_init_pin_hex_to_kagome_part.png"

    size_par < size_trap checked right!
    """

    #step2
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    trap_name = "testkagome_part3-11-6"
    #get simulation results
    
    import symmetry_transformation_v2_9.pin_seed_oop as pin
    wk = pin.workflow_uniform(index1,account,k1,stp,kend,lcr,kT,seed,trap_name)
    end_index = wk.workflow()
    #end_index = index1 + 9
    'get file index123'

    #step3
    #get analyzed data
    import data_analysis_cycle as da
    filename_kl=da.saveIndexklTCN3CN4Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)
    print('\n'+filename_kl)
    'get file named index1 index2 klt'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_kagome_part_klt'
        |SimuIndex|HarmonicK|LinearCompressionRatio|kT|
        CoordinationNum3Rate|CoordinationNum4Rate|RandomSeed|
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='pin_hex_to_kagome_part_klt')
    
    
    #step5
    #watch kT limitation while cooling
    
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,
        coordination_number=True,
        account=account)
        i+=1
    
    return end_index

def workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m(index1,lcr,kT=1.0,seed=9,account='tplab',check=True):#[x]
    R"""
    INTRODUCTION:
    example4 :
         #scan seed
        seed=0
        while seed<8.5:
            #pin sequence-GPU
            index1=4466
            lcr1=0.80#less than 0.75 is dangerous! some particles may not effected by trap!
            while lcr1<0.905:
                #print(index1,lcr1,seed)
                end_index=tt.workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m(index1=index1,lcr=lcr1,seed=seed,account='remote')
                index1=end_index+1
                lcr1=lcr1+0.01
            seed+=1
    exp5:
        seed=9
        index1=4356
        lcr1=0.80
        while lcr1<0.905:
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m(index1=index1,lcr=lcr1,seed=seed,account='remote')
            index1 = index_end + 1
            lcr1 = lcr1 + 0.01
    exp6:
        import workflow_part as tt
        import numpy as np
        seed=9
        index1=5483
        list_lcr = np.linspace(0.8,0.91,12)
        list_lcr[-1] = 0.866
        for lcr1 in list_lcr:
            print(index1,lcr1)#kagome pin precise
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m(index1=index1,lcr=lcr1,seed=seed,account='remote',check=False)
            index1 = index1 + 10
            lcr1 = lcr1 + 0.01
    exp7:
        import workflow_part as tt
        import numpy as np
        seed=0
        index1=4718
        list_lcr = np.linspace(0.851,0.861,6)
        for lcr1 in list_lcr:
            print(index1,lcr1)#kagome pin precise
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m(index1=index1,lcr=lcr1,seed=seed,account='remote',check=False)#
            index1 = index1 + 10
            lcr1 = lcr1 + 0.01
    exp8:
        import workflow_part as tt
        import numpy as np
        seed=4
        index1=5723
        list_lcr = np.linspace(0.841,0.849,5)
        for lcr1 in list_lcr:
            print(index1,lcr1)#kagome pin precise
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m\
                (index1=index1,lcr=lcr1,seed=seed,account='remote',check=False)#
            index1 = index1 + 10
            lcr1 = lcr1 + 0.01
    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome3-11-6"
    #particle:"testhex3-16-8"lcr=0.8
    #"lcr0.8~k100kagome3-11-6&hex3-16-8.png"

    size_par < size_trap checked right!
    """

    #step2
    #set parameters
    k1=21#41#40.0#100.0
    stp=2#6.0#100.0
    kend=39#59#94.0#1000.0
    trap_name = "testkagome3-11-6"
    #get simulation results
    if not check:
        import symmetry_transformation_v2_9.pin_seed_oop as pin
        wk = pin.workflow_uniform(index1,account,k1,stp,kend,lcr,kT,seed,trap_name,mode="--mode=cpu")
        end_index = wk.workflow()
        #end_index = index1 + 9
        'get file index123'
    else:
        end_index = index1 + 9
        print(k1)

    #step3
    #get analyzed data
    if not check:
        import data_analysis_cycle as da
        filename_kl=da.saveIndexklTCN3CN4Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)
        print('\n'+filename_kl)
        'get file named index1 index2 klt'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_kagome_klt_2m'
        |SimuIndex|HarmonicK|LinearCompressionRatio|kT|
        CoordinationNum3Rate|CoordinationNum4Rate|RandomSeed|
    """
    if not check:
        import opertateOnMysql as osql
        osql.loadTxtDataToMysql\
        (path_to_file_name=filename_kl,
        table_name='pin_hex_to_kagome_klt_2m')
    
    
    #step5
    #watch kT limitation while cooling
    """
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,
        coordination_number=True,
        account=account)
        i+=1
    
    """
    
    return end_index

def workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m_kt25(index1,lcr,kT=1.0,seed=9,account='tplab'):#[x]
    R"""
    INTRODUCTION:
    example4 :
        seed=9
        index1=4696
        lcr1=0.80
        while lcr1<0.905:
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_oop_klt_2m_kt25(index1=index1,lcr=lcr1,seed=seed,account='remote')
            index1 = index_end + 1
            lcr1 = lcr1 + 0.01
    
    
    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome3-11-6"
    #particle:"testhex3-16-8"lcr=0.8
    #"lcr0.8~k100kagome3-11-6&hex3-16-8.png"

    size_par < size_trap checked right!
    """

    #step2
    #set parameters
    k1=50.0
    stp=30.0
    kend=50.0
    trap_name = "testkagome3-11-6"
    #get simulation results
    """
    import symmetry_transformation_v2_9.pin_seed_oop as pin
    wk = pin.workflow_uniform(index1,account,k1,stp,kend,lcr,kT,seed,trap_name)
    end_index = wk.workflow()
    
    """
    end_index = index1# + 9
    'get file index123'
    
    #step3
    #get analyzed data
    """
    import data_analysis_cycle as da
    filename_kl=da.saveIndexklTCN3CN4Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)
    print('\n'+filename_kl)
    
    """
    'get file named index1 index2 klt'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_kagome_klt_2m'
        |SimuIndex|HarmonicK|LinearCompressionRatio|kT|
        CoordinationNum3Rate|CoordinationNum4Rate|RandomSeed|
    """
    """
    import opertateOnMysql as osql
    osql.loadDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='pin_hex_to_kagome_klt_2m')
    """
    
    
    
    #step5
    #watch kT limitation while cooling
    
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,
        coordination_number=True,
        account=account)
        i+=1
    
    return end_index

def workflow_simu_to_mysql_pin_hex_to_kagome_part_oop_klt_2m(index1,lcr,kT=1.0,seed=9,account='tplab'):#[x]
    R"""
    INTRODUCTION:
    example4 :
        seed=9
        index1=3696
        lcr1=0.88
        kT=0.0
        while kT<1.01:
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_part_oop_kT(index1=index1,lcr=lcr1,kT=kT,seed=seed)
            print(index1)
            print(kT)
            index1 = index_end + 1
            kT = kT + 0.1
    exp5:
        seed=9
        index1=4356
        lcr1=0.80
        while lcr1<0.905:
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_part_oop_klt_2m(index1=index1,lcr=lcr1,seed=seed,account='remote')
            index1 = index_end + 1
            lcr1 = lcr1 + 0.01
    exp low T:
        import workflow_part as tt
        seed=9
        index1=5108
        lcr1=0.81
        while lcr1<0.905:
            #print(index1,lcr1,seed)
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_part_oop_klt_2m(index1=index1,lcr=lcr1,kT=0.1,seed=seed,account='remote')
            index1 = index_end + 1#index1 = index1+10#
            lcr1 = lcr1 + 0.01
    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_part3-11-6"
    #particle:"testhex3-16-8"lcr=0.8
    #"check_init_pin_hex_to_kagome_part.png"

    size_par < size_trap checked right!
    """

    #step2
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    trap_name = "testkagome_part3-11-6"
    #get simulation results
    
    import symmetry_transformation_v2_9.pin_seed_oop as pin
    wk = pin.workflow_uniform(index1,account,k1,stp,kend,lcr,kT,seed,trap_name,mode="--mode=cpu")
    end_index = wk.workflow()
    #end_index = index1 + 9
    'get file index123'

    #step3
    #get analyzed data
    import data_analysis_cycle as da
    filename_kl=da.saveIndexklTCN3CN4Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)
    print('\n'+filename_kl)
    'get file named index1 index2 klt'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_kagome_part_klt_2m'
        |SimuIndex|HarmonicK|LinearCompressionRatio|kT|
        CoordinationNum3Rate|CoordinationNum4Rate|RandomSeed|
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='pin_hex_to_kagome_part_klt_2m')
    
    
    #step5
    #watch kT limitation while cooling
    """
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,
        coordination_number=True,
        account=account)
        i+=1
    """
    
    return end_index

def workflow_simu_to_mysql_melt_from_kagome_part(index1,lcr,seed=9):
    R"""
    EXAMPLE:
    #scan seed
    seed=0
    while seed<9.5:
        index1=3063
        lcr1=0.80
        #scan lcr and kT 0.80-0.90
        while lcr1<0.905:
            tt.workflow_simu_to_mysql_melt_from_kagome_part(index1,lcr1,seed)
            index1+=10
            lcr1+=0.01

        seed+=1
    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_part3-11-6"
    #particle:"testkagome3-10-6"
    #"check_init_depin_from_kagome_part.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    kt1=11.0
    stp=1.0
    ktend=20.0
    kset=300
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_kagome_part_melt_seed as sa_k
    end_index=sa_k.workflow(index1=index1,kt1=kt1,step=stp,kt_end=ktend,linear_compression_ratio=lcr,kset=kset,seed_set=seed)
    'get file index123'
    #end_index=int(index1+9)
    
    #step3
    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    filename_kl=da.saveIndexkTCN4CN3SeedPsi6(start_index=index1,end_index=end_index,kt1=kt1,step=stp,
                                            linear_compression_ratio=lcr,kset=kset,randomseed=seed)
    #saveIndexCN4CN6SeedPsi6Seed
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    #print('\n'+filename_kl)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='melt_from_kagome_part'
        simu_index | HarmonicK | LinearCompressionRatio | kT |
        CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed |
        Psi6Global|
    """
    '''
    prefix='/home/tplab/Downloads/'
    start_index=index1
    end_index=index1+30
    filename_kl=prefix+str(start_index)+'-'+str(end_index)+'kl'
    '''

    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='melt_from_kagome_part_test')

def workflow_simu_to_mysql_heat_treat_from_kagome_part():
    R"""
    Example:
        tt.workflow_simu_to_mysql_heat_treat_from_kagome_part()
    """
    #init
    init_gsd = '/home/tplab/hoomd-examples_0/trajectory_auto3496_8.gsd'
    lcr=0.86602
    kset=900
    seed=8
    #heating
    index1_heat=3498
    index_end_heat = workflow_simu_to_mysql_heat_from_kagome_part(init_gsd,index1_heat,lcr,kset,seed)
    #cooling
    init = index1_heat#3498
    ends = index_end_heat#3501
    index1_cool=ends+1#3502
    workflow_simu_to_mysql_cool_from_kagome_part(init,ends,index1_cool,lcr,kset,seed)

def workflow_simu_to_mysql_heat_from_kagome_part(gsd,index1,lcr,kset,seed=9):
    R"""
    EXAMPLE:
    seed=9
    index1=3413
    lcr1=0.88
    #scan lcr and kT 0.80-0.90
    tt.workflow_simu_to_mysql_heat_from_kagome_part(index1,lcr1,seed)
    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_part3-11-6"
    #particle:"testkagome3-10-6"
    #"check_init_depin_from_kagome_part.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    kt1=2.0
    stp=1.0
    ktend=5.0
    #kset=300
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_kagome_part_heat_seed as sa_k
    end_index=sa_k.workflow(gsd,index1=index1,kt1=kt1,step=stp,kt_end=ktend,linear_compression_ratio=lcr,kset=kset,seed_set=seed)
    'get file index123'
    #end_index=int(index1+9)
    
    #step3
    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    filename_kl=da.saveIndexkTCN4CN3SeedPsi6(start_index=index1,end_index=end_index,kt1=kt1,step=stp,
                                            linear_compression_ratio=lcr,kset=kset,randomseed=seed)
    #saveIndexCN4CN6SeedPsi6Seed
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    #print('\n'+filename_kl)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='melt_from_kagome_part'
        simu_index | HarmonicK | LinearCompressionRatio | kT |
        CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed |
        Psi6Global|
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='melt_from_kagome_part_test')

    return end_index

def workflow_simu_to_mysql_cool_from_kagome_part(index_init,index_init_end,index1,lcr,kset,seed=9):
    R"""
    Introduction:
        cool other systems(by index_init~end) into target systems(by index1~end).
    Example:
        init = 3413
        ends = 3416
        index1=3417
        lcr1=0.88
        seed=9
        tt.workflow_simu_to_mysql_cool_from_kagome_part(init,ends,index1,lcr1,seed)
    """
    #step1
    #depin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_part3-11-6"
    #particle:"testkagome3-10-6"
    #"check_init_depin_from_kagome_part.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    #lcr=0.88
    #kset=300
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_kagome_part_cool_seed as sa_k
    end_index=sa_k.workflow(index_init,index_init_end,index1,linear_compression_ratio=lcr,kset=kset,seed_set=seed)
    'get file index123'
    #end_index=int(index1+9)

    #step3
    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    kt1=1.0
    stp=1.0
    filename_kl=da.saveIndexkTCN4CN3SeedPsi6(start_index=index1,end_index=end_index,kt1=kt1,step=stp,
                                            linear_compression_ratio=lcr,kset=kset,randomseed=seed)
    #saveIndexCN4CN6SeedPsi6Seed
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    #print('\n'+filename_kl)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='melt_from_kagome_part'
        simu_index | HarmonicK | LinearCompressionRatio | kT |
        CoordinationNum4Rate | CoordinationNum6Rate | RandomSeed |
        Psi6Global|
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='melt_from_kagome_part_test')

    #step5
    #watch kT limitation while cooling
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,final_cut=True,
                                bond_plot =True,
                                show_traps=True,
                                trap_filename="/home/tplab/hoomd-examples_0/testkagome_part3-11-6",
                                trap_lcr=lcr)
        i+=1

def workflow_simu_to_mysql_depin_from_kagome_cycle_oop(index1,lcr,seed=9,account='tplab'):#[x]
    R"""
    INTRODUCTION:

    Parameters:
        initial state: kagome3-12-6  
        traps:kagome_cycle3-4-6 
        safe lcr: [0.80,1.00]
    Examples1:
        #scan seed
        seed=0
        while seed<10:
            #pin sequence-GPU
            index1=3526
            lcr1=0.80#less than 0.71 is dangerous! some particles may not effected by trap!
            while lcr1<0.905:
                end_index = tt.workflow_simu_to_mysql_depin_from_kagome_cycle_oop(index1=index1,lcr=lcr1,seed=seed)
                index1=end_index+1
                lcr1=lcr1+0.01
            seed+=1
    Example2：
        seed=9
        index1=3526
        lcr1=0.866
        tt.workflow_simu_to_mysql_depin_from_kagome_cycle_oop(index1=index1,lcr=lcr1,seed=seed)
    example3：
        seed=9
        index1=3546
        lcr1=0.866
        tt.workflow_simu_to_mysql_depin_from_kagome_cycle_oop(index1=index1,lcr=lcr1,seed=seed,account='remote')

    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_cycle3-4-6"lcr=1.0
    #particle:"testkagome3-12-6"
    #"lcr1~k100kagome_cycle3-4-6&hex3-16-8.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    k1=1100.0
    stp=100.0
    kend=2000.0
    trap_name = "testkagome_cycle3-4-6"
    #get simulation results
    import symmetry_transformation_v2_9.pin_seed_oop as pin
    wk = pin.workflow_depin(index1,account,k1,stp,kend,lcr,seed,trap_name)
    end_index = wk.workflow()
    #end_index =3535
    'get file index123'

    #step3
    #get analyzed data
    import data_analysis_cycle as da
    filename_kl=da.saveIndexCN4CN3SeedPsi6(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed,account=account)
    print('\n'+filename_kl)
    'get file named index1 index2 kl'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='depin_from_kagome_part'
        simu_index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum4Rate | CoordinationNum3Rate | RandomSeed |
        Psi6Global|
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='depin_from_kagome_part')

    #step5
    #watch kT limitation while cooling
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,
        coordination_number=True,
        account=account)
        """
        da.save_from_gsd(simu_index=i,seed=seed,final_cut=False,
                                bond_plot =True,
                                show_traps=True,
                                trap_filename='/home/'+account+'/hoomd-examples_0/testkagome_cycle3-4-6',
                                trap_lcr=lcr,
                                account=account)
        """
        i+=1
        
def workflow_simu_to_mysql_depin_from_kagome_cycle_oop_kT(index1,lcr,kT=1.0,seed=9,account='tplab'):
    R"""
    INTRODUCTION:

    Parameters:
        initial state: kagome3-12-6  
        traps:kagome_cycle3-4-6 
        safe lcr: [0.80,1.00]
    Examples1:
        #scan seed
        seed=0
        while seed<10:
            #pin sequence-GPU
            index1=3526
            lcr1=0.80#less than 0.71 is dangerous! some particles may not effected by trap!
            while lcr1<0.905:
                end_index = tt.workflow_simu_to_mysql_depin_from_kagome_cycle_oop(index1=index1,lcr=lcr1,seed=seed)
                index1=end_index+1
                lcr1=lcr1+0.01
            seed+=1
    example3：
        seed=9
        index1=3546
        lcr1=0.866
        tt.workflow_simu_to_mysql_depin_from_kagome_cycle_oop(index1=index1,lcr=lcr1,seed=seed,account='remote')
    example4 :
        seed=9
        index1=3586
        lcr1=0.866
        kT=0.0
        while kT<1.01:
            index_end=tt.workflow_simu_to_mysql_depin_from_kagome_cycle_oop_kT(index1=index1,lcr=lcr1,kT=kT,seed=seed)
            print(index1)
            print(kT)
            index1 = index_end + 1
            kT = kT + 0.1

    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_cycle3-4-6"lcr=1.0
    #particle:"testkagome3-12-6"
    #"lcr1~k100kagome_cycle3-4-6&hex3-16-8.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    trap_name = "testkagome_cycle3-4-6"
    #get simulation results

    import symmetry_transformation_v2_9.pin_seed_oop as pin
    wk = pin.workflow_uniform(index1,account,k1,stp,kend,lcr,kT,seed,trap_name)
    wk.set_init_state_parameters(nx=12,ny=6,depin_from_kagome=True)
    end_index = wk.workflow()

    #end_index = index1 + 9
    'get file index123'

    #step3
    #get analyzed data
    import data_analysis_cycle as da
    filename_kl=da.saveIndexklTCN3CN4Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)
    print('\n'+filename_kl)
    'get file named index1 index2 klt'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='depin_from_kagome_part_cycle'
        |SimuIndex|HarmonicK|LinearCompressionRatio|kT|
        CoordinationNum3Rate|CoordinationNum4Rate|RandomSeed|
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='depin_from_kagome_part_cycle')
    
    
    #step5
    #watch kT limitation while cooling
    """
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=i,seed=seed,
        coordination_number=True,
        account=account)
        i+=1
    """
    return end_index

def workflow_simu_to_mysql_pin_hex_to_kagome_cycle_oop_kT(index1,lcr,kT=1.0,seed=9,account='tplab'):
    R"""
    INTRODUCTION:

    Parameters:
        initial state: kagome3-12-6  
        traps:kagome_cycle3-4-6 
        safe lcr: [0.80,1.00]
    Examples1:
        seed=9
        index1=4116
        lcr1=0.88
        kT=0.1
        index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_cycle_oop_kT(index1=index1,lcr=lcr1,kT=kT,seed=seed)
        print(index1)
        print(kT)

    examples2:
        seed=9
        index1=4126
        lcr1=0.88
        kT=0.2
        while kT < 0.61:
            index_end=tt.workflow_simu_to_mysql_pin_hex_to_kagome_cycle_oop_kT(index1=index1,lcr=lcr1,kT=kT,seed=seed)
            print(index1)
            print(kT)
            index1 = index_end + 1
            kT = kT + 0.1
            
    """
    #step1
    #pin check
    R"""
    import getDataAndScatter as scatt
    scatt.showTrapsMap()

    #traps:"testkagome_cycle3-4-6"lcr=1.0
    #particle:"testkagome3-12-6"
    #"lcr1~k100kagome_cycle3-4-6&hex3-16-8.png"

    size_par < size_trap checked right!
    """
    #step2
    #set parameters
    k1=100.0
    stp=100.0
    kend=1000.0
    trap_name = "testkagome_cycle3-4-6"
    #get simulation results
    
    import symmetry_transformation_v2_9.pin_seed_oop as pin
    wk = pin.workflow_uniform(index1,account,k1,stp,kend,lcr,kT,seed,trap_name)
    end_index = wk.workflow()
    #end_index = index1 + 9
    'get file index123'

    #step3
    #get analyzed data
    import data_analysis_cycle as da
    filename_kl=da.saveIndexklTCN3CN4Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,kT=kT,randomseed=seed,account=account)
    print('\n'+filename_kl)
    'get file named index1 index2 klt'

    #step4
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_kagome_part_cycle'
        |SimuIndex|HarmonicK|LinearCompressionRatio|kT|
        CoordinationNum3Rate|CoordinationNum4Rate|RandomSeed|
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql\
    (path_to_file_name=filename_kl,
    table_name='pin_hex_to_kagome_part_cycle')
    
    """
    #step5
    #watch kT limitation while cooling
    import data_analysis_cycle as da
    i=index1
    while i<end_index+1:
        da.save_from_gsd(simu_index=4125,seed=9,final_cut=True,
        bond_plot=True,show_traps=True,
        trap_filename='/home/tplab/hoomd-examples_0/testkagome_cycle3-4-6',trap_lcr=0.88)
        #da.save_from_gsd(simu_index=i,seed=seed,
        #coordination_number=True,
        #account=account)
        i+=1
    """

    return end_index

def workflow_simu_to_mysql_depin_from_cairo(index1,lcr,seed=9):#[x]
    R"""
    Parameters:
        initial state: hex3-16-8
        traps:honeycomb3-8-12
        safe lcr: [0.6,0.80]
        critical lcr: 0.681
    Examples:
        #depin sequence-GPU
        index1=2563
        lcr1=0.681#less than 0.60 is dangerous! some particles may not effected by trap!
        tt.workflow_simu_to_mysql_depin_from_cairo(index1=index1,lcr=lcr1)
    """
    #set parameters
    table_name = "depin_from_cairo"

    k1=0.0
    stp=1.0
    kend=9.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_cairo_depin as sa_c
    #end_index=sa_c.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    end_index=2582
    'get file index123'
    #the real order parameter is not decided yet!!!!!!!!!!
    #------------------------------------------------------
    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    da.saveIndexCairoPsi345(index1,end_index,k1,stp,lcr)
    da.saveIndexCairoCN3456(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)
    #print('\n'+filename_kl)
    'get file named index1 index2 kl'
    
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_cairo'
        Simu_Index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum3Rate | CoordinationNum4Rate | 
        CoordinationNum6Rate |RandomSeed |
    """
    #import opertateOnMysql as osql
    #osql.loadDataToMysql(path_to_file_name=filename_kl,table_name=table_name)#"/home/tplab/Downloads/193-205kl"

    return end_index
    
def workflow_simu_to_mysql_pin_hex_to_cairo(index1,lcr,seed=9):#[x]
    R"""
    Parameters:
        initial state: hex3-16-8
        traps:honeycomb3-8-12
        safe lcr: [0.6,0.80]
        critical lcr: 0.681
    Examples:
        #scan seed
        seed=0
        while seed<9.5:
            #pin sequence-GPU
            index1=2153
            lcr1=0.60#less than 0.60 is dangerous! some particles may not effected by trap!
            while lcr1<0.805:
                tt.workflow_simu_to_mysql_pin_hex_to_cairo(index1=index1,lcr=lcr1,seed=seed)
                index1=index1+10
                lcr1=lcr1+0.01
            seed+=1
    Example2:
        import data_analysis_cycle as da
        index1=2203#2253
        end_index = 2212 #2262
        k1 = 100
        stp = 100
        lcr = 0.65
        seed=9
        da.saveIndexCairoPsi345(index1,end_index,k1,stp,lcr)
        da.saveIndexCairoCN3456(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)    
    """
    #set parameters
    table_name = "pin_hex_to_cairo"

    k1=100.0
    stp=100.0
    kend=1000.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_cairo_pin as sa_c
    #end_index=sa_c.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    end_index=index1+9
    'get file index123'

    #[x]
    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    filename_kl=da.saveIndexCN346Seed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)
    print('\n'+filename_kl)
    'get file named index1 index2 kl'
    
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_cairo'
        Simu_Index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum3Rate | CoordinationNum4Rate | 
        CoordinationNum6Rate |RandomSeed |
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql(path_to_file_name=filename_kl,table_name=table_name)#"/home/tplab/Downloads/193-205kl"

    return end_index

def workflow_simu_to_mysql_pin_hex_to_cairo_egct(index1,lcr,seed=9):#[x]
    R"""
    Parameters:
        initial state: hex3-16-8
        traps:honeycomb3-8-12
        safe lcr: [0.6,0.80]
        critical lcr: 0.681
    Examples:

    Example2:
        import workflow_part as tt
        seed=9#no randomseed saved in txt
        #pin sequence-GPU
        index1=2153
        lcr1=0.60#less than 0.60 is dangerous! some particles may not effected by trap!
        while lcr1<0.805:
            tt.workflow_simu_to_mysql_pin_hex_to_cairo_egct(index1=index1,lcr=lcr1,seed=seed)
            index1=index1+10
            lcr1=lcr1+0.01
            """
    #set parameters
    table_name = "pin_hex_to_cairo_egct2lcra"

    k1=100.0#0.0#
    stp=100.0#10.0#
    kend=1000.0#90.0#
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_cairo_pin as sa_c
    end_index=sa_c.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    end_index=index1+9
    'get file index123'

    #get analyzed data and generate txt table
    import data_analysis_cycle as da
    filename_kl=da.saveIndexCN346PCairoSeed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed)
    print('\n'+filename_kl)
    'get file named index1 index2 kl'
    
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_cairo_egct2lcra'
        [ simu_index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum3Rate | CoordinationNum4Rate | 
        CoordinationNum6Rate | PCairo | RandomSeed ]
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql(path_to_file_name=filename_kl,table_name=table_name)#"/home/tplab/Downloads/193-205kl"

    return end_index

def workflow_simu_to_mysql_pin_hex_to_cairo_egct_uniform(index1,lcr,seed=9,check=False,account='remote'):
    R"""
    Parameters:
        initial state: hex3-16-8
        traps:cairo3-6-6
        safe lcr: [0.6,0.80]
        critical lcr: 0.681
    Examples:

    Example2:
        import workflow_part as tt
        seed=9#no randomseed saved in txt
        #pin sequence-GPU
        index1=2153
        lcr1=0.60#less than 0.60 is dangerous! some particles may not effected by trap!
        while lcr1<0.805:
            tt.workflow_simu_to_mysql_pin_hex_to_cairo_egct(index1=index1,lcr=lcr1,seed=seed)
            index1=index1+10
            lcr1=lcr1+0.01
            """
    #set parameters
    table_name = "pin_hex_to_cairo_egct2lcra"

    k1=0.0#100.0#
    stp=10.0#100.0#
    kend=90.0#1000.0#

    trap_filename="testcairo3-6-6"
    #get simulation results
    #import symmetry_transformation_v2_9.symmetry_transformation_auto_cairo_pin as sa_c
    if check:
        end_index=index1+9
    else:
        import symmetry_transformation_v2_9.pin_seed_oop as sa_c
        #wk=sa_c.workflow_uniform(index1=index1,account=account,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,
        #                                seed_set=seed,trap_name=trap_filename,period=20,steps=2e4)
        #end_index=wk.workflow()
        end_index=index1+9
    'get file index123'

    #get analyzed data and generate txt table
    if not check:
        import data_analysis_cycle as da
        filename_kl=da.saveIndexCN346PCairoSeed(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr,randomseed=seed,account=account)
        print('\n'+filename_kl)
        'get file named index1 index2 kl'
        
    #loadDataToMysql
    R"""
    Note: the format of table_name='pin_hex_to_cairo_egct2lcra'
        [ simu_index | HarmonicK | LinearCompressionRatio | 
        CoordinationNum3Rate | CoordinationNum4Rate | 
        CoordinationNum6Rate | PCairo | RandomSeed ]
    """
    if not check:
        import opertateOnMysql as osql
        osql.loadTxtDataToMysql(path_to_file_name=filename_kl,table_name=table_name)#"/home/tplab/Downloads/193-205kl"

    return end_index

R"""
OLD CODE
"""

def workflow_simu_to_mysql_kl(index1,lcr,seed):#num
    R"""
    import workflow_part as tt
    index1 = 307
    listl = np.linspace(0.76,0.99,24)

    for lcr in listl:
        tt.workflow_simu_to_mysql_kl(index1=index1,lcr=lcr,seed=8)
        #print(index1,lcr)
        index1 = index1 + 20
    """
    #set parameters
    k1=0.1
    stp=0.1
    kend=2.0
    #get simulation results
    import symmetry_transformation_v2_9.melt_auto_honeycomb as sa_h
    #end_index=sa_h.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr,seed_set=seed)
    'get file index123'

    #get analyzed data
    import data_analysis_cycle as da
    filename_kl=da.saveIndexPressure(start_index=index1,end_index=index1+19,k1=k1,step=stp,linear_compression_ratio=lcr,seed_set=seed)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_kl)
    'get file named index1 index2 kl'

    #loadDataToMysql
    R"""
    Note: the format of table_name='depin_from_honeycomb'
        | SimuIndex| KBT| LinearCompressionRatio| Pressure| Psi6Global| RandomSeed
    """
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql(path_to_file_name=filename_kl,table_name='melt_hex_from_honeycomb_check')#"/home/tplab/Downloads/193-205kl"
    #'hex_from_honeycomb'
    #'depin_from_honeycomb'

def workflow_simu_to_mysql_hex_to_honeycomb(index1,lcr):#num
    #set parameters

    k1=500.0
    stp=10.0
    kend=800.0
    #get simulation results
    import symmetry_transformation_v2_9.symmetry_transformation_auto_honeycomb_pin as sa_k
    end_index=sa_k.workflow(index1=index1,k1=k1,step=stp,k_end=kend,linear_compression_ratio=lcr)
    'get file index123'

    #get analyzed data
    import data_analysis_cycle as da
    filename_kl=da.saveIndexPsi3Ratio(start_index=index1,end_index=end_index,k1=k1,step=stp,linear_compression_ratio=lcr)
    #filename_kl=da.saveIndexPsi(start_index=206,end_index=215,k1=k1,step=stp,linear_compression_ratio=0.79)
    print('\n'+filename_kl)
    'get file named index1 index2 kl'
    
    #loadDataToMysql
    R"""
    Note: the format of table_name='hex_to_honeycomb'
        simu_index | HarmonicK | LinearCompressionRatio | Psi3Global | Psi3Ratio
    """
    prefix='/home/tplab/Downloads/'
    start_index=index1
    end_index=index1+30
    filename_kl=prefix+str(start_index)+'-'+str(end_index)+'kl'
    import opertateOnMysql as osql
    osql.loadTxtDataToMysql(path_to_file_name=filename_kl,table_name='hex_to_honeycomb')#"/home/tplab/Downloads/193-205kl"
