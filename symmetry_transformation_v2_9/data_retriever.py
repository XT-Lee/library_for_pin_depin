R"""
    To easily retrieve data.
"""
import opertateOnMysql as osql

class search_engine_for_simulation_database:
    def __init__(self):
        pass

    def get_table_names(self):
        R"""
        import data_retriever as dr
        se = dr.search_engine_for_simulation_database()
        se.get_table_names()

        table_name                      index
        'pin_hex_to_honeycomb_klt_2m'   
        'pin_hex_to_honeycomb_part_klt_2m'  
        """
        list_table_names = osql.showTables()
        for i in range(len(list_table_names)):
            print(list_table_names[i])

    def search_single_simu_by_lcr_k(self,table_name = 'pin_hex_to_honeycomb_part_klt_2m',lcr1=0.8,k1=300,kt1=1.0):
        R"""
        | SimuIndex | HarmonicK | LinearCompressionRatio | kT   | Psi3     | Psi6     | RandomSeed |
        'pin_hex_to_honeycomb_part_klt_2m'
        'pin_hex_to_honeycomb_klt_2m'
        """
        
        lcr_step = 0.0001
        lcr_min=lcr1 - 0.5*lcr_step
        lcr_max=lcr1 + 0.5*lcr_step
        kt_step = 0.1
        kt_min = kt1 - 0.5*kt_step
        kt_max = kt1 + 0.5*kt_step
        
        cont=' distinct SimuIndex '#, RandomSeed
        con=' where HarmonicK='+str(int(k1))+\
            ' and LinearCompressionRatio >'+str(lcr_min)+' and LinearCompressionRatio <'+str(lcr_max)+\
            ' and kT >'+str(kt_min)+' and kT <'+str(kt_max)
            #' order by RandomSeed asc'
        simu_index_seed=osql.getDataFromMysql(table_name=table_name,
                            search_condition=con,select_content=cont)
        print(simu_index_seed)#what if return more than 1 results? 0.8,300 as an example -> 4288,5060.
        #where 5060 is low Temperature.
        #just take the first result!
        return  simu_index_seed[0][0]#((index1,),)

    def search_single_simu_by_lcr(self,k1,table_name = 'pin_hex_to_honeycomb_part_klt_2m',lcr1=0.8,kt1=1.0):
        R"""
        | SimuIndex | HarmonicK | LinearCompressionRatio | kT   | Psi3     | Psi6     | RandomSeed |
        'pin_hex_to_honeycomb_part_klt_2m'
        'pin_hex_to_honeycomb_klt_2m'
        """
        
        lcr_step = 0.0001
        lcr_min=lcr1 - 0.5*lcr_step
        lcr_max=lcr1 + 0.5*lcr_step
        kt_step = 0.1
        kt_min = kt1 - 0.5*kt_step
        kt_max = kt1 + 0.5*kt_step
        
        cont=' distinct SimuIndex '#, RandomSeed
        con=' where HarmonicK='+str(int(k1))+\
            ' and LinearCompressionRatio >'+str(lcr_min)+' and LinearCompressionRatio <'+str(lcr_max)+\
            ' and kT >'+str(kt_min)+' and kT <'+str(kt_max)
            #' order by RandomSeed asc'
        simu_index_seed=osql.getDataFromMysql(table_name=table_name,
                            search_condition=con,select_content=cont)
        print(simu_index_seed)#what if return more than 1 results? 0.8,300 as an example -> 4288,5060.
        #where 5060 is low Temperature.
        #just take the first result!
        return  simu_index_seed[0][0]#((index1,),)


