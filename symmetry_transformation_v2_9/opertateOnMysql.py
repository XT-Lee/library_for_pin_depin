import numpy as np
import pymysql
def introduction():
    R"""
    Function list:
        connectToMysql: it helps program connect to Mysql database.
        loadDataToMysql: it is used to load data to Mysql database.
        getDataFromMysql: it is used to extract data from Mysql database through 'data=getDataFromMysql()'.
    Reference：
        https://zhuanlan.zhihu.com/p/51553625
    """
    pass

def createTableInMysql(new_table_name,param_name_data_type=None,old_table_name=None):
    R"""
    DATA_TYPE:    (<param_name1> <data_type1>,<param_name2> <data_type2>) 

    example:
        osql.createTableInMysql('pin_hex_to_honeycomb_klt_2m_gauss',\
        'simu_index integer unsigned not null,seed integer unsigned not null,lcr float, trap_gauss_epsilon float, temperature float')

    """
    mysql_conn = connectToMysql()

    if not param_name_data_type is None:
        sql = 'create table '+new_table_name+'( '+param_name_data_type+');'
    elif not old_table_name is None:
        sql = 'create table '+new_table_name+' like '+old_table_name+';'
        #create table xxx like pin_hex_to_cairo_egct2lcra;

    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(sql)
        mysql_conn.commit()
    except Exception as e:
        print(e)
        mysql_conn.rollback()

    mysql_conn.close()

def loadTxtDataToMysql(path_to_file_name,table_name):
    R"""
    Examples:
        operateOnMysql.loadTxtDataToMysql(
        path_to_file_name='/home/tplab/Downloads/91-192kl',
        table_name='honeycomb')
    """
    
    mysql_conn = connectToMysql()
    data=np.loadtxt(path_to_file_name)
    shape_data=np.shape(data)

    for row in np.linspace(1,shape_data[0],shape_data[0]):
        data_to_string='('
        #print(data[row.astype(int)-1])
        for column in np.linspace(1,shape_data[1],shape_data[1]):
            data_to_string=data_to_string+str(data[row.astype(int)-1,column.astype(int)-1])
            if column < shape_data[1]:
                data_to_string=data_to_string+','
        
        data_to_string=data_to_string+');'
        #print(data_to_string)
        sql = 'insert into '+table_name+' values'+data_to_string
        try:
            with mysql_conn.cursor() as cursor:
                cursor.execute(sql)
            mysql_conn.commit()
        except Exception as e:
            print(e)
            mysql_conn.rollback()

    mysql_conn.close()

def loadCsvDataToMysql(path_to_file_name,table_name):
    R"""
    Examples:
        operateOnMysql.loadCsvDataToMysql(
        path_to_file_name='/home/tplab/Downloads/91-192kl',
        table_name='honeycomb')
    """
    import pandas as pd
    mysql_conn = connectToMysql()
    data=pd.read_csv(path_to_file_name)
    
    shape_data=np.shape(data)
    col = data.columns
    len_col = len(col)
    #data[col[]]
    

    for row in range(shape_data[0]):
        data_to_string='('
        #print(data[row.astype(int)-1])
        for column in range(shape_data[1]-1):
            data_to_string=data_to_string+str(data.iloc[row,column+1])
            if column < shape_data[1]-2:
                data_to_string=data_to_string+','
        
        data_to_string=data_to_string+');'
        #print(data_to_string)
        sql = 'insert into '+table_name+' values'+data_to_string
        try:
            with mysql_conn.cursor() as cursor:
                cursor.execute(sql)
            mysql_conn.commit()
        except Exception as e:
            print(e)
            mysql_conn.rollback()

    mysql_conn.close()

def getDataFromMysql(table_name='',search_condition='',select_content='*'):#path_to_save_file=None,
    R"""
    Examples:
        data = operateOnMysql.getDataFromMysql(
                path_to_save_file='/home/tplab/Downloads/91-192kl'
                table_name='honeycomb',
                search_condition='where HarmonicK > 100')
    """
    
    mysql_conn = connectToMysql()

    sql = 'select '+select_content+' from '+table_name+' '+search_condition+';'#where column_name = xx
    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(sql)
        select_result = cursor.fetchall()
        #np.savetxt(path_to_save_file,select_result)
        return select_result
    except Exception as e:
        print(e)
        
        mysql_conn.rollback()

    mysql_conn.close()

def showTables(search_condition=''):
    R"""
    #show tables with p as head, se as tail
    show tables like ‘p%’ ‘%se’;
    """
    mysql_conn = connectToMysql()

    sql = 'show tables '+search_condition+';'
    try:
        with mysql_conn.cursor() as cursor:
            cursor.execute(sql)
        select_result = cursor.fetchall()
        #np.savetxt(path_to_save_file,select_result)
        return select_result
    except Exception as e:
        print(e)
        
        mysql_conn.rollback()

    mysql_conn.close()

def connectToMysql():
    mysql_conn = pymysql.connect(host='localhost',user='root',password='4KjnuS99n2tZO#ghs',db='hoomd_data')
    return mysql_conn