import symmetry_transformation_v4_3.analysis_controller as ac
import numpy as np
import pandas as pd
import os


class analyze_a_gsd_file:
    def __init__(self):
        pass

    def analyze_gsd_files_and_print_6013(self):
        n_file = 30
        list_index = np.linspace(6013, 6042, n_file, dtype=int)
        list_gauss_epsilon = np.linspace(-3, -90, n_file, dtype=int)
        list_cn3 = np.zeros((n_file,))
        seed = 19
        print(seed)
        for i in range(n_file):
            simu_index = list_index[i]
            s2g = ac.get_a_gsd_from_setup()
            s2g.set_file_parameters(simu_index, seed)

            g2d = ac.get_data_from_a_gsd_frame(s2g.gsd_data[-1])
            list_cn3[i] = g2d.get_cn_k_from_a_gsd_frame()
            print(list_gauss_epsilon[i], list_cn3[i])

    def analyze_gsd_files_and_print_6043(self):
        n_file = 10
        list_index = np.linspace(6043, 6052, n_file, dtype=int)
        list_gauss_epsilon = np.linspace(-100, -1000, n_file, dtype=int)
        list_cn3 = np.zeros((n_file,))
        seed = 19
        print(seed)
        for i in range(n_file):
            simu_index = list_index[i]
            s2g = ac.get_a_gsd_from_setup()
            s2g.set_file_parameters(simu_index, seed)

            g2d = ac.get_data_from_a_gsd_frame(s2g.gsd_data[-1])
            list_cn3[i] = g2d.get_cn_k_from_a_gsd_frame()
            print(list_gauss_epsilon[i], list_cn3[i])

    def analyze_gsd_files_and_record_as_csv(self):
        import pandas as pd
        prefix_write = "/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_pin/"

        output_file_csv = prefix_write+'honeycomb_pin_scan_k100_1000'+'.csv'
        n_file = 10
        list_index = np.linspace(5793, 5802, n_file, dtype=int)
        list_lcr = [0.81]*n_file
        list_trap_gauss_epsilon = np.linspace(-100, -1000, n_file, dtype=int)
        list_cn3 = np.zeros((n_file,))
        # df_all = pd.DataFrame()
        for seed in [0, 1, 2]:  # ,3,4,7,8,9
            # seed = 7
            list_seed_record = [seed]*n_file
            for i in range(n_file):
                simu_index = list_index[i]
                s2g = ac.get_a_gsd_from_setup()
                s2g.set_file_parameters(simu_index, seed)

                g2d = ac.get_data_from_a_gsd_frame()
                list_cn3[i] = g2d.get_cn_k_from_a_gsd_frame(s2g.gsd_data[-1])

            df = pd.DataFrame(list_index, columns=['simu_index'])
            df['seed'] = list_seed_record
            df['lcr'] = list_lcr
            df['trap_gauss_epsilon'] = list_trap_gauss_epsilon
            df['cn3'] = list_cn3
            if seed == 0:
                df_all = df
            elif seed > 0:
                df_all = pd.concat([df_all, df])  # .append(df)
        pd.DataFrame.to_csv(df_all, output_file_csv)

    def analyze_gsd_files_and_print(self):
        n_file = 20
        list_index = np.linspace(5773, 5792, n_file, dtype=int)
        list_gauss_epsilon = np.linspace(-3, -60, n_file, dtype=int)
        list_cn3 = np.zeros((n_file,))
        seed = 7
        print(seed)
        for i in range(n_file):
            simu_index = list_index[i]
            s2g = ac.get_a_gsd_from_setup()
            s2g.set_file_parameters(simu_index, seed)

            g2d = ac.get_data_from_a_gsd_frame()
            list_cn3[i] = g2d.get_cn_k_from_a_gsd_frame(s2g.gsd_data[-1])
            print(list_cn3[i])

    def get_bond_plot_from_a_gsd(self):
        ss = ac.get_a_gsd_from_setup()
        for index1 in np.linspace(5971, 5974, 4):  # 5950,5959
            ss.set_file_parameters(index1, 9)
            ss.get_gsd_data_from_file()
            lf = ac.get_data_from_a_gsd_frame(ss.gsd_data[-1])
            lf.get_bonds_from_a_gsd_frame()

    def get_bond_plot_with_time_from_a_gsd(self, index1):
        ss = ac.get_a_gsd_from_setup()
        ss.set_file_parameters(index1, 9)
        ss.get_gsd_data_from_file()
        num_list = [100, 500, 1000, 1500]  # np.linspace(10,1999,10,dtype=int)
        for i in num_list:  # range(2000-1):
            print(ss.gsd_data[i].configuration.box)
            # print(ss.gsd_data[i].configuration.box.dimensions)
            lf = ac.get_data_from_a_gsd_frame(ss.gsd_data[i])
            png_filename = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_part_pin/' + \
                str(ss.simu_index)+'_'+str(i)+'.png'
            lf.get_bonds_png_from_a_gsd_frame(png_filename)


class analyze_a_series_of_gsd_file:
    def __init__(self):
        pass

    def get_cnks_from_csv_files_type_n_part(self, output_file_csv, seed_limit=9, add_type_3=False):
        R"""
        parameter:
            cn_k:(int) the k to calculate cn_k
        example:
            write_prefix = "/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_depin/"
            output_file_csvs = ["depin_type_n_from_type_n_part_klt_2m_gauss_6053.csv",
            "depin_type_n_from_type_n_part_klt_2m_gauss_6293.csv",
            "depin_type_n_from_type_n_part_klt_2m_gauss_7163.csv"]
            import symmetry_transformation_v4_3.list_code_analysis as lca
            asg = lca.analyze_a_series_of_gsd_file()
            for output_file_csv in output_file_csvs:
                fn = write_prefix+output_file_csv
                print(fn)
                asg.get_cnks_from_csv_files_type_n_part(fn)
        example2:
            "/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv"#0-6
            "/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv"#0-9
            "/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv"#0-9
            "/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv"#0-9
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        import workflow_analysis as wa
        at = wa.archimedean_tilings()
        gg = ac.get_gsds_from_mysql_or_csv()
        gsds_filename = gg.get_record_from_csv(output_file_csv)
        # list_simu = gg.record['simu_index'].values
        list_seeds = gg.record['seed'].values
        lcrs = gg.record['lcr'].values
        if add_type_3:
            gg.record['type_n'] = 3
        type_ns = gg.record['type_n'].values
        pdata = ac.get_a_gsd_from_setup()
        cnks = np.zeros((len(gsds_filename),))
        for i in range(len(gsds_filename)):
            if list_seeds[i] <= seed_limit:
                isExists = os.path.exists(gsds_filename[i])
                if isExists:
                    file_size_b = os.path.getsize(gsds_filename[i])
                    file_size_kb = file_size_b/1024
                    if file_size_kb > 1000:
                        pdata.get_gsd_data_from_filename(gsds_filename[i])
                        gdata = ac.get_data_from_a_gsd_frame(pdata.gsd_data[-1])
                        coord_num_k = at.get_coordination_number_k_for_type_n(type_ns[i])
                        cnk = gdata.get_cn_k_from_a_gsd_frame(tune_dis=3*lcrs[i], k=coord_num_k)
                        cnks[i] = cnk
                    else:
                        cnks[i] = -1
                else:
                    cnks[i] = -1
            else:
                cnks[i] = -1
        col_cnk = 'cn'+str(int(coord_num_k))  # the last type_n's cnk as column_name
        gg.record[col_cnk] = cnks
        gg.record['U_eq'] = gg.record['trap_gauss_epsilon'].values*0.86466  # *0.99613
        import pandas as pd
        # to strip 'unnamed:0' column which contains [1,2,3...]
        list_col = gg.record.columns.values[1:]
        pd.DataFrame.to_csv(gg.record[list_col], output_file_csv)

    def get_cn3s_from_mysql_honeycomb(self):
        R"""
            lcr0.81 honeycomb pin
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        ggfs = ac.get_gsds_from_mysql_or_csv()
        record = ggfs.get_record_from_sql_by_lcr()
        # record_np = np.array(record)
        gsds = ggfs.get_gsds_from_mysql_record(record)
        n_simu = len(gsds)
        list_cn3 = np.zeros((n_simu,))
        # list_cn3[:,:2] =  record_np[:,2:4]
        gf = ac.get_data_from_a_gsd_frame()
        for i in range(n_simu):
            frame = ggfs.get_last_frame_from_gsd(gsds[i])
            tune_dis = record[i][2]*3  # list_cn3[i,0]*3
            cn3 = gf.get_cn_k_from_a_gsd_frame(frame, tune_dis)
            list_cn3[i] = cn3
            # del gf

        """
        import matplotlib.pyplot as plt
        import matplotlib
        #Backend agg is non-interactive backend. Turning interactive mode off. 'QtAgg' is interactive mode
        matplotlib.use(backend="QtAgg")
        fig,ax = plt.subplots()
        ax.plot(list_lcr_k_cn3[:,1],list_lcr_k_cn3[:,2])
        #ax.plot()
        ax.set_aspect('equal','box')
        plt.show()
        plt.close()"""

        # save as csv
        import pandas as pd
        df = pd.DataFrame(record, columns=['simu_index', 'seed',
                          "lcr", 'trap_gauss_epsilon', 'temperature'])
        df['cn3'] = list_cn3
        # df.sort_values(by=['seed','trap_gauss_epsilon'])
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_pin/'
        csv_filename = prefix_write + 'honeycomb_pin_scan_k3_1000.csv'
        pd.DataFrame.to_csv(df, csv_filename)
        return list_cn3

    def get_cn3s_from_mysql_honeycomb_part(self):
        R"""
            lcr0.81 honeycomb pin
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        ggfs = ac.get_gsds_from_mysql_or_csv()
        record = ggfs.get_record_from_sql_by_lcr(
            lcr1=0.80, table_name='pin_hex_to_honeycomb_part_klt_2m_gauss')
        # record_np = np.array(record)
        gsds = ggfs.get_gsds_from_mysql_record(record)
        n_simu = len(gsds)
        list_cn3 = np.zeros((n_simu,))
        # list_cn3[:,:2] =  record_np[:,2:4]

        for i in range(n_simu):
            frame = ggfs.get_last_frame_from_gsd(gsds[i])
            tune_dis = record[i][2]*3  # list_cn3[i,0]*3
            gf = ac.get_data_from_a_gsd_frame(frame)  # error:missing last frame
            cn3 = gf.get_cn_k_from_a_gsd_frame(frame, tune_dis)
            list_cn3[i] = cn3
            del gf

        # save as csv
        import pandas as pd
        df = pd.DataFrame(record, columns=['simu_index', 'seed',
                          "lcr", 'trap_gauss_epsilon', 'temperature'])
        df['cn3'] = list_cn3

        """import matplotlib.pyplot as plt
        import matplotlib
        #Backend agg is non-interactive backend. Turning interactive mode off. 'QtAgg' is interactive mode
        matplotlib.use(backend="QtAgg")
        fig,ax = plt.subplots()
        ax.semilogx(-df['trap_gauss_epsilon'],list_cn3)
        #ax.plot()
        #ax.set_aspect('equal','box')
        plt.show()
        plt.close()"""

        # df.sort_values(by=['seed','trap_gauss_epsilon'])
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_part_pin/'
        csv_filename = prefix_write + 'honeycomb_part_pin_lcr80_scan_k3_1000.csv'
        pd.DataFrame.to_csv(df, csv_filename)
        return list_cn3

    def get_cn3s_from_csv_honeycomb_part_gauss_eq(self, output_file_csv):
        R"""
            lcr0.81 honeycomb pin
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        """prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_part_pin/'
        index1 = 6013
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_'+str(int(index1))+'.csv'"""
        ggfs = ac.get_gsds_from_mysql_or_csv()
        gsds = ggfs.get_record_from_csv(output_file_csv)
        n_simu = len(gsds)
        list_cn3 = np.zeros((n_simu,))
        # list_cn3[:,:2] =  record_np[:,2:4]

        for i in range(n_simu):
            frame = ggfs.get_last_frame_from_gsd(gsds[i])
            lcr1 = ggfs.record['lcr'].values[i]
            tune_dis = lcr1*3  # list_cn3[i,0]*3
            gf = ac.get_data_from_a_gsd_frame(frame)  # error:missing last frame
            cn3 = gf.get_cn_k_from_a_gsd_frame(tune_dis)
            list_cn3[i] = cn3
            del gf

        # save as csv
        import pandas as pd
        df = pd.DataFrame(ggfs.record, columns=['simu_index',
                          'seed', "lcr", 'trap_gauss_epsilon', 'temperature'])
        df['cn3'] = list_cn3
        df['U_eq'] = df['trap_gauss_epsilon'].values*0.86466  # *0.99613
        """import matplotlib.pyplot as plt
        import matplotlib
        #Backend agg is non-interactive backend. Turning interactive mode off. 'QtAgg' is interactive mode
        matplotlib.use(backend="QtAgg")
        fig,ax = plt.subplots()
        ax.semilogx(-df['trap_gauss_epsilon'],list_cn3)
        #ax.plot()
        #ax.set_aspect('equal','box')
        plt.show()
        plt.close()"""
        # df.sort_values(by=['seed','trap_gauss_epsilon'])
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_part_pin/'
        csv_filename = prefix_write+'pin_hex_to_honeycomb_part_klt_2m_gauss_6013_19_res.csv'
        # csv_filename = prefix_write + 'honeycomb_part_pin_lcr81_scan_k3_1000_fill_brownian.csv'
        pd.DataFrame.to_csv(df, csv_filename)
        # print(df.head(35))
        return list_cn3

    def get_cnks_from_csv_type_4569_part(self):
        R"""
        import symmetry_transformation_v4_3.list_code_analysis as lca
        asg = lca.analyze_a_series_of_gsd_file()
        asg.get_cnks_from_csv_type_4569_part()
        """
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/'

        # generate_simu_index_csv_type_4_pin_3_30()
        index1 = 6963  # 6823#[x]
        list_type_n = 4  # [x]
        output_file_csv = prefix_write + 'pin_hex_to_type_'+str(
            int(list_type_n))+'_part_klt_2m_gauss_'+str(int(index1))+'.csv'  # [x]
        self.get_cnks_from_csv_files_type_n_part(3, output_file_csv)

        # generate_simu_index_csv_type_5_pin_3_30()
        index1 = 7013  # 6823#[x]
        list_type_n = 5  # [x]
        output_file_csv = prefix_write + 'pin_hex_to_type_'+str(
            int(list_type_n))+'_part_klt_2m_gauss_'+str(int(index1))+'.csv'  # [x]
        self.get_cnks_from_csv_files_type_n_part(3, output_file_csv)

        # generate_simu_index_csv_type_6_pin_3_30()
        index1 = 7063  # 6823#[x]
        list_type_n = 6  # [x]
        output_file_csv = prefix_write + 'pin_hex_to_type_'+str(
            int(list_type_n))+'_part_klt_2m_gauss_'+str(int(index1))+'.csv'  # [x]
        self.get_cnks_from_csv_files_type_n_part(3, output_file_csv)

        # generate_simu_index_csv_type_9_pin_3_30()
        index1 = 7113  # 6823#[x]
        list_type_n = 9  # [x]
        output_file_csv = prefix_write + 'pin_hex_to_type_'+str(
            int(list_type_n))+'_part_klt_2m_gauss_'+str(int(index1))+'.csv'  # [x]
        self.get_cnks_from_csv_files_type_n_part(5, output_file_csv)

    def get_cn5s_from_csv_files_type_11_part(self):
        R"""
            import symmetry_transformation_v4_3.list_code_analysis as lca
            asg = lca.analyze_a_series_of_gsd_file()
            asg.get_cn5s_from_csv_files()
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/'
        index1 = 6733  # index1 = 6643
        output_file_csv = prefix_write + \
            'pin_hex_to_type_11_part_klt_2m_gauss_'+str(int(index1))+'.csv'
        gg = ac.get_gsds_from_mysql_or_csv()
        gsds_filename = gg.get_record_from_csv(output_file_csv)
        lis_simu = gg.record['simu_index'].values
        lcrs = gg.record['lcr'].values
        pdata = ac.get_a_gsd_from_setup()
        cn5s = np.zeros((len(gsds_filename),))
        for i in range(len(gsds_filename)):
            if lis_simu[i] > 6734:  # lis_simu[i] < 6798 and
                pdata.get_gsd_data_from_filename(gsds_filename[i])
                gdata = ac.get_data_from_a_gsd_frame(pdata.gsd_data[-1])
                cn5 = gdata.get_cn_k_from_a_gsd_frame(tune_dis=3*lcrs[i], k=5)
                cn5s[i] = cn5
            else:
                cn5s[i] = -1
        gg.record['cn5'] = cn5s
        gg.record['U_eq'] = gg.record['trap_gauss_epsilon'].values*0.86466  # *0.99613
        import pandas as pd
        # to strip 'unnamed:0' column which contains [1,2,3...]
        list_col = gg.record.columns.values[1:]
        pd.DataFrame.to_csv(gg.record[list_col], output_file_csv)

    def get_cn4s_from_csv_files_type_7_part(self):
        R"""
            import symmetry_transformation_v4_3.list_code_analysis as lca
            asg = lca.analyze_a_series_of_gsd_file()
            asg.get_cn5s_from_csv_files()
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/'
        index1 = 6913  # 6823#[x]
        # [x]
        output_file_csv = prefix_write + \
            'pin_hex_to_type_7_part_klt_2m_gauss_'+str(int(index1))+'.csv'
        gg = ac.get_gsds_from_mysql_or_csv()
        gsds_filename = gg.get_record_from_csv(output_file_csv)
        lis_simu = gg.record['simu_index'].values
        lcrs = gg.record['lcr'].values
        pdata = ac.get_a_gsd_from_setup()
        cn4s = np.zeros((len(gsds_filename),))
        for i in range(len(gsds_filename)):
            pdata.get_gsd_data_from_filename(gsds_filename[i])
            gdata = ac.get_data_from_a_gsd_frame(pdata.gsd_data[-1])
            cn4 = gdata.get_cn_k_from_a_gsd_frame(tune_dis=3*lcrs[i], k=4)
            cn4s[i] = cn4

        gg.record['cn4'] = cn4s
        gg.record['U_eq'] = gg.record['trap_gauss_epsilon'].values*0.86466  # *0.99613
        import pandas as pd
        # to strip 'unnamed:0' column which contains [1,2,3...]
        list_col = gg.record.columns.values[1:]
        pd.DataFrame.to_csv(gg.record[list_col], output_file_csv)

    def get_cn4s_from_csv_files_type_8(self):
        R"""
            import symmetry_transformation_v4_3.list_code_analysis as lca
            asg = lca.analyze_a_series_of_gsd_file()
            asg.get_cn4s_from_csv_files_type_8()
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/'
        index1 = 243  # [x]
        # [x]
        output_file_csv = prefix_write + 'pin_hex_to_type_8_klt_2m_gauss_'+str(int(index1))+'.csv'
        gg = ac.get_gsds_from_mysql_or_csv()
        gsds_filename = gg.get_record_from_csv(output_file_csv)
        lis_simu = gg.record['simu_index'].values
        list_seed = gg.record['seed'].values
        lcrs = gg.record['lcr'].values
        pdata = ac.get_a_gsd_from_setup()
        cn4s = np.zeros((len(gsds_filename),))
        for i in range(len(gsds_filename)):
            if list_seed[i] == 0:
                pdata.get_gsd_data_from_filename(gsds_filename[i])
                gdata = ac.get_data_from_a_gsd_frame(pdata.gsd_data[-1])
                cn4 = gdata.get_cn_k_from_a_gsd_frame(tune_dis=3*lcrs[i], k=4)
                cn4s[i] = cn4
            else:
                cn4s[i] = -1
        gg.record['cn4'] = cn4s
        gg.record['U_eq'] = gg.record['trap_gauss_epsilon'].values*0.86466  # *0.99613
        import pandas as pd
        # to strip 'unnamed:0' column which contains [1,2,3...]
        list_col = gg.record.columns.values[1:]
        pd.DataFrame.to_csv(gg.record[list_col], output_file_csv)

    def get_cn4s_from_csv_files_type_8_part(self):
        R"""
            import symmetry_transformation_v4_3.list_code_analysis as lca
            asg = lca.analyze_a_series_of_gsd_file()
            asg.get_cn4s_from_csv_files_type_8_part()
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/'
        index1 = 513  # [x]
        # [x]
        output_file_csv = prefix_write + \
            'pin_hex_to_type_8_part_klt_2m_gauss_'+str(int(index1))+'.csv'
        gg = ac.get_gsds_from_mysql_or_csv()
        gsds_filename = gg.get_record_from_csv(output_file_csv)
        lis_simu = gg.record['simu_index'].values
        list_seed = gg.record['seed'].values
        lcrs = gg.record['lcr'].values
        pdata = ac.get_a_gsd_from_setup()
        cn4s = np.zeros((len(gsds_filename),))
        for i in range(len(gsds_filename)):
            # if list_seed[i]==0:
            pdata.get_gsd_data_from_filename(gsds_filename[i])
            gdata = ac.get_data_from_a_gsd_frame(pdata.gsd_data[-1])
            cn4 = gdata.get_cn_k_from_a_gsd_frame(tune_dis=3*lcrs[i], k=4)
            cn4s[i] = cn4
            # else:
            #    cn4s[i] = -1
        gg.record['cn4'] = cn4s
        gg.record['U_eq'] = gg.record['trap_gauss_epsilon'].values*0.86466  # *0.99613
        import pandas as pd
        # to strip 'unnamed:0' column which contains [1,2,3...]
        list_col = gg.record.columns.values[1:]
        pd.DataFrame.to_csv(gg.record[list_col], output_file_csv)

    def get_cn3s_from_csv_files_type_3(self):
        R"""
        import symmetry_transformation_v4_3.list_code_analysis as lca
        agf = lca.analyze_a_series_of_gsd_file()
        agp = agf.get_cn3s_from_csv_files_type_3()
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_pin/'
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'
        gg = ac.get_gsds_from_mysql_or_csv()
        gsds_filename = gg.get_record_from_csv(output_file_csv)
        list_seed = gg.record['seed'].values
        lcrs = gg.record['lcr'].values
        pdata = ac.get_a_gsd_from_setup()
        cn3s = np.zeros((len(gsds_filename),))
        for i in range(len(gsds_filename)):
            if list_seed[i] < 1:
                pdata.get_gsd_data_from_filename(gsds_filename[i])
                gdata = ac.get_data_from_a_gsd_frame(pdata.gsd_data[-1])
                cn3 = gdata.get_cn_k_from_a_gsd_frame(tune_dis=3*lcrs[i])
                cn3s[i] = cn3
            else:
                cn3s[i] = -1
        gg.record['cn3'] = cn3s
        gg.record['U_eq'] = gg.record['trap_gauss_epsilon'].values*0.86466  # *0.99613
        import pandas as pd
        # to strip 'unnamed:0' column which contains [1,2,3...]
        list_col = gg.record.columns.values[1:]
        pd.DataFrame.to_csv(gg.record[list_col], output_file_csv)

    def get_cn3s_from_csv_files_type_3_part(self):
        R"""
            import symmetry_transformation_v4_3.list_code_analysis as lca
            asg = lca.analyze_a_series_of_gsd_file()
            asg.get_cn3s_from_csv_files_type_3_part()
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_part_pin/'
        output_file_csv = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
        gg = ac.get_gsds_from_mysql_or_csv()
        gsds_filename = gg.get_record_from_csv(output_file_csv)
        lis_simu = gg.record['simu_index'].values
        list_seed = gg.record['seed'].values
        lcrs = gg.record['lcr'].values
        pdata = ac.get_a_gsd_from_setup()
        cn3s = np.zeros((len(gsds_filename),))
        for i in range(len(gsds_filename)):
            if list_seed[i] < 1 and lis_simu[i] > 6373:
                pdata.get_gsd_data_from_filename(gsds_filename[i])
                gdata = ac.get_data_from_a_gsd_frame(pdata.gsd_data[-1])
                cn3 = gdata.get_cn_k_from_a_gsd_frame(tune_dis=3*lcrs[i], k=3)
                cn3s[i] = cn3
            else:
                cn3s[i] = -1
        gg.record['cn3'] = cn3s
        gg.record['U_eq'] = gg.record['trap_gauss_epsilon'].values*0.86466  # *0.99613
        import pandas as pd
        # to strip 'unnamed:0' column which contains [1,2,3...]
        list_col = gg.record.columns.values[1:]
        pd.DataFrame.to_csv(gg.record[list_col], output_file_csv)

    def get_cn3s_from_two_csv_files(self):
        import symmetry_transformation_v4_3.analysis_controller as ac
        # import symmetry_transformation_v4_3.simulation_controller as sc
        # sct = sc.simulation_controller_honeycomb_part_traps()
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_part_pin/'
        index1 = 6013
        output_file_csv = prefix_write + \
            'pin_hex_to_honeycomb_part_klt_2m_gauss_'+str(int(index1))+'_not8.csv'
        output_file_csv2 = prefix_write + \
            'pin_hex_to_honeycomb_part_klt_2m_gauss_b_'+str(int(index1))+'.csv'
        output_file_csv3 = prefix_write + \
            'pin_hex_to_honeycomb_part_klt_2m_gauss_'+str(int(index1))+'_09.csv'
        import proceed_file as pf
        cm = pf.merge_two_csvs(output_file_csv, output_file_csv2, output_file_csv3)
        # print(cm.csv_merged.shape)
        # sct.generate_initial_state_hexagonal_particle_honeycomb_part_trap_scan_csv(output_file_csv)
        gg = ac.get_gsds_from_mysql_or_csv()
        gsds_filename = gg.get_record_from_csv(output_file_csv3)
        """lcrs = gg.record['lcr'].values
        ks = gg.record['trap_gauss_epsilon'].values"""
        pdata = ac.get_a_gsd_from_setup()
        cn3s = np.zeros((len(gsds_filename),))
        for i in range(len(gsds_filename)):
            pdata.get_gsd_data_from_file(gsds_filename[i])
            gdata = ac.get_data_from_a_gsd_frame(pdata.gsd_data[-1])
            cn3 = gdata.get_cn_k_from_a_gsd_frame()
            cn3s[i] = cn3
        gg.record['cn3'] = cn3s
        import pandas as pd
        # to strip 'unnamed:0' column which contains [1,2,3...]
        list_col = gg.record.columns.values[1:]
        pd.DataFrame.to_csv(gg.record[list_col], output_file_csv3)

    def plot_from_csv(self):
        import pandas as pd
        """prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_pin/'
        csv_filename = prefix_write + 'honeycomb_pin_scan_k3_1000.csv'"""
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_part_pin/'
        csv_filename = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_6013_09.csv'
        df = pd.read_csv(csv_filename)
        df.sort_values(by=['seed', 'trap_gauss_epsilon'], ascending=False,
                       inplace=True)  # new sorted df replace old unsorted df
        # print(df.head(50))#df['seed']

        df_seed = df['seed'].values
        ar_seed = np.array(df_seed, dtype=int)
        list_seeds = np.unique(ar_seed)

        n_seed = len(list_seeds)
        n_sim = int(len(df)/n_seed)
        record_cn3s = np.zeros((n_sim, n_seed))
        record_k_cn3_std = np.zeros((n_sim, 3))
        for seed in list_seeds:  # seed=0#
            new_df = df[df['seed'] == seed]
            cn3 = np.array(new_df['cn3'].values)  # cn3=[0.1,0.5,0.9]#
            if seed == 0:
                k = np.array(-new_df['trap_gauss_epsilon'].values)  # k=[1,2,3]#
            record_cn3s[:, seed] = cn3
        record_k_cn3_std[:, 0] = k
        record_k_cn3_std[:, 1] = np.average(record_cn3s, axis=1)
        record_k_cn3_std[:, 2] = np.std(record_cn3s, axis=1)

        # plot 10 seeds
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # ,label=str(seed) x and y shoundn't be larger than 10 times, such as 10 vs 1, or plot will be invisible
        ax.semilogx(k, record_k_cn3_std[:, 1])
        plt.errorbar(k, record_k_cn3_std[:, 1], yerr=record_k_cn3_std[:, 2])
        plt.show()
        print('ok')
        """
        #plot 10 seeds
        import matplotlib.pyplot as plt
        #import matplotlib
        #matplotlib.use(backend="QtAgg")#Backend agg is non-interactive backend. Turning interactive mode off. 'QtAgg' is interactive mode
        fig,ax = plt.subplots()
        for seed in list_seeds:#seed=0#
            new_df = df[df['seed']==seed]
            k = np.array(-new_df['trap_gauss_epsilon'].values)#k=[1,2,3]#
            cn3 = np.array(new_df['cn3'].values)#cn3=[0.1,0.5,0.9]#
            #,label=str(seed) x and y shoundn't be larger than 10 times, such as 10 vs 1, or plot will be invisible
            ax.semilogx(k,cn3)
        #ax.set_aspect('equal','box')
        plt.legend()
        plt.show()
        print('ok')"""

    def get_k_cn3_stds_from_csv(self, csv_filename):
        import pandas as pd
        df = pd.read_csv(csv_filename)
        df.sort_values(by=['seed', 'trap_gauss_epsilon'], ascending=False,
                       inplace=True)  # new sorted df replace old unsorted df
        # print(df.head(50))#df['seed']

        df_seed = df['seed'].values
        ar_seed = np.array(df_seed, dtype=int)
        list_seeds = np.unique(ar_seed)

        n_seed = len(list_seeds)
        n_sim = int(len(df)/n_seed)
        record_cn3s = np.zeros((n_sim, n_seed))
        record_k_cn3_std = np.zeros((n_sim, 3))
        for seed in list_seeds:  # seed=0#
            new_df = df[df['seed'] == seed]
            cn3 = np.array(new_df['cn3'].values)  # cn3=[0.1,0.5,0.9]#
            if seed == 0:
                k = np.array(-new_df['trap_gauss_epsilon'].values)  # k=[1,2,3]#
            record_cn3s[:, seed] = cn3
        record_k_cn3_std[:, 0] = k
        record_k_cn3_std[:, 1] = np.average(record_cn3s, axis=1)
        record_k_cn3_std[:, 2] = np.std(record_cn3s, axis=1)
        # get averaged and std data
        # sta_df = df[df['seed']==0]

        return record_k_cn3_std

    def get_k_cn3_stds_csv_from_csv(self, csv_filename):
        import pandas as pd
        df = pd.read_csv(csv_filename)
        df.sort_values(by=['seed', 'trap_gauss_epsilon'], ascending=False,
                       inplace=True)  # new sorted df replace old unsorted df
        # print(df.head(50))#df['seed']

        df_seed = df['seed'].values
        ar_seed = np.array(df_seed, dtype=int)
        list_seeds = np.unique(ar_seed)

        n_seed = len(list_seeds)
        n_sim = int(len(df)/n_seed)
        record_cn3s = np.zeros((n_sim, n_seed))
        for seed in list_seeds:  # seed=0#
            new_df = df[df['seed'] == seed]
            cn3 = np.array(new_df['cn3'].values)  # cn3=[0.1,0.5,0.9]#
            if seed == 0:
                k = np.array(-new_df['trap_gauss_epsilon'].values)  # k=[1,2,3]#
            record_cn3s[:, seed] = cn3
        df_single_seed = df[df['seed'] == 0]
        df_single_seed['U_eq'] = k*0.39347  # sigma=1,rcut=1, equalvalence U trap
        df_single_seed['cn3avg'] = np.average(record_cn3s, axis=1)
        df_single_seed['cn3std'] = np.std(record_cn3s, axis=1)

        pd.DataFrame.to_csv(df_single_seed, 'pin_hex_to_honeycomb_klt_2m_gauss_5773_5812_res.csv')
        # 'pin_hex_to_honeycomb_part_klt_2m_gauss_6013_09_res.csv'

    def add_cn3_csv_from_csv(self, csv_filename):
        import pandas as pd
        df = pd.read_csv(csv_filename)
        df.sort_values(by=['seed', 'trap_gauss_epsilon'], ascending=False,
                       inplace=True)  # new sorted df replace old unsorted df
        # print(df.head(50))#df['seed']

        df_seed = df['seed'].values
        ar_seed = np.array(df_seed, dtype=int)
        list_seeds = np.unique(ar_seed)

        n_seed = len(list_seeds)
        n_sim = int(len(df)/n_seed)
        record_cn3s = np.zeros((n_sim, n_seed))
        for seed in list_seeds:  # seed=0#
            new_df = df[df['seed'] == seed]
            cn3 = np.array(new_df['cn3'].values)  # cn3=[0.1,0.5,0.9]#
            if seed == 0:
                k = np.array(-new_df['trap_gauss_epsilon'].values)  # k=[1,2,3]#
            record_cn3s[:, seed] = cn3
        df_single_seed = df[df['seed'] == 0]
        df_single_seed['U_eq'] = k*0.39347  # sigma=1,rcut=1, equalvalence U trap
        df_single_seed['cn3avg'] = np.average(record_cn3s, axis=1)
        df_single_seed['cn3std'] = np.std(record_cn3s, axis=1)

        pd.DataFrame.to_csv(df_single_seed, 'pin_hex_to_honeycomb_klt_2m_gauss_5773_5812_res.csv')
        # 'pin_hex_to_honeycomb_part_klt_2m_gauss_6013_09_res.csv'

    def plot_k_cn3_stds(self, fig, ax, record_k_cn3_std, label=None):
        import matplotlib.pyplot as plt
        # fig,ax = plt.subplots()
        # ,label=str(seed) x and y shoundn't be larger than 10 times, such as 10 vs 1, or plot will be invisible
        ax.semilogx(record_k_cn3_std[:, 0], record_k_cn3_std[:, 1])
        ax.errorbar(
            record_k_cn3_std[:, 0],
            record_k_cn3_std[:, 1],
            yerr=record_k_cn3_std[:, 2],
            capsize=6, label=label)
        return fig, ax

    def get_2_k_cn3_stds_and_plot(self):
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_pin/'
        csv_filename = prefix_write + 'honeycomb_pin_scan_k3_1000.csv'
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_part_pin/'
        csv_filename2 = prefix_write + 'pin_hex_to_honeycomb_part_klt_2m_gauss_6013_09.csv'
        record_k_cn3_std1 = self.get_k_cn3_stds_from_csv(csv_filename)
        record_k_cn3_std2 = self.get_k_cn3_stds_from_csv(csv_filename2)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig, ax = self.plot_k_cn3_stds(fig, ax, record_k_cn3_std1, 'hc')
        fig, ax = self.plot_k_cn3_stds(fig, ax, record_k_cn3_std2, 'hp')
        plt.legend()
        plt.show()

    def get_bonds_from_simu_indices_type_n(self, index0, seed0=9):
        R"""
            lcr0.81 honeycomb pin
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        # prefix_write = '/home/remote/xiaotian_file/link_to_HDD//hoomd-examples_0/'
        # index0 = 7223  # 6302
        for i in range(10):
            index1 = index0+i  # 658#
            # gsd_file = prefix_write + 'trajectory_auto'+str(int(index1))+'_9.gsd'
            ggsd = ac.get_a_gsd_from_setup()
            ggsd.set_file_parameters(index1, seed0)
            ggsd.get_gsd_data_from_file()
            frame = ggsd.gsd_data[-1]

            gf = ac.get_data_from_a_gsd_frame(frame)  # error:missing last frame
            if i < 5:
                gf.get_bonds_png_from_a_gsd_frame(
                    'depin_from_type_'+str(6)+'_part_special_'+str(index1)+'.png')  # 4+i
            else:
                gf.get_bonds_png_from_a_gsd_frame(
                    'depin_from_type_'+str(7)+'_part_special_'+str(index1)+'.png')  # 4+i
            del gf

    def get_bonds_from_simu_indices_type_n(self, index0, seed0=9):
        R"""
            lcr0.81 honeycomb pin
        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        # prefix_write = '/home/remote/xiaotian_file/link_to_HDD//hoomd-examples_0/'
        # index0 = 7223  # 6302
        for i in range(10):
            index1 = index0+i  # 658#
            # gsd_file = prefix_write + 'trajectory_auto'+str(int(index1))+'_9.gsd'
            ggsd = ac.get_a_gsd_from_setup()
            ggsd.set_file_parameters(index1, seed0)
            ggsd.get_gsd_data_from_file()
            frame = ggsd.gsd_data[-1]

            gf = ac.get_data_from_a_gsd_frame(frame)  # error:missing last frame
            if i < 5:
                gf.get_bonds_png_from_a_gsd_frame(
                    'depin_from_type_'+str(6)+'_part_special_'+str(index1)+'.png')  # 4+i
            else:
                gf.get_bonds_png_from_a_gsd_frame(
                    'depin_from_type_'+str(7)+'_part_special_'+str(index1)+'.png')  # 4+i
            del gf

    def get_bonds_from_simu_indices_list_type_n(self, list_index, list_seed=9, list_lcra=2.4):
        R"""

        """
        import symmetry_transformation_v4_3.analysis_controller as ac
        # list_index = [7001, 7052, 7112, 7137, 7141]
        for i in range(len(list_index)):  # range(5):
            index1 = list_index[i]
            seed1 = list_seed[i]
            lcra1 = list_lcra[i]
            # gsd_file = prefix_write + 'trajectory_auto'+str(int(index1))+'_9.gsd'
            ggsd = ac.get_a_gsd_from_setup()
            ggsd.set_file_parameters(index1, seed1)
            ggsd.get_gsd_data_from_file()
            frame = ggsd.gsd_data[-1]

            gf = ac.get_data_from_a_gsd_frame(frame)  # error:missing last frame
            gf.get_bonds_png_from_a_gsd_frame('bond_'+str(index1)+'_'+str(seed1)+'.png', lcra1)
            del gf

    def get_bonds_from_simu_indices_type_n_from_csv(self):
        R"""
            column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']
            manually set bond_length_max = 3*lcr0*1.2, set 1.2 is to avoid 1,414 and 1.73 bond.
        exp:
            import symmetry_transformation_v4_3.list_code_analysis as lca
            agf = lca.analyze_a_series_of_gsd_file()
            agp = agf.get_bonds_from_simu_indices_type_n_from_csv()
        """
        import pandas as pd
        import symmetry_transformation_v4_3.analysis_controller as ac
        prefix_csv = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_depin/'  # type_n_depin/
        filename_csv = prefix_csv + 'depin_type_n_from_type_n_part_klt_2m_gauss_7163.csv'
        # 'pin_hex_to_type_n_part_klt_2m_gauss_6613.csv'#
        # depin_type_n_from_type_n_part_klt_2m_gauss_7163,6053,6293
        record_csv = pd.read_csv(filename_csv)
        list_type_n_to_watch = [4, 5, 6, 9]  # [7,10,11]
        ggsd = ac.get_a_gsd_from_setup()
        for type_n in list_type_n_to_watch:
            record_type_n = record_csv[record_csv['type_n'] == type_n]
            list_index = record_type_n['simu_index'].values
            lcr0 = record_type_n['lcr'].values[0]
            list_trap_gauss_epsilon = record_type_n['trap_gauss_epsilon'].values
            for i in range(len(list_index)):
                # gsd_file = prefix_write + 'trajectory_auto'+str(int(index1))+'_9.gsd'
                ggsd.set_file_parameters(list_index[i], 9)
                # ggsd.get_gsd_data_from_file()
                frame = ggsd.gsd_data[-1]
                gf = ac.get_data_from_a_gsd_frame(frame)  # error:missing last frame
                # gf.get_given_bonds_png_from_a_gsd_frame('pin_hex_to_type_type_'+str(type_n)+'_part_'+str(-int(list_trap_gauss_epsilon[i]))+'_.png',3*lcr0*1.2)
                gf.get_given_bonds_png_from_a_gsd_frame(
                    'depin_from_type_' + str(type_n) + '_part_' +
                    str(-int(list_trap_gauss_epsilon[i])) + '_.png', 3 * lcr0 * 1.2)
                del gf

    def get_bonds_from_simu_indices_type_11_from_csv(self):
        R"""
            column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']
            manually set bond_length_max = 3*lcr0*1.2, set 1.2 is to avoid 1,414 and 1.73 bond.
        exp:
            import symmetry_transformation_v4_3.list_code_analysis as lca
            agf = lca.analyze_a_series_of_gsd_file()
            agp = agf.get_bonds_from_simu_indices_type_10_from_csv()
        """
        import pandas as pd
        import symmetry_transformation_v4_3.analysis_controller as ac
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/'
        index1 = 6733  # index1 = 6643
        output_file_csv = prefix_write + \
            'pin_hex_to_type_11_part_klt_2m_gauss_'+str(int(index1))+'.csv'
        # depin_type_n_from_type_n_part_klt_2m_gauss_6053,6293
        record_csv = pd.read_csv(output_file_csv)
        ggsd = ac.get_a_gsd_from_setup()
        record_type_n = record_csv[record_csv['cn5'] > 0.2]
        list_index = record_type_n['simu_index'].values
        lcr0 = record_type_n['lcr'].values  # [0]
        list_trap_gauss_epsilon = record_type_n['trap_gauss_epsilon'].values
        for i in range(len(list_index)):
            # gsd_file = prefix_write + 'trajectory_auto'+str(int(index1))+'_9.gsd'
            ggsd.set_file_parameters(list_index[i], 9)
            # ggsd.get_gsd_data_from_file()
            frame = ggsd.gsd_data[-1]
            gf = ac.get_data_from_a_gsd_frame(frame)  # error:missing last frame
            gf.get_given_bonds_png_from_a_gsd_frame(
                'pin_hex_to_type_type_11_part_'+str(int(list_index[i]))+'_9_.png', 3*lcr0[i]*1.2)
            # gf.get_given_bonds_png_from_a_gsd_frame('depin_from_type_'+str(type_n)+'_part_'+str(-int(list_trap_gauss_epsilon[i]))+'_.png',3*lcr0*1.2)
            del gf

    def get_bonds_from_simu_indices_type_10_from_csv(self):
        R"""
            column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']
            manually set bond_length_max = 3*lcr0*1.2, set 1.2 is to avoid 1,414 and 1.73 bond.
        exp:
            import symmetry_transformation_v4_3.list_code_analysis as lca
            agf = lca.analyze_a_series_of_gsd_file()
            agp = agf.get_bonds_from_simu_indices_type_10_from_csv()
        """
        import pandas as pd
        import symmetry_transformation_v4_3.analysis_controller as ac
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/'
        index1 = 6643  # index1 = 6733#
        output_file_csv = prefix_write + \
            'pin_hex_to_type_10_part_klt_2m_gauss_'+str(int(index1))+'.csv'
        # depin_type_n_from_type_n_part_klt_2m_gauss_6053,6293
        record_csv = pd.read_csv(output_file_csv)
        ggsd = ac.get_a_gsd_from_setup()
        record_type_n = record_csv[record_csv['cn5'] > 0.8]
        list_index = record_type_n['simu_index'].values
        lcr0 = record_type_n['lcr'].values  # [0]
        list_trap_gauss_epsilon = record_type_n['trap_gauss_epsilon'].values
        for i in range(len(list_index)):
            # gsd_file = prefix_write + 'trajectory_auto'+str(int(index1))+'_9.gsd'
            ggsd.set_file_parameters(list_index[i], 9)
            # ggsd.get_gsd_data_from_file()
            frame = ggsd.gsd_data[-1]
            gf = ac.get_data_from_a_gsd_frame(frame)  # error:missing last frame
            gf.get_given_bonds_png_from_a_gsd_frame(
                'pin_hex_to_type_type_10_part_'+str(int(list_index[i]))+'_9_.png', 3*lcr0[i]*1.2)
            # gf.get_given_bonds_png_from_a_gsd_frame('depin_from_type_'+str(type_n)+'_part_'+str(-int(list_trap_gauss_epsilon[i]))+'_.png',3*lcr0*1.2)
            del gf

    def get_bonds_from_simu_indices_type_7_from_csv(self):
        R"""
            column = ['simu_index','seed','lcr','trap_gauss_epsilon','temperature','type_n']
            manually set bond_length_max = 3*lcr0*1.2, set 1.2 is to avoid 1,414 and 1.73 bond.
        exp:
            import symmetry_transformation_v4_3.list_code_analysis as lca
            agf = lca.analyze_a_series_of_gsd_file()
            agp = agf.get_bonds_from_simu_indices_type_7_from_csv()
        """
        import pandas as pd
        import symmetry_transformation_v4_3.analysis_controller as ac
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/'
        # index1 = 6643#index1 = 6733#
        # +str(int(index1))+
        output_file_csv = prefix_write + 'pin_hex_to_type_7_part_klt_2m_gauss_merge.csv'
        # depin_type_n_from_type_n_part_klt_2m_gauss_6053,6293
        record_csv = pd.read_csv(output_file_csv)
        ggsd = ac.get_a_gsd_from_setup()
        record_type_n = record_csv[record_csv['cn4'] > 0.74]
        list_index = record_type_n['simu_index'].values
        lcr0 = record_type_n['lcr'].values  # [0]
        list_trap_gauss_epsilon = record_type_n['trap_gauss_epsilon'].values
        for i in range(len(list_index)):
            # gsd_file = prefix_write + 'trajectory_auto'+str(int(index1))+'_9.gsd'
            ggsd.set_file_parameters(list_index[i], 9)
            # ggsd.get_gsd_data_from_file()
            frame = ggsd.gsd_data[-1]
            gf = ac.get_data_from_a_gsd_frame(frame)  # error:missing last frame
            gf.get_given_bonds_png_from_a_gsd_frame(
                'pin_hex_to_type_7_part_'+str(int(list_index[i]))+'_9_.png', 3*lcr0[i]*1.2)
            # gf.get_given_bonds_png_from_a_gsd_frame('depin_from_type_'+str(type_n)+'_part_'+str(-int(list_trap_gauss_epsilon[i]))+'_.png',3*lcr0*1.2)
            del gf

    def analyze_gsd_files_and_record_as_csv(self):
        import pandas as pd
        prefix_write = "/home/remote/xiaotian_file/link_to_HDD//record_results_v430/honeycomb_pin/"

        output_file_csv = prefix_write+'honeycomb_pin_scan_k100_1000'+'.csv'
        n_file = 10
        list_index = np.linspace(5793, 5802, n_file, dtype=int)
        list_lcr = [0.81]*n_file
        list_trap_gauss_epsilon = np.linspace(-100, -1000, n_file, dtype=int)
        list_cn3 = np.zeros((n_file,))
        # df_all = pd.DataFrame()
        for seed in [0, 1, 2]:  # ,3,4,7,8,9
            # seed = 7
            list_seed_record = [seed]*n_file
            for i in range(n_file):
                simu_index = list_index[i]
                s2g = ac.get_a_gsd_from_setup()
                s2g.set_file_parameters(simu_index, seed)

                g2d = ac.get_data_from_a_gsd_frame()
                list_cn3[i] = g2d.get_cn_k_from_a_gsd_frame(s2g.gsd_data[-1])

            df = pd.DataFrame(list_index, columns=['simu_index'])
            df['seed'] = list_seed_record
            df['lcr'] = list_lcr
            df['trap_gauss_epsilon'] = list_trap_gauss_epsilon
            df['cn3'] = list_cn3
            if seed == 0:
                df_all = df
            elif seed > 0:
                df_all = pd.concat([df_all, df])  # .append(df)
        pd.DataFrame.to_csv(df_all, output_file_csv)


class analyze_a_series_of_gsd_file_dynamic:
    def __init__(self):
        pass

    def get_polygon_from_csv(self):
        R"""
        csvs = ['/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv',
        '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv']
        parts = [True,False,True,False]
        types = [3,3,8,8]
        import getDataAndDiagramCsv as dac
        for i in range(4):
            dac.get_diagram_from_csv_type_n(csvs[i],types[i],parts[i])
        """
        # prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/'
        # index1 = 6733#index1 = 6643
        output_file_csv = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
        # depin_type_n_from_type_n_part_klt_2m_gauss_6053,6293
        # record_csv = pd.read_csv(output_file_csv)
        ggsd = ac.get_a_gsd_from_setup()
        """list_index = record_type_n['simu_index'].values
        lcr0 = record_type_n['lcr'].values#[0]
        list_trap_gauss_epsilon = record_type_n['trap_gauss_epsilon'].values"""
        list_index = [632, 370, 6521, 129]  # kgp,kg,hp,h
        list_seed = [9, 2, 8, 0]
        list_lcr = [0.86, 0.87, 0.81, 0.81]

        for i in range(len(list_index)):
            # gsd_file = prefix_write + 'trajectory_auto'+str(int(index1))+'_9.gsd'
            ggsd.set_file_parameters(list_index[i], list_seed[i], False)
            dt = ac.get_data_from_a_trajectory(list_index[i], list_seed[i])  # ,ggsd.input_file_gsd
            dt.get_polygon(3*list_lcr[i])

    def get_cnks_vs_t_from_csv(self):
        import pandas as pd
        import symmetry_transformation_v4_3.analysis_controller as ac
        ggsd = ac.get_a_gsd_from_setup()
        list_index = [632, 370, 6521, 129]  # kgp,kg,hp,h
        list_seed = [9, 2, 8, 0]
        list_lcr = [0.86, 0.87, 0.81, 0.81]
        import workflow_analysis as wa
        ck = wa.show_cn_k()
        for i in range(len(list_index)):
            # gsd_file = prefix_write + 'trajectory_auto'+str(int(index1))+'_9.gsd'
            ggsd.set_file_parameters(list_index[i], list_seed[i], False)
            dt = ac.get_data_from_a_trajectory(list_index[i], list_seed[i])  # ,ggsd.input_file_gsd
            csv_filename = dt.directory_to_trajectory_data + 'cnks.csv'
            dt.get_cnks(3*list_lcr[i], csv_filename)
            df = pd.read_csv(csv_filename)
            str_index = str(list_index[i])+'_'+str(list_seed[i])
            record = df.values[:, 1:]  # to remove the 0th unnamed column
            ck.show_cn_k(0, record, dt.directory_to_trajectory_data, str_index)
            ck.show_cn_k_logt(0, record, dt.directory_to_trajectory_data, str_index)

    def get_transformation_velocity_scan_csv_read(
            self, csv_filename=None, cnk=3, index_uplimit=9999):
        R"""

        """
        # series setup
        import proceed_file as pf
        import points_analysis_2D as pa
        import data_analysis_cycle as dac
        import symmetry_transformation_v4_3.simulation_core as sc
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/record_trajectories'
        prefix_read = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430'
        dir_h = '/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'
        dir_hp = '/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
        dir_kg = '/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv'
        dir_kgp = '/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv'
        # gsd_reocrd_location: example_1,1,1,1; dir_hp moved to exple_1 and remain only seed1-9
        dirs = [dir_h, dir_hp, dir_kg, dir_kgp]
        # print(dirs)
        if csv_filename is None:
            csv_filename = prefix_read+dirs[0]
        df = pd.read_csv(csv_filename)
        ggsd = ac.get_a_gsd_from_setup()

        n_size = [16, 8]
        a_particle = 3
        list_simu_index = df['simu_index'].values
        list_seed = df['seed'].values
        list_lcr = df['lcr'].values
        list_trap_gauss_epsilon = df['trap_gauss_epsilon'].values
        list_temperature = df['temperature'].values
        list_type_n = df['type_n'].values

        pfs = pf.proceed_file_shell()
        # start = 150
        # record cn vca
        rvv = -np.ones((len(list_simu_index), 2))  # np.zeros((240, 2))
        for i in range(len(list_simu_index)):
            # i >= 0 and i <= 269:  # i >= 240 and i <= 479:  # i >= 0 and i <= 239:  # i >= start and i <= start+29:
            if i <= index_uplimit:
                simu_index = list_simu_index[i]
                seed = list_seed[i]
                lcr = list_lcr[i]
                # ggsd.set_file_parameters(list_simu_index[i], list_seed[i], False)
                # create a folder to store data for a simu_index_seed
                # temp, str_simu_index = pfs.create_prefix_results(simu_index=simu_index, seed=seed)
                # gdf = ac.get_data_from_a_trajectory(simu_index, seed)
                # file_txy = gdf.directory_to_trajectory_data+'txyz.npy'
                # if not os.path.exists(file_txy):
                # else:
                #    gdf = ac.get_data_from_a_trajectory(simu_index, seed)
                #
                # prefix_new = pfs.create_folder(prefix_write+'/', str_simu_index)
                gdf = ac.get_data_from_a_trajectory(simu_index, seed)
                """
                prefix_write = "/home/remote/xiaotian_file/link_to_HDD//hoomd-examples_1/"
                output_file_gsd = prefix_write+'trajectory_auto' + \
                    str(int(simu_index))+'_'+str(int(seed))+'.gsd'
                # sct = sc.simulation_core_traps(simu_index, seed)
                gsd_filename = output_file_gsd  # sct.output_file_gsd
                gdf = ac.get_data_from_a_trajectory(simu_index, seed, gsd_filename=gsd_filename)

                dtt = pf.proceed_gsd_file(filename_gsd_seed=ggsd.input_file_gsd)
                file_txy = prefix_new+'txyz.npy'
                if not os.path.exists(file_txy):
                    dtt.get_trajectory_data_with_traps(prefix_new)
                txyz = np.load(file_txy)"""
                # record_filename = prefix_new + 'T_VS_CN_k_cut'+'index'+str_simu_index+'.csv'
                record_filename = gdf.directory_to_trajectory_data+'T_VS_CN_k_cut'+'index'+gdf.str_simu_index+'.csv'
                if not os.path.exists(record_filename):  # True:  #
                    gdf.get_cnks(3 * lcr, record_filename, cn_final_check=cnk)  # , True)
                tcnk = pd.read_csv(record_filename)
                tcnk = tcnk.values[:, 1:]  # the 1st column is just data index
                pda = pa.dynamic_points_analysis_2d(gdf.txyz)
                # record_filename = prefix_new + 'T_VS_CN_k_cut'+'index'+str_simu_index+'.npy'
                # print(record_filename)
                # if not os.path.exists(record_filename):
                """record_filename = dac.save_from_txyz_to_cn3(
                    simu_index, seed, coordination_number=True, lattice_constant=3 * lcr,
                    prefix=prefix_new,txyz=txyz,final_cut=True)

                ccn = dac.save_from_txyz_to_cn3_f(
                    simu_index, seed, coordination_number=True, lattice_constant=3 * lcr,
                    prefix=prefix_new, txyz=txyz, final_cut=True)
                """
                # dac.save_t_cnk_png(tcnk, gdf.directory_to_trajectory_data,
                #                   gdf.str_simu_index)
                # print(f"simu_index:{simu_index},seed:{seed},saturate_value:{tcnk[-1,cnk]}")
                # tcnk = np.load(record_filename)
                st, avv = pda.get_saturate_time_of_tanh_like_data(tcnk[:, cnk])
                rvv[i] = [st, avv]
                print(
                    f"simu_index:{simu_index},seed:{seed},saturate_value:{tcnk[-1, cnk]},saturate_time:{st},averaged_velocity:{avv}")
                print(i+1, '/', len(list_simu_index))
        df['saturate_value'] = rvv[:, 0]
        df['averaged_velocity'] = rvv[:, 1]
        df_sub = df.iloc[:index_uplimit]  # df[df['seed'] == 0]
        df_sub2 = df_sub[['rho_trap_relative', 'U_eq', 'saturate_value', 'averaged_velocity']]
        df_sub2.to_csv(gdf.work_space+'trans_velocity'+'_index_'+str(list_simu_index[0])+'.csv')
        """
        df_sub.sort_index(ascending=True, inplace=True)
        dfr = pd.DataFrame(rvv, columns=['saturate_value', 'averaged_velocity'])
        dfr.to_csv(gdf.work_space+'trans_velocity'+'_index_'+str(list_simu_index[0])+'.csv')
        """

    def get_transformation_velocity_scan_csv(self, csv_filename=None, cnk=3, index_uplimit=9999):
        R"""

        """
        # series setup
        import proceed_file as pf
        import points_analysis_2D as pa
        import data_analysis_cycle as dac
        import symmetry_transformation_v4_3.simulation_core as sc
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430/record_trajectories'
        prefix_read = '/home/remote/xiaotian_file/link_to_HDD/record_results_v430'
        dir_h = '/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'
        dir_hp = '/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
        dir_kg = '/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv'
        dir_kgp = '/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv'
        # gsd_reocrd_location: example_1,1,1,1; dir_hp moved to exple_1 and remain only seed1-9
        dirs = [dir_h, dir_hp, dir_kg, dir_kgp]
        # print(dirs)
        if csv_filename is None:
            csv_filename = prefix_read+dirs[0]
        df = pd.read_csv(csv_filename)
        ggsd = ac.get_a_gsd_from_setup()

        list_simu_index = df['simu_index'].values
        list_seed = df['seed'].values
        list_lcr = df['lcr'].values

        # start = 150
        # record cn vca
        rvv = -np.ones((len(list_simu_index), 2))  # np.zeros((240, 2))
        # (6392, 6602, 8, dtype=int)  # temp
        target_simu_index = np.linspace(532, 772, 9, dtype=int)  # (532, 772, 9, dtype=int)
        for i in range(len(list_simu_index)):
            # i >= 0 and i <= 269:  # i >= 240 and i <= 479:  # i >= 0 and i <= 239:  # i >= start and i <= start+29:
            if i >= 0 and i <= index_uplimit:
                simu_index = list_simu_index[i]
                if simu_index in target_simu_index:  # temp
                    seed = list_seed[i]
                    lcr = list_lcr[i]

                    prefix_write = "/home/remote/xiaotian_file/link_to_HDD//hoomd-examples_1/"
                    output_file_gsd = prefix_write+'trajectory_auto' + \
                        str(int(simu_index))+'_'+str(int(seed))+'.gsd'

                    gsd_filename = output_file_gsd  # sct.output_file_gsd
                    gdf = ac.get_data_from_a_trajectory(
                        simu_index, seed, gsd_filename=gsd_filename, stable=True)
                    """
                    record_filename = gdf.directory_to_trajectory_data+'T_VS_CN_k_cut'+'index'+gdf.str_simu_index+'.csv'
                    if not os.path.exists(record_filename):  # True:  #
                        gdf.get_cnks(3 * lcr, record_filename, cn_final_check=cnk,
                                     bench_mark=0.1)  # , True)
                    tcnk = pd.read_csv(record_filename)
                    tcnk = tcnk.values[:, 1:]  # the 1st column is just data index
                    pda = pa.dynamic_points_analysis_2d(gdf.txyz)

                    st, avv = pda.get_saturate_time_of_tanh_like_data(tcnk[:, cnk])
                    rvv[i] = [st, avv]
                    print(
                        f"simu_index:{simu_index},seed:{seed},saturate_value:{tcnk[-1, cnk]},saturate_time:{st},averaged_velocity:{avv}")
                    """
                    print(i+1, '/', len(list_simu_index))
        df['saturate_value'] = rvv[:, 0]
        df['averaged_velocity'] = rvv[:, 1]
        df_sub = df[df['seed'] == 0]
        df_sub2 = df_sub[['rho_trap_relative', 'U_eq', 'saturate_value', 'averaged_velocity']]
        df_sub2.to_csv(gdf.work_space+'trans_velocity'+'_index_'+str(list_simu_index[0])+'.csv')
        """
        df_sub.sort_index(ascending=True, inplace=True)
        dfr = pd.DataFrame(rvv, columns=['saturate_value', 'averaged_velocity'])
        dfr.to_csv(gdf.work_space+'trans_velocity'+'_index_'+str(list_simu_index[0])+'.csv')
        """

    def get_transformation_ratio_vs_t_scan_csv_read(
            self, csv_filename=None, cnk=3, index_uplimit=9999):
        R"""

        """
        # series setup
        import proceed_file as pf
        import points_analysis_2D as pa
        import data_analysis_cycle as dac
        import symmetry_transformation_v4_3.simulation_core as sc
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/record_trajectories'
        prefix_read = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430'
        dir_h = '/honeycomb_pin/pin_hex_to_honeycomb_klt_2m_gauss_3_242.csv'
        dir_hp = '/honeycomb_part_pin/pin_hex_to_honeycomb_part_klt_2m_gauss_6373_6612.csv'
        dir_kg = '/type_n_pin/pin_hex_to_type_8_klt_2m_gauss_243.csv'
        dir_kgp = '/type_n_pin/pin_hex_to_type_8_part_klt_2m_gauss_513.csv'
        # gsd_reocrd_location: example_1,1,1,1; dir_hp moved to exple_1 and remain only seed1-9
        dirs = [dir_h, dir_hp, dir_kg, dir_kgp]
        # print(dirs)
        if csv_filename is None:
            csv_filename = prefix_read+dirs[0]
        df = pd.read_csv(csv_filename)
        ggsd = ac.get_a_gsd_from_setup()

        n_size = [16, 8]
        a_particle = 3
        list_simu_index = df['simu_index'].values
        list_seed = df['seed'].values
        list_lcr = df['lcr'].values
        list_trap_gauss_epsilon = df['trap_gauss_epsilon'].values
        list_temperature = df['temperature'].values
        list_type_n = df['type_n'].values

        pfs = pf.proceed_file_shell()
        # start = 150
        # record cn vca
        import data_decorate as dd
        ddr = dd.data_decorator()
        rvv = -np.ones((len(list_simu_index), 2))  # np.zeros((240, 2))
        target_simu_index = np.linspace(6392, 6602, 8, dtype=int)  # temp
        for i in range(len(list_simu_index)):
            # i >= 0 and i <= 269:  # i >= 240 and i <= 479:  # i >= 0 and i <= 239:  # i >= start and i <= start+29:
            if i <= index_uplimit:
                simu_index = list_simu_index[i]
                if simu_index in target_simu_index:  # temp
                    seed = list_seed[i]
                    lcr = list_lcr[i]

                    gdf = ac.get_data_from_a_trajectory(simu_index, seed)

                    record_filename = gdf.directory_to_trajectory_data+'T_VS_CN_k_cut'+'index'+gdf.str_simu_index+'.csv'
                    if True:  # not os.path.exists(record_filename):  #
                        gdf.get_cnks(3 * lcr, record_filename, cn_final_check=cnk,
                                     bench_mark=0.1)  # , True)
                    tcnk = pd.read_csv(record_filename)
                    tcnk = tcnk.values[:, 1:]  # the 1st column is just data index
                    pda = pa.dynamic_points_analysis_2d(gdf.txyz)

                    list_index, data_decorated = ddr.coarse_grainize_and_average_data_log(
                        tcnk[:, cnk], coarse_grain_to_n_points=10, navg_odd=5)
                    df_new = pd.DataFrame({'frame': list_index, 'cnk': data_decorated})
                    df_new.to_csv(
                        gdf.work_space+'trans_cnk_vs_t'+'_index_'+str(list_simu_index[0])+'.csv')
                    print(
                        f"simu_index:{simu_index},seed:{seed},saturate_value:{tcnk[-1, cnk]}")
                    print(i+1, '/', len(list_simu_index))


class analyze_a_csv_file:
    def __init__(self):
        pass

    def pin(self):
        R"""
        | simu_index | seed | lcr  | trap_gauss_epsilon | temperature | type_n | cnk | U_eq |
        """
        prefix_write = '/home/remote/xiaotian_file/link_to_HDD//record_results_v430/type_n_pin/'
        index1 = 513  # [x]
        # [x]
        output_file_csv = prefix_write + \
            'pin_hex_to_type_8_part_klt_2m_gauss_'+str(int(index1))+'.csv'
        import getDataAndDiagramCsv as gddc
        csvd = gddc.csv_data_processor(output_file_csv)
        csvd.select_single_seed(0)
        lcrs = csvd.sub_record['lcr'].values
        us = csvd.sub_record['U_eq'].values
        cn4s = csvd.sub_record['cn4s'].values
        diagramp = gddc.diagram_processor()
        diagramp.draw_diagram_scatter_oop()


def check_lcr():
    import workflow_analysis as wa
    import numpy as np
    record_lcr0 = np.zeros((11,))
    for i in range(11):
        at = wa.archimedean_tilings()
        at.generate_type_n(i+1)
        cross_lattice = np.cross(at.a1, at.a2)
        area_per_particle = cross_lattice[2]/len(at.position)
        area_hex = np.sqrt(3)/2.0
        lcr0 = np.sqrt(area_hex/area_per_particle)
        record_lcr0[i] = lcr0
        # print('type'+str(i+1)+': '+str(np.round(lcr0,4) ))
        del at
    return record_lcr0


def show_type_3_11_dual():
    import workflow_analysis as wa
    import numpy as np
    sdl = wa.show_dual_lattice()
    for i in np.linspace(3, 11, 9, dtype=int):
        sdl.show_dual_type_n_part(i, xylim=5)


def show_type_3_11_polygon_dye():
    import workflow_analysis as wa
    import numpy as np
    sdl = wa.archimedean_tilings_polygon_dye()
    for i in np.linspace(3, 11, 9, dtype=int):
        sdl.workflow_type_n_part_points_type_n_polygon(i, xylim=5, n_plus=3)


def compare_differen_gauss():
    pass
