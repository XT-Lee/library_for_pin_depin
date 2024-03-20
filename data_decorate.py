import numpy as np
import pandas as pd


class data_decorator:
    def __init__(self) -> None:
        pass

    def coarse_grainize_data_log(self, data, coarse_grain_to_n_points=10):
        R"""
        parameters:
            data: 1d array with Ndata_points (float)
            Ndata_points: num of data points
            coarse_grain_to_n_points(n2c): (int)coarse_grain_to_n_points
            list_index:
            data_decorated: n2c rows of data points. 

        caution:
            the shape of data points is not log function, 
            so fitting is of nonsense.
        """
        n2c = coarse_grain_to_n_points
        sz = np.shape(data)  # rows for Ndata
        log_max = np.log(sz[0])
        list_log_index = np.linspace(0, log_max, n2c)
        list_index = np.zeros((n2c,), dtype=int)
        list_index[1:] = np.exp(list_log_index[1:]).astype(int)
        list_index[-1] = list_index[-1]
        return list_index

    def coarse_grainize_and_average_data_log(self, data, coarse_grain_to_n_points=10, navg_odd=5):
        R"""
        parameters:
            data: 1d array with Ndata_points (float)
            Ndata_points: num of data points
            coarse_grain_to_n_points(n2c): (int)coarse_grain_to_n_points
            Navg(navg): num of data points to average, 
                positive odd integral only(1,3,5...)
            list_index:
            data_decorated: n2c rows of data points averaged with (Navg-1) neighbors. 

        return 
            list_index,
            data_decorated

        introduction:
            the 1st data point is not in need of average, but others are. 
            hence the index of last data point must be set smaller than 
            Ndata_points - (Navg-1)/2

        caution:
            the shape of data points is not log function, 
            so fitting is of nonsense.

        idea:
            maybe i can tune the navg_odd depening on the density of data at each index.
        """
        n2c = coarse_grain_to_n_points
        sz = np.shape(data)  # rows for Ndata
        log_max = np.log(sz[0])
        list_log_index = np.linspace(0, log_max, n2c)
        list_index = np.zeros((n2c,), dtype=int)
        list_index[1:] = np.exp(list_log_index[1:]).astype(int)
        list_index[-1] = list_index[-1]-(navg_odd-1)/2
        data_decorated = np.zeros((n2c,))
        # print(data[sz[0]-2:])
        for i in range(n2c):
            if i == 0:
                data_decorated[i] = data[0]
                """
                elif i==n2c-1:
                in_st = list_index[i]-(navg-1)
                #in_ed = list_index[i]
                print(i,data[in_st:])
                data_decorated[i] =np.average(data[in_st:])
                """
            else:
                in_st = list_index[i]-(navg_odd-1)/2
                in_ed = list_index[i]+(navg_odd-1)/2
                in_st = in_st.astype(int)
                in_ed = in_ed.astype(int)+1  # for numpy +1 is of necessity
                # print(i,data[in_st:in_ed])
                data_decorated[i] = np.average(data[in_st:in_ed])
        return list_index, data_decorated

    def coarse_grainize_and_average_data_log_with_dynamic_navg_odd(
            self, data, coarse_grain_to_n_points=10, navg_odd=5):
        R"""
        parameters:
            data: 1d array with Ndata_points (float)
            Ndata_points: num of data points
            coarse_grain_to_n_points(n2c): (int)coarse_grain_to_n_points
            Navg(navg): num of data points to average, 
                positive odd integral only(1,3,5...)
            list_index:
            data_decorated: n2c rows of data points averaged with (Navg-1) neighbors. 

        return 
            list_index,
            data_decorated

        introduction:
            the 1st data point is not in need of average, but others are. 
            hence the index of last data point must be set smaller than 
            Ndata_points - (Navg-1)/2

        caution:
            the shape of data points is not log function, 
            so fitting is of nonsense.

        idea:
            maybe i can tune the navg_odd depening on the density of data at each index.
        """
        pass

    def coarse_grainize_and_hist_average_data_log_with_dynamic_navg_odd(
            self, data, coarse_grain_to_n_points=10):
        n2c = coarse_grain_to_n_points
        sz = np.shape(data)  # rows for Ndata
        log_max = np.log(sz[0])
        list_log_index = np.linspace(0, log_max, n2c)
        log_interval = log_max/(n2c-1)
        list_log_interval_index = np.linspace(-0.5*log_interval, log_max-0.5*log_interval, n2c)
        list_index = np.zeros((n2c,), dtype=int)
        list_interval_index = np.zeros((n2c,), dtype=int)
        list_index[1:] = np.exp(list_log_index[1:]).astype(int)
        list_interval_index[1:] = np.exp(list_log_interval_index[1:]).astype(int)
        # list_index[-1] = list_index[-1]-(navg_odd-1)/2
        data_decorated = np.zeros((n2c,))
        # print(data[sz[0]-2:])
        for i in range(n2c):
            if i == 0:
                in_st = list_index[i]
                in_ed = list_interval_index[i+1]+1
                data_decorated[i] = np.average(data[:in_ed])
            elif i == n2c-1:
                in_st = list_interval_index[i]-1
                in_ed = list_index[i]
                # print(i, data[in_st:])
                data_decorated[i] = np.average(data[in_st:])
            else:
                in_st = list_interval_index[i]
                in_ed = list_interval_index[i+1]
                in_st = in_st.astype(int)
                in_ed = in_ed.astype(int)+1  # for numpy +1 is of necessity
                # print(i,data[in_st:in_ed])
                data_decorated[i] = np.average(data[in_st:in_ed])
        return list_index, data_decorated
