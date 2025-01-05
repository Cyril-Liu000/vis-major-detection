import scipy.stats as st
import numpy as np
import statsmodels.tsa.api as tsa
from statsmodels.tsa.arima.model import ARIMA
import math as mt
import pandas as pd
import matplotlib.pyplot as plot
import datetime
import scipy.optimize as opt
import scipy.fft as fft
import math
from matplotlib import gridspec

file_path_data_sheet = './data/data_sheet.xlsx'
file_path_tem = './data/石門站溫度資料.txt'
file_path_total_data = './data/石門水庫雨量與流量資料.xlsx'
file_path_ar = './data/日雨量溫度資料.xlsx'
file_path_riskmap_p = "./data/riskmap_data/"

class read_inflow_estimate_data:
    def __init__(self, 
                 file_path_total_data, 
                 file_path_tem, 
                 file_path_data_sheet,
                 file_path_ar):
        self.file_path = file_path_total_data
        self.file_parh_tem = file_path_tem
        self.file_path_data_sheet = file_path_data_sheet
        self.file_path_ar = file_path_ar
    
    def get_dateseries_str(self):
        file_path = self.file_path
        data = pd.read_excel(file_path)
        dataFrame = pd.DataFrame(data)
        temp = pd.Series(dataFrame['日期'])
        dateseries = []
        for i in range(len(temp)):
            dateseries.append(str(temp[i]))
        return dateseries
    
    def get_temperature_series(self):
        file_path_tem = self.file_parh_tem
        t = []    
        data  = open(file_path_tem)
        date = []
        for line in data:
            temp = line.strip("\n")
            temp2 = temp.split("\t")
    
            if temp2[1] == 'Temperature':
                continue
            else:
                t.append(float(temp2[1]))
                date.append(temp2[0])
        return t
    
    def get_precipitation_series(self):
        file_path = self.file_path
        data = pd.read_excel(file_path)
        dataFrame = pd.DataFrame(data)
    
        p_data_1 = 0.0327 * pd.Series(dataFrame['石門(mm)'], dtype='float64')
        p_data_2 = 0.1708 * pd.Series(dataFrame['霞雲(mm)'], dtype='float64')
        p_data_3 = 0.0817 * pd.Series(dataFrame['高義(mm)'], dtype='float64')
        p_data_4 = 0.1853 * pd.Series(dataFrame['嘎拉賀(mm)'], dtype='float64')
        p_data_5 = 0.0718 * pd.Series(dataFrame['玉峰(mm)'], dtype='float64')
        p_data_6 = 0.367 * pd.Series(dataFrame['鎮西堡(mm)'], dtype='float64')
        p_data_7 = 0.0905 * pd.Series(dataFrame['巴陵(mm)'], dtype='float64')
    
        precipitation_series = (p_data_1 + p_data_2 + p_data_3 + p_data_4 + p_data_5
                              + p_data_6 + p_data_7) / 10
    
        return round(precipitation_series, 5)
    
    def get_inflow_cms_series(self):
        file_path = self.file_path
        data = pd.read_excel(file_path)
        df = pd.DataFrame(data)
        inflow = df['石門水庫流量(cms)']
        return inflow
    
    def get_date_series(self):
        file_path = self.file_path
        data = pd.read_excel(file_path)
        dataFrame = pd.DataFrame(data)
        temp = pd.Series(dataFrame['日期'])
        return temp
    
    def get_monthly_rainfall_data(self):
        data_frame = self.get_rainfall_data_frame()
        data_frame = data_frame.dropna()
        data_frame = data_frame.reset_index(drop=True)
        
        output = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(len(data_frame)) :
            for j in range(1, 13):
                if data_frame['date_month'][i].month == j :
                    if data_frame['preception'][i] >= 0: 
                        output[j-1].append(data_frame['preception'][i])
                    else:
                        continue
                    
        for i in range(len(output)):
            output[i] = np.asarray(output[i], dtype = np.float64)
            
        return output
    
    def get_month_rainfall_data_frame(self):
        file_path = self.file_path_data_sheet
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        date_month = pd.Series(df['date_month'])
        prec = pd.Series(df['preception'])
        
        df_r = pd.DataFrame()
        df_r.insert(loc = 0, column = 'date_month', value = date_month)
        df_r.insert(loc = 1, column = 'preception', value = prec)
    
        return df_r
    
    def get_precipitation_data_frame(self):
        date_series = self.get_date_series()
        pre_series = self.get_precipitation_series()
        
        df = pd.DataFrame()
        df.insert(loc = 0, value = date_series, column = "date")
        df.insert(loc = 1, value = pre_series, column = "precipitation")
        return df
        
    def get_consum_data_frame(self):
        file_path = self.file_path_data_sheet 
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        date_month = pd.Series(df['date_month'])
        consum = pd.Series(df['total_consumption'])
        
        df_r = pd.DataFrame()
        df_r.insert(loc = 0, column = 'date_month', value = date_month)
        df_r.insert(loc = 1, column = 'C', value = consum)
        
        df_r = df_r.dropna()
        df_r = df_r.reset_index(drop = True)
        
        return df_r
    
    def get_storage_data_frame(self):
        file_path = self.file_path_data_sheet
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        date_month = pd.Series(df['date_month'])
        storage = pd.Series(df['storage'])
        
        df_r = pd.DataFrame()
        df_r.insert(loc = 0, column = 'date_month', value = date_month)
        df_r.insert(loc = 1, column = 'S', value = storage)
        
        df_r = df_r.dropna()
        df_r = df_r.reset_index(drop = True)
        
        return df_r
    
    def get_avaliability_data_frame(self):
        file_path = self.file_path_data_sheet
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        date_month = pd.Series(df['date_month'])
        avaliability = pd.Series(df['inflow'])
        
        df_r = pd.DataFrame()
        df_r.insert(loc = 0, column = 'date_month', value = date_month)
        df_r.insert(loc = 0, column = 'inflow', value = avaliability)
        
        df_r = df_r.dropna()
        df_r = df_r.reset_index(drop = True)
        
        return df_r
    
    def get_cta_data_frame(self):
        file_path = self.file_path_data_sheet
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        date_month = pd.Series(df['date_month'])
        avaliability = pd.Series(df['CTA'])
        
        df_r = pd.DataFrame()
        df_r.insert(loc = 0, column = 'date_month', value = date_month)
        df_r.insert(loc = 0, column = 'cta', value = avaliability)
        
        df_r = df_r.dropna()
        df_r = df_r.reset_index(drop = True)
        
        return df_r
    
    def get_tem_date_data_frame(self):
        date_series = pd.Series(self.get_date_series())
        tem_series = pd.Series(self.get_temperature_series())
        
        df = pd.DataFrame()
        df.insert(loc = 0, column = 'date_month', value = date_series)
        df.insert(loc = 1, column = 'temperature', value = tem_series)
        
        return df
    
    def get_month_temperature_data(self):
        data_frame = self.get_tem_date_data_frame()
        data_frame = data_frame.dropna()
        data_frame = data_frame.reset_index(drop = True)
        
        output = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(len(data_frame)) :
            for j in range(1,13):
                if data_frame['date_month'][i].month == j :
                    if data_frame['temperature'][i] >= 0:
                        output[j-1].append(data_frame['temperature'][i])
                    else:
                        continue
        
        for i in range(len(output)):
            
            output[i] = np.asarray(output[i], dtype = np.float64)
        return output
    
    def get_month_daily_precipitation_data(self):
        file_path = self.file_path_ar    
        data = pd.read_excel(file_path)
        df_ar = pd.DataFrame(data)
        
        df_ar = df_ar.dropna()
        df_ar = df_ar.reset_index(drop = True)
        month_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
    
        for i in range(len(df_ar)):
            for j in range(1,13):
                if df_ar['Date_series'][i].month == j:
                    month_list[j-1].append(df_ar['Daily Preception'][i])
                else:
                    continue
            
        for i in range(len(month_list)):
           month_list[i] = np.asarray(month_list[i],dtype = np.float64)
        
        return month_list
    
    def get_month_daily_precipitation_data_v2(self):
        data = self.get_month_daily_precipitation_data()
        occur_month_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
        rain_month_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(len(occur_month_list)):
            for j in range(len(data[i])):
                if data[i][j] <= 0:
                    occur_month_list[i].append(0)
                else:
                    occur_month_list[i].append(1)
                    rain_month_list[i].append(data[i][j])
        
        for i in range(len(data)):
            occur_month_list[i] = np.asarray(occur_month_list[i], dtype = np.float64)
            rain_month_list[i] = np.asarray(rain_month_list[i], dtype = np.float64)
        
        return occur_month_list, rain_month_list
    
    def get_month_zero_df(self):
        mean = pd.Series([0,0,0,0,0,0,0,0,0,0,0,0])
        std = pd.Series([0,0,0,0,0,0,0,0,0,0,0,0])
        
        df = pd.DataFrame()
        df.insert(loc = 0, column = 'mean', value = mean)
        df.insert(loc = 1, column = 'std', value = std)
    
        return df
  
    def get_RFD_threshold(self):
        file_path = self.file_path_data_sheet
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        date_month = pd.Series(df['date_month'])
        avaliability = pd.Series(df['RFD_his'])
        
        df_r = pd.DataFrame()
        df_r.insert(loc = 0, column = 'date_month', value = date_month)
        df_r.insert(loc = 0, column = 'RFD_threshold', value = avaliability)
        
        df_r = df_r.dropna()
        df_r = df_r.reset_index(drop = True)
        output = []
        for i in range(12):
            output.append(df_r["RFD_threshold"][i])
        
        return np.array(output)
    
    def get_BIER(self):
        file_path = self.file_path_data_sheet
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        date_month = pd.Series(df['date_month'])
        avaliability = pd.Series(df['BIER_L'])
        
        df_r = pd.DataFrame()
        df_r.insert(loc = 0, column = 'date_month', value = date_month)
        df_r.insert(loc = 0, column = 'BIER_L', value = avaliability)        
        
        df_r = df_r.reset_index(drop = True)
        output = []
        for i in range(12):
            output.append(df_r["BIER_L"][i])
        
        return np.array(output)

    def get_average_evapotranspiration_data(self):
        file_path = self.file_path_data_sheet
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        date_month = pd.Series(df["date_month"])
        ev = pd.Series(df["E"])*7.64*10**6 * 10**-3 / 86400
        
        df_r = pd.DataFrame()
        df_r.insert(loc = 0, column = "date_month", value = date_month)
        df_r.insert(loc = 1, column = "ev", value = ev)
        
        df_r = df_r.dropna()
        df_r = df_r.reset_index(drop = True)
    
        month_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(len(df_r)):
            for j in range(1,13):
                if df_r['date_month'][i].month == j:
                    month_list[j-1].append(df_r['ev'][i])
                else:
                    continue
            
        for i in range(len(month_list)):
           month_list[i] = np.mean(month_list[i])
        
        return np.asarray(month_list, dtype = np.float64)        
 
    def get_monthly_inflow_data(self):
        file_path = self.file_path_data_sheet
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        date_month = pd.Series(df["date_month"])
        ev = pd.Series(df["inflow"])
        
        df_r = pd.DataFrame()
        df_r.insert(loc = 0, column = "date_month", value = date_month)
        df_r.insert(loc = 1, column = "inflow", value = ev)
        
        df_r = df_r.dropna()
        df_r = df_r.reset_index(drop = True)
    
        month_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(len(df_r)):
            for j in range(1,13):
                if df_r['date_month'][i].month == j:
                    month_list[j-1].append(df_r['inflow'][i])
                else:
                    continue
            
        for i in range(12):
            
            month_list[i] = np.array(month_list[i])
        
        return month_list

    def get_monthly_discharge_correction(self):
        file_path = self.file_path_data_sheet
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        df_r = pd.DataFrame()
        
        capacity = 3000
        date_series = pd.Series(df["date_month"])
        inflow = np.array(pd.Series(df["inflow"]))
        discharge = np.array(pd.Series(df["discharge_power"]))
        pre_storage = np.array(pd.Series(df["initial_storage"]))
        consum = np.array(pd.Series(df["total_consumption"]))
        df_r.insert(loc = 0, column = "date", value = date_series)
        df_r.insert(loc = 1, column = "inflow", value = inflow)
        df_r.insert(loc = 2, column = "discharge", value = discharge)
        df_r.insert(loc = 3, column = "storage", value = pre_storage)
        df_r.insert(loc = 4, column = "consum", value = consum)
        df_r = df_r.dropna()
        df_r = df_r.reset_index(drop = True)        
        
        outflow = df_r["storage"] + df_r["inflow"] - df_r["consum"] - df_r["discharge"] - capacity
        outflow[outflow < 0] = 0        
        df_r.insert(loc = 5, column = "outflow", value = outflow)
            
        month_list_discharge = [[],[],[],[],[],[],[],[],[],[],[],[]]
        month_list_inflow = [[],[],[],[],[],[],[],[],[],[],[],[]]
        df_rr = df_r[:][0:444]
        
        for i in range(len(df_rr)):
            for j in range(len(month_list_discharge)):
                if df_rr["date"][i].month == j + 1:                  
                    month_list_discharge[j].append(df_rr["discharge"][i])
                    month_list_inflow[j].append(df_rr["inflow"][i])
                else:
                    continue
        
        for i in range(len(month_list_discharge)):
            
            month_list_discharge[i] = np.array(month_list_discharge[i])
            month_list_inflow[i] = np.array(month_list_inflow[i])
        
        return np.mean(month_list_discharge, axis = 1), (month_list_discharge, month_list_inflow) 

    def get_monthly_correction_storage(self):
        file_path = self.file_path_data_sheet
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        df_r = pd.DataFrame()
        
        capacity = 3000
        date_series = pd.Series(df["date_month"])
        storage = np.array(pd.Series(df["initial_storage"]))
        inflow = np.array(pd.Series(df["inflow"]))
        discharge = np.array(pd.Series(df["discharge_power"]))
        consumption = np.array(pd.Series(df["total_consumption"]))
        de_storage = np.array(pd.Series(df["del_storage"]))
        
        df_r.insert(loc = 0, column = "date", value = date_series)
        df_r.insert(loc = 1, column = "storage", value = storage)
        df_r.insert(loc = 2, column = "de_storage", value = de_storage)
        df_r.insert(loc = 3, column = "inflow", value = inflow)
        df_r.insert(loc = 4, column = "consum", value = consumption)
        df_r.insert(loc = 5, column = "discharge", value = discharge)
        df_r = df_r.dropna()
        df_r = df_r.reset_index(drop = True)        
        overflow = df_r["storage"] + df_r["inflow"] - df_r["consum"] - df_r["discharge"] - capacity
        overflow[overflow < 0] = 0        
        df_r.insert(loc = 6, column = "overflow", value = overflow)
        correct = df_r["de_storage"] - (df_r["inflow"] - df_r["consum"] - df_r["discharge"] - df_r["overflow"])
        df_r.insert(loc = 7, column = "correct", value = correct)

        month_list_correction = [[],[],[],[],[],[],[],[],[],[],[],[]]
        month_list_inflow = [[],[],[],[],[],[],[],[],[],[],[],[]]
        df_rr = df_r[:][0:444]
        
        for i in range(len(df_rr)):
            for j in range(len(month_list_correction)):
                if df_rr["date"][i].month == j + 1:                  
                    month_list_correction[j].append(df_rr["correct"][i])
                    month_list_inflow[j].append(df_rr["inflow"][i])
                else:
                    continue
        
        for i in range(len(month_list_correction)):
            month_list_correction[i] = np.array(month_list_correction[i])
            month_list_inflow[i] = np.array(month_list_inflow[i])
        
        return np.mean(month_list_correction, axis = 1), (month_list_correction, month_list_inflow) 
        
    def get_storage_correction_inflow_consum_dataframe(self):
        capacity = 3000
        file_path = self.file_path_data_sheet
        read_data = pd.read_excel(file_path)
        df = pd.DataFrame(read_data)
        
        df_r = pd.DataFrame()
        date_series = pd.Series(df["date_month"])
        storage_state = np.array(pd.Series(df["storage"]))
        inflow = np.array(pd.Series(df["inflow"]))
        consumption = np.array(pd.Series(df["total_consumption"]))
        discharge = np.array(pd.Series(df["discharge_power"]))
        pre_storage = np.array(pd.Series(df["initial_storage"]))
        rfd = np.array(pd.Series(df["RFD"]))
        threshold = np.array(pd.Series(df["RFD_his"]))
        cwdi = np.array(pd.Series(df["WDI"]))
        imsrri_6 = np.array(pd.Series(df["IMSRRI_6"]))
        imsrri_12 = np.array(pd.Series(df["IMSRRI_12"]))
        de_storage = np.array(pd.Series(df["del_storage"]))
        drought_record = np.array(pd.Series(df["drought_record"]))
        
        df_r.insert(loc = 0, column = "date", value = date_series)
        df_r.insert(loc = 1, column = "storage", value = pre_storage)
        df_r.insert(loc = 2, column = "de_storage", value = de_storage)
        df_r.insert(loc = 3, column = "inflow", value = inflow)
        df_r.insert(loc = 4, column = "consum", value = consumption)
        df_r.insert(loc = 5, column = "discharge", value = discharge)    
        overflow = df_r["storage"] + df_r["inflow"] - df_r["consum"] - df_r["discharge"] - capacity
        overflow[overflow < 0] = 0        
        df_r.insert(loc = 6, column = "overflow", value = overflow)
        correct = df_r["de_storage"] - (df_r["inflow"] - df_r["consum"] - df_r["discharge"] - df_r["overflow"])
        df_r.insert(loc = 7, column = "correct", value = correct)        
        
        df_r.insert(loc = 8, column = "rfd", value = rfd)
        df_r.insert(loc = 9, column = "cwdi", value = cwdi)
        df_r.insert(loc = 10, column = "threshold", value = threshold)
        df_r.insert(loc = 11, column = "imsrri_6", value = imsrri_6)
        df_r.insert(loc = 12, column = "imsrri_12", value = imsrri_12)
        df_r.insert(loc = 13, column = "storage_state", value = storage_state)
        df_r.insert(loc = 14, column = "drought_record", value = drought_record)
        df_r = df_r.dropna()
        df_r = df_r.reset_index(drop = True)        
        return df_r  
         
class hydraulic_freq_analyse:
    def __init__(self,data_list , log_list = True):
        self.data_list = data_list      
        array_list = []
        property_list = []
        
        if log_list == True:
            distribution_list =  ['norm', 'gumbel_r', 'gamma',  'pearson3', 'lognorm', "gumbel_l"]
        else :
            distribution_list =  ['norm', 'gumbel_r', 'gamma',  'pearson3', "gumbel_l"]
        
        for i in range(len(data_list)):
            temp = np.asarray(data_list[i])
            array_list.append(temp)
            temp_mean = temp.mean()
            temp_var = temp.var()
            temp_skew = st.skew(temp)
            temp_kurt = st.kurtosis(temp)
            temp_property = np.asarray([temp_mean, temp_var, temp_skew, temp_kurt])
            property_list.append(temp_property)
        
        self.array_list = array_list
        self.property_list = property_list
        self.distribution_list = distribution_list

    def get_norm_param(self, sample):
        output = st.norm.fit(sample)
        return output 
    
    def get_gumbel_r_param(self, sample):
        output = st.gumbel_r.fit(sample)
        return output
    
    def get_gumbel_l_param(self, sample):
        output = st.gumbel_l.fit(sample)
        return output
    
    def get_pearson3_param(self, sample):
        output = st.pearson3.fit(sample)
        return output
    
    def get_gamma_param(self, sample):
        output = st.gamma.fit(sample)
        return output
    
    def get_lognorm_param(self, sample):
        output = st.lognorm.fit(sample)
        return output
    
    def get_loggamma_param(self, sample):
        output = st.loggamma.fit(sample)
        return output
    
    def ktest(self, sample, pick_distribution):
        if pick_distribution == 'norm':
            parameter = self.get_norm_param(sample)
            return st.kstest(sample, 'norm', parameter)[1], parameter
        
        elif pick_distribution == 'gamma':
            parameter = self.get_gamma_param(sample)
            return st.kstest(sample, 'gamma',parameter)[1], parameter
        
        elif pick_distribution == 'gumbel_r':
            parameter = self.get_gumbel_r_param(sample)
            return st.kstest(sample, 'gumbel_r', parameter)[1], parameter
        
        elif pick_distribution == 'gumbel_l':
            parameter = self.get_gumbel_l_param(sample)
            return st.kstest(sample, 'gumbel_l', parameter)[1], parameter
        
        elif pick_distribution == 'lognorm':
            parameter = self.get_lognorm_param(sample)
            return st.kstest(sample, 'lognorm', parameter)[1], parameter
        
        elif pick_distribution == 'pearson3':
            parameter = self.get_pearson3_param(sample)
            return st.kstest(sample, 'pearson3', parameter)[1], parameter
        
        elif pick_distribution == 'loggamma':
            parameter = self.get_loggamma_param(sample)
            return st.kstest(sample, 'loggamma', parameter)[1], parameter   
        
    def get_suitable_distribution_list(self):
        array_list = self.array_list
        distribution_list = self.distribution_list
        result = []
        for i in range(len(array_list)):
            compare_list = []
            sample = array_list[i]
            temp_parameter_list = []
            
            for j in range(len(distribution_list)):
                temp = self.ktest(sample, distribution_list[j])[0]
                temp_p = self.ktest(sample, distribution_list[j])[1]
                compare_list.append(temp)
                temp_parameter_list.append(temp_p)
            
            compare_list = np.asarray(compare_list)

            maxx = np.max(compare_list)
            k = np.argmax(compare_list)

            temp_result = distribution_list[k]
            temp_parameter = temp_parameter_list[k]
            result.append([temp_result,temp_parameter,maxx])
                
        return result

class temperature_ar_model:
    def __init__(self, array_list,log_list = False):
        self.array_list = array_list
        array_af = hydraulic_freq_analyse(self.array_list, log_list = log_list)
        self.array_af = array_af
        suitable_distribution_list = self.array_af.get_suitable_distribution_list()
        dis_statproperty_list = self.array_af.property_list 
        
        self.suitable_distribution_list = suitable_distribution_list
        self.dis_param_list = dis_statproperty_list
        self.his_cdf_list = self.CDF_input_data_list()
        self.his_cdf_list_norm = self.CDF_input_data_list_norm()
        
        self.n_lag_list = self.n_lag_list()
        self.cdf_acf_list = self.acf_list()
        
        self.cdf_property_list = self.cdf_property_list()
        self.cdf_property_list_norm = self.cdf_property_list_norm()
        
        self.model_params = self.cdf_ar_model()
        self.model_params_v2 = self.cdf_ar_model_norm()
        
    def get_multinormal_cov(self, acf, std, n_lag):
        array_list = []
        for i in range(n_lag):
            temp_list = []
            for j in range(n_lag):
                for k in range(n_lag):
                    if abs(j-i) == k:
                        temp = std*std*acf[k]
                        temp_list = temp
                        
            array_list.append(temp_list)
        cov = np.asarray(array_list, dtype = np.float64)
        return cov

    def get_multinormal_mean(self, mean, n_lag):
        array_list = []
        for i in range(n_lag):
            array_list.append(mean)
        mean_vector = np.array(array_list,dtype = np.float64)
        return mean_vector  
        
    def get_initial_sample(self, mean, std, n_lag, acf):
        cov = self.get_multinormal_cov(acf, std, n_lag)
        mean = self.get_multinormal_mean(mean, n_lag)
        initial = st.multivariate_normal.rvs(mean = mean, cov = cov)
        return initial
    
    def get_white_noise_property(self, ar_model_params, acf, std, n_lag):
        constant = 1
        for i in range(1,n_lag + 1):
            temp = acf[i] * ar_model_params[i]
            constant = constant - temp
        
        nois_std = np.sqrt(constant * std*std) 
        return np.asarray( [0, nois_std] ,dtype = np.float64)

    def cdf_property_list(self):
        cdf_list = self.his_cdf_list
        output_list = []
        for i in range(len(cdf_list)):
            temp = [np.mean(cdf_list[i]), np.std(cdf_list[i]), st.skew(cdf_list[i])]
            output_list.append(temp)
        return output_list

    def cdf_property_list_norm(self):
        cdf_list = self.his_cdf_list_norm
        output_list = []
        for i in range(len(cdf_list)):
            temp = [np.mean(cdf_list[i]), np.std(cdf_list[i]), st.skew(cdf_list[i])]
            output_list.append(temp)
        return output_list
 
    def parameter_distribution_transform(self, name, dis_param, data):
        if name == 'norm':
            return st.norm.cdf(data,
                               loc = dis_param[0],
                               scale = dis_param[1])
        elif name == 'gamma':
            return st.gamma.cdf(data, 
                                a = dis_param[0],
                                loc = dis_param[1], 
                                scale = dis_param[2])
        elif name == 'pearson3':
            return st.pearson3.cdf(data,  
                                   skew = dis_param[0],
                                   loc = dis_param[1],
                                   scale = dis_param[2])
        elif name == 'gumbel_r':
            return st.gumbel_r.cdf(data,
                                   loc = dis_param[0],
                                   scale = dis_param[1])
        elif name == 'gumbel_l':
            return st.gumbel_l.cdf(data,
                                   loc = dis_param[0],
                                   scale = dis_param[1])
        elif name == 'lognorm' :
            return st.lognorm.cdf(data,
                                  s = dis_param[0],
                                  loc = dis_param[1],
                                  scale = dis_param[2])
        elif name == 'loggamma' :
            return st.loggamma.cdf(data,
                                   c = dis_param[0],
                                   loc = dis_param[1], 
                                   scale = dis_param[2])
        
    def ppf_transform(self, name, dis_param, data):
        if name == 'norm':
            return st.norm.ppf(data,
                               loc = dis_param[0],
                               scale = dis_param[1])
        elif name == 'gamma':
            return st.gamma.ppf(data, 
                                a = dis_param[0],
                                loc = dis_param[1], 
                                scale = dis_param[2])
        elif name == 'pearson3':
            return st.pearson3.ppf(data,  
                                   skew = dis_param[0],
                                   loc = dis_param[1],
                                   scale = dis_param[2])
        elif name == 'gumbel_r':
            return st.gumbel_r.ppf(data,
                                   loc = dis_param[0],
                                   scale = dis_param[1])
        elif name == 'gumbel_l':
            return st.gumbel_l.ppf(data,
                                   loc = dis_param[0],
                                   scale = dis_param[1])
        elif name == 'lognorm':
            return st.lognorm.ppf(data,
                                  s = dis_param[0],
                                  loc = dis_param[1],
                                  scale = dis_param[2])
        elif name == 'loggamma':
            return st.loggamma.ppf(data,
                                   c = dis_param[0],
                                   loc = dis_param[1], 
                                   scale = dis_param[2])
    
    def CDF_input_data_list(self):
        array_list = self.array_list
        body = self.suitable_distribution_list
        output_list = []
        for i in range(len(array_list)):
            name = body[i][0]
            dis_param = body[i][1]
            data = array_list[i]
            temp = self.parameter_distribution_transform(name, dis_param, data)
            output_list.append(temp)
        return output_list

    def CDF_input_data_list_norm(self):
        array_list = self.array_list
        output_list = []
        for i in range(len(array_list)):
            name = "norm"
            dis_param = [self.dis_param_list[i][0],self.dis_param_list[i][0]]
            data = array_list[i]
            temp = self.parameter_distribution_transform(name, dis_param, data)
            output_list.append(temp)
        
        return output_list
    
    def n_lag_list(self):
        data_list = self.his_cdf_list
        n_lag_list = []
        for i in range(len(data_list)):
            temp_lag_array = tsa.pacf_yw(data_list[i])
            for j in range(len(temp_lag_array)):
                if abs(temp_lag_array[j]) <= 1.96 / np.sqrt(len(temp_lag_array)) :
                    n_lag = j + 1
                    break
                else:
                    continue
            n_lag_list.append(n_lag)
        return n_lag_list
    
    def acf_list(self):
        data_list = self.his_cdf_list
        output_list = []
        for i in range(len(data_list)):
             temp_acf = tsa.acf(data_list[i])
             output_list.append(temp_acf)
        return output_list       
        
    def cdf_ar_model(self):
        model_list = []
        cdf_array_list = self.his_cdf_list
        for i in range(len(cdf_array_list)):
            data_series = cdf_array_list[i]
            n_lag = self.n_lag_list[i]
            model = ARIMA(data_series, order=(n_lag-1,0,0))
            fitting = model.fit()
            fitting_params_p = fitting.params
            model_list.append(fitting_params_p)
        return model_list  

    def cdf_ar_model_norm(self):
        model_list = []
        cdf_array_list = self.his_cdf_list_norm
        for i in range(len(cdf_array_list)):
            data_series = cdf_array_list[i]
            n_lag = self.n_lag_list[i]
            model = ARIMA(data_series, order=(n_lag-1,0,0))
            fitting = model.fit()
            fitting_params_p = fitting.params
            model_list.append(fitting_params_p)
        return model_list

    def get_monthly_cdf_series_sampling(self, month):
        mean, std, skew = self.cdf_property_list[month - 1]
        acf = self.cdf_acf_list[month - 1]
        n_lag = self.n_lag_list[month - 1]
        params = self.model_params[month - 1]
        
        def month_day(month):
            month_list = [31,28,31,30,31,30,31,31,30,31,30,31]
            return month_list[month - 1]
        
        initial = self.get_initial_sample(mean, std, n_lag, acf)
        mean_array = params[0] * np.ones(n_lag)
        
        initial = initial - mean_array
        noise_params = self.get_white_noise_property(params, acf, std, n_lag)
        initial_pa = np.hstack((np.array(1), initial))
        
        simulation_time = month_day(month) - n_lag
        output_array = initial
        
        for i in range(simulation_time):
            if i == 0 :
                next_step = np.dot(initial_pa, params) + st.norm.rvs(loc = noise_params[0],
                                                                    scale = noise_params[1], 
                                                                    size = 1)
                output_array = np.concatenate((np.array(next_step),output_array), axis = None)
            else :
                temp_array = output_array[i: i + n_lag]
                temp_array = temp_array 
                temp_array = np.hstack((np.array(1), temp_array))
                next_step = np.dot(temp_array,params) + st.norm.rvs(loc = noise_params[0],
                                                                    scale = noise_params[1],
                                                                    size = 1)
                output_array = np.concatenate((np.array(next_step),output_array),axis = None)
        
        output_array_v = st.norm.cdf(output_array, loc = mean, scale = std)

        return output_array_v

    def get_monthly_cdf_series_sampling_v2(self, month):
        mean, std, skew = self.cdf_property_list_norm[month - 1]
        acf = self.cdf_acf_list[month - 1]
        n_lag = self.n_lag_list[month - 1]
        params = self.model_params_v2[month - 1]
        
        def month_day(month):
            month_list = [31,28,31,30,31,30,31,31,30,31,30,31]
            return month_list[month - 1]
        
        initial = self.get_initial_sample(mean, std, n_lag, acf)
        mean_array = params[0] * np.ones(n_lag)
        
        initial = initial - mean_array
        noise_params = self.get_white_noise_property(params, acf, std, n_lag)
        initial_pa = np.hstack((np.array(1), initial))
        
        simulation_time = month_day(month) - n_lag
        output_array = initial
        
        for i in range(simulation_time):
            if i == 0 :
                next_step = np.dot(initial_pa, params) + st.norm.rvs(loc = noise_params[0],
                                                                    scale = noise_params[1], 
                                                                    size = 1)
                output_array = np.concatenate((np.array(next_step),output_array), axis = None)
                
            else :
                temp_array = output_array[i: i + n_lag]
                temp_array = temp_array 
                temp_array = np.hstack((np.array(1), temp_array))
                next_step = np.dot(temp_array,params) + st.norm.rvs(loc = noise_params[0],
                                                                    scale = noise_params[1],
                                                                    size = 1)
                output_array = np.concatenate((np.array(next_step),output_array),axis = None)
        
        output_array_v = st.norm.cdf(output_array, loc = mean, scale = std)
        
        return output_array_v

    def get_temperature_sampling(self, month):
        cdf_series = self.get_monthly_cdf_series_sampling(month)
        body = self.suitable_distribution_list[month - 1]
        name = body[0]
        dis_param = body[1]
        
        return self.ppf_transform(name, dis_param, cdf_series)

    def get_temperature_sampling_v2(self, month):
        cdf_series = self.get_monthly_cdf_series_sampling_v2(month)
        body = self.suitable_distribution_list[month - 1]
        name = body[0]
        dis_param = body[1]
        return self.ppf_transform(name, dis_param, cdf_series)
        
    def check_temperature_sampling_mean(self, month, times = 1000):
        test = 0
        for i in range(times):
            temp = np.mean(self.get_temperature_sampling(month))
            test = test + temp
        return test / times
    
    def check_temperature_sampling_acf(self, month, times = 1000):
        array = []
        for i in range(times):
            temp = self.get_temperature_sampling(month)
            array.append(temp)
        array = np.array(array)
        array = array.flatten()
        return tsa.acf(array)

    def check_sampling_stat(self, times = 1000):
        mean = []
        acf = []
        for i in range(12):
            mean.append(self.check_temperature_sampling_mean(i + 1))
            acf.append(self.check_temperature_sampling_acf(i + 1 )[0 : 4])
        return mean, acf 
    
    def generate_monthly_temperature_simulation(self, time = 1000):
        output_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(len(output_list)):
            month_array = []
            for j in range(time):
                temp = self.get_temperature_sampling(i + 1)
                month_array.append(temp)
            
            month_array = np.array(month_array)
            month_array = month_array.flatten()
            
            output_list[i] = month_array
        
        return output_list

    def generate_monthly_temperature_simulation_v2(self, time = 1000):
        output_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(len(output_list)):
            month_array = []
            for j in range(time):
                temp = self.get_temperature_sampling_v2(i + 1)
                month_array.append(temp)
            month_array = np.array(month_array)
            month_array = month_array.flatten()
            
            output_list[i] = month_array
        
        return output_list

class precipitation_ar_model:
    def __init__(self, rainy_day_list, rainfall_amount_list, log_list = False):
        self.rainy_day_list = rainy_day_list
        self.rainfall_amount_list = rainfall_amount_list
        
        rainy_day_hf = hydraulic_freq_analyse(rainy_day_list, log_list = log_list)
        rainfall_amount_hf = hydraulic_freq_analyse(rainfall_amount_list, log_list = log_list)
        
        self.rainy_day_af = rainy_day_hf
        self.rainfall_amount_hf = rainfall_amount_hf
        
        rainfall_distribution_list = self.rainfall_amount_hf.get_suitable_distribution_list()
        rainfall_statproperty_list = self.rainfall_amount_hf.property_list
        rainy_distribution_list = self.rainy_day_af.get_suitable_distribution_list()
        self.rainy_statproperty_list = self.rainy_day_af.property_list
        
        self.rainfall_distribution_list = rainfall_distribution_list
        self.rainy_distribution_list = rainy_distribution_list
        
        self.rainfall_statproperty_list = rainfall_statproperty_list
        self.his_cdf_list = self.CDF_input_data_list()
        self.his_cdf_list_norm = self.CDF_input_data_list_norm()
        
        self.n_lag_list = self.n_lag_list()
        self.n_lag_list_nt = self.n_lag_list_nt()
        self.cdf_acf_list = self.acf_list()
        self.acf_list_nt = self.acf_list_nt()
        
        self.cdf_property_list = self.cdf_property_list()
        self.cdf_property_list_norm = self.cdf_property_list_norm()
        
        self.model_params = self.cdf_ar_model()
        self.model_params_norm = self.cdf_ar_model_norm()
        self.model_params_nt = self.cdf_ar_model_nt()
        
    def get_multinormal_cov(self, acf, std, n_lag):
        array_list = []
        for i in range(n_lag):
            temp_list = []
            for j in range(n_lag):
                for k in range(n_lag):
                    if abs(j-i) == k:
                        temp = std*std*acf[k]
                        temp_list = temp
                        
            array_list.append(temp_list)
        cov = np.asarray(array_list, dtype = np.float64)
        
        return cov

    def get_multinormal_mean(self, mean, n_lag):
        array_list = []
        for i in range(n_lag):
            array_list.append(mean)
        mean_vector = np.array(array_list,dtype = np.float64)
        return mean_vector
         
    def get_initial_sample(self, mean, std, n_lag, acf): 
        cov = self.get_multinormal_cov(acf, std, n_lag)
        mean = self.get_multinormal_mean(mean, n_lag)
        initial = st.multivariate_normal.rvs(mean = mean, cov = cov)
        return initial
    
    def get_white_noise_property(self, ar_model_params, acf, std, n_lag):
        constant = 1
        for i in range(1,n_lag + 1):
            temp = acf[i] * ar_model_params[i]
            constant = constant - temp
        nois_std = np.sqrt(constant * std*std) 
        
        return np.asarray( [0, nois_std] ,dtype = np.float64)
    
    def cdf_property_list(self):
        cdf_list = self.his_cdf_list
        output_list = []
        for i in range(len(cdf_list)):
            temp = [np.mean(cdf_list[i]), np.std(cdf_list[i]), st.skew(cdf_list[i])]
            output_list.append(temp)
        return output_list

    def cdf_property_list_norm(self):
        cdf_list = self.his_cdf_list_norm
        output_list = []
        for i in range(len(cdf_list)):
            temp = [np.mean(cdf_list[i]), np.std(cdf_list[i]), st.skew(cdf_list[i])]
            output_list.append(temp)
        return output_list
        
    def parameter_distribution_transform(self, name, dis_param, data):
        if name == 'norm':
            return st.norm.cdf(data,
                               loc = dis_param[0],
                               scale = dis_param[1])
        elif name == 'gamma':
            return st.gamma.cdf(data, 
                                a = dis_param[0],
                                loc = dis_param[1], 
                                scale = dis_param[2])
        elif name == 'pearson3':
            return st.pearson3.cdf(data,  
                                   skew = dis_param[0],
                                   loc = dis_param[1],
                                   scale = dis_param[2])
        elif name == 'gumbel_r':
            return st.gumbel_r.cdf(data,
                                   loc = dis_param[0],
                                   scale = dis_param[1])
        elif name == 'gumbel_l':
            return st.gumbel_l.cdf(data,
                                   loc = dis_param[0],
                                   scale = dis_param[1])
        elif name == 'lognorm':
            return st.lognorm.cdf(data,
                                  s = dis_param[0],
                                  loc = dis_param[1],
                                  scale = dis_param[2])
        elif name == 'loggamma':
            return st.loggamma.cdf(data,
                                   c = dis_param[0],
                                   loc = dis_param[1], 
                                   scale = dis_param[2])
        
    def ppf_transform(self, name, dis_param, data):
        if name == 'norm':
            return st.norm.ppf(data,
                               loc = dis_param[0],
                               scale = dis_param[1])
        elif name == 'gamma':
            return st.gamma.ppf(data, 
                                a = dis_param[0],
                                loc = dis_param[1], 
                                scale = dis_param[2])
        elif name == 'pearson3':
            return st.pearson3.ppf(data,  
                                   skew = dis_param[0],
                                   loc = dis_param[1],
                                   scale = dis_param[2])
        elif name == 'gumbel_r':
            return st.gumbel_r.ppf(data,
                                   loc = dis_param[0],
                                   scale = dis_param[1])
        elif name == 'gumbel_l':
            return st.gumbel_l.ppf(data,
                                   loc = dis_param[0],
                                   scale = dis_param[1])
        elif name == 'lognorm':
            return st.lognorm.ppf(data,
                                  s = dis_param[0],
                                  loc = dis_param[1],
                                  scale = dis_param[2])
        elif name == 'loggamma':
            return st.loggamma.ppf(data,
                                   c = dis_param[0],
                                   loc = dis_param[1], 
                                   scale = dis_param[2])

    def CDF_input_data_list(self):
        array_list = self.rainy_day_list
        body = self.rainy_distribution_list
        output_list = []
        for i in range(len(array_list)):
            name = body[i][0]
            dis_param = body[i][1]
            data = array_list[i]
        
            temp = self.parameter_distribution_transform(name, dis_param, data)
            output_list.append(temp)
        
        return output_list

    def CDF_input_data_list_norm(self):
        array_list = self.rainy_day_list
        body = self.rainy_statproperty_list
        output_list = []
        
        for i in range(len(array_list)):
            name = "norm"
            dis_param = [body[i][0], body[i][1]]
            data = array_list[i]
        
            temp = self.parameter_distribution_transform(name, dis_param, data)
            output_list.append(temp)
        
        return output_list
    
    def n_lag_list(self):
        data_list = self.his_cdf_list
        n_lag_list = []
        for i in range(len(data_list)):
            temp_lag_array = tsa.pacf_yw(data_list[i])
            for j in range(len(temp_lag_array)):
                if abs(temp_lag_array[j]) <= 1.96 / np.sqrt(len(temp_lag_array)) :
                    n_lag = j + 1
                    break
                else:
                    continue
            n_lag_list.append(n_lag)
        
        return n_lag_list
    
    def n_lag_list_nt(self):
        data_list = self.rainy_day_list
        n_lag_list = []
        for i in range(len(data_list)):
            temp_lag_array = tsa.pacf_yw(data_list[i])
            for j in range(len(temp_lag_array)):
                if abs(temp_lag_array[j]) <= 1.96 / np.sqrt(len(temp_lag_array)) :
                    n_lag = j + 1
                    break
                
                else:
                    continue
            n_lag_list.append(n_lag)
        return n_lag_list
    
    def acf_list(self):
        data_list = self.his_cdf_list
        output_list = []
        for i in range(len(data_list)):
             temp_acf = tsa.acf(data_list[i])
             output_list.append(temp_acf)
             
        return output_list

    def acf_list_nt(self):
        data_list = self.rainy_day_list
        output_list = []
        for i in range(len(data_list)):
             temp_acf = tsa.acf(data_list[i])
             output_list.append(temp_acf)
             
        return output_list            
              
    def cdf_ar_model(self):
        model_list = []
        cdf_array_list = self.his_cdf_list
        for i in range(len(cdf_array_list)):
            data_series = cdf_array_list[i]
            n_lag = self.n_lag_list[i]
            model = ARIMA(data_series, order=(n_lag-1,0,0))
            fitting = model.fit()
            fitting_params_p = fitting.params
            model_list.append(fitting_params_p)
        
        return model_list
      
    def cdf_ar_model_norm(self):
        model_list = []
        cdf_array_list = self.his_cdf_list_norm
        for i in range(len(cdf_array_list)):
            data_series = cdf_array_list[i]
            n_lag = self.n_lag_list[i]
            model = ARIMA(data_series, order=(n_lag-1,0,0))
            fitting = model.fit()
            fitting_params_p = fitting.params
            model_list.append(fitting_params_p)
        
        return model_list  
    
    def cdf_ar_model_nt(self):
        model_list = []
        array_list = self.rainy_day_list
        for i in range(len(array_list)):
            data_series = array_list[i]
            n_lag = self.n_lag_list[i]
            model = ARIMA(data_series, order=(n_lag-1,0,0))
            fitting = model.fit()
            fitting_params_p = fitting.params
            model_list.append(fitting_params_p)
        
        return model_list

    def get_monthly_cdf_series_sampling_nt(self, month):
        mean, std, skewness, kurt = self.rainy_statproperty_list[month - 1]
        acf = self.acf_list_nt[month - 1]
        n_lag = self.n_lag_list_nt[month - 1]
        params = self.model_params_nt[month - 1]
        
        def month_day(month):
            month_list = [31,28,31,30,31,30,31,31,30,31,30,31]
            return month_list[month - 1]
        
        initial = self.get_initial_sample(mean, std, n_lag, acf)
        mean_array = params[0] * np.ones(n_lag)
        
        initial = initial - mean_array
        noise_params = self.get_white_noise_property(params, acf, std, n_lag)
        initial_pa = np.hstack((np.array(1), initial))
        
        simulation_time = month_day(month) - n_lag
        output_array = initial
    
        for i in range(simulation_time):
            if i == 0 :
                next_step = np.dot(initial_pa, params) + st.norm.rvs(loc = noise_params[0],
                                                                    scale = noise_params[1], 
                                                                    size = 1)
                output_array = np.concatenate((np.array(next_step),output_array), axis = None)
            else :
                temp_array = output_array[i: i + n_lag]
                temp_array = temp_array 
                temp_array = np.hstack((np.array(1), temp_array))
                next_step = np.dot(temp_array,params) + st.norm.rvs(loc = noise_params[0],
                                                                    scale = noise_params[1],
                                                                    size = 1)
                output_array = np.concatenate((np.array(next_step),output_array),axis = None)
        
        output_array_v = st.norm.cdf(output_array, loc = mean, scale = std)
        
        return output_array_v

    def get_monthly_cdf_series_sampling(self, month):
        mean, std, skew = self.cdf_property_list[month - 1]
        acf = self.cdf_acf_list[month - 1]
        n_lag = self.n_lag_list[month - 1]
        params = self.model_params[month - 1]

        def month_day(month):
            month_list = [31,28,31,30,31,30,31,31,30,31,30,31]
            return month_list[month - 1]
        
        initial = self.get_initial_sample(mean, std, n_lag, acf)
        mean_array = params[0] * np.ones(n_lag)
        
        initial = initial - mean_array
        noise_params = self.get_white_noise_property(params, acf, std, n_lag)
        initial_pa = np.hstack((np.array(1), initial))
        
        simulation_time = month_day(month) - n_lag
        output_array = initial
        
        for i in range(simulation_time):
            if i == 0 :
                next_step = np.dot(initial_pa, params) + st.norm.rvs(loc = noise_params[0],
                                                                    scale = noise_params[1], 
                                                                    size = 1)
                output_array = np.concatenate((np.array(next_step),output_array), axis = None)
            else :
                temp_array = output_array[i: i + n_lag]
                temp_array = temp_array 
                temp_array = np.hstack((np.array(1), temp_array))
                next_step = np.dot(temp_array,params) + st.norm.rvs(loc = noise_params[0],
                                                                    scale = noise_params[1],
                                                                    size = 1)
                output_array = np.concatenate((np.array(next_step),output_array),axis = None)
        
        
        output_array_v = st.norm.cdf(output_array, loc = mean, scale = std)
        
        return output_array_v
    
    def get_monthly_cdf_series_sampling_norm(self, month):
        mean, std, skew = self.cdf_property_list_norm[month - 1]
        acf = self.cdf_acf_list[month - 1]
        n_lag = self.n_lag_list[month - 1]
        params = self.model_params_norm[month - 1]
        
        def month_day(month):
            month_list = [31,28,31,30,31,30,31,31,30,31,30,31]
            return month_list[month - 1]
        
        initial = self.get_initial_sample(mean, std, n_lag, acf)
        mean_array = params[0] * np.ones(n_lag)
        
        initial = initial - mean_array
        noise_params = self.get_white_noise_property(params, acf, std, n_lag)
        initial_pa = np.hstack((np.array(1), initial))
        
        simulation_time = month_day(month) - n_lag
        output_array = initial
        
        
        for i in range(simulation_time):
            if i == 0 :
                next_step = np.dot(initial_pa, params) + st.norm.rvs(loc = noise_params[0],
                                                                    scale = noise_params[1], 
                                                                    size = 1)
                output_array = np.concatenate((np.array(next_step),output_array), axis = None)
                
            else :
                temp_array = output_array[i: i + n_lag]
                temp_array = temp_array 
                temp_array = np.hstack((np.array(1), temp_array))
                next_step = np.dot(temp_array,params)  + st.norm.rvs(loc = noise_params[0],
                                                                    scale = noise_params[1],
                                                                    size = 1)
                output_array = np.concatenate((np.array(next_step),output_array),axis = None)
        
        output_array_v = st.norm.cdf(output_array, loc = mean, scale = std)
        
        return output_array_v

    def get_precipitation_sampling_v1(self, month):
        cdf_series = self.get_monthly_cdf_series_sampling(month)
        body = self.rainfall_distribution_list[month - 1]
        truncated_value = 1 - self.rainy_statproperty_list[month - 1][0]
        
        r_cdf_series = (cdf_series - truncated_value) / (1 - truncated_value)
        r_cdf_series[r_cdf_series < 0] = 0
        
        name = body[0]
        dis_param = body[1]
        # minmum = np.min(self.ppf_transform(name, dis_param, r_cdf_series))
        
        output = self.ppf_transform(name, dis_param, r_cdf_series)
        output[output <= 0.06] = 0
        
        return output    

    def get_precipitation_sampling_v2(self, month):
        cdf_series = self.get_monthly_cdf_series_sampling_norm(month)
        body = self.rainfall_distribution_list[month - 1]
        truncated_value = 1 - self.rainy_statproperty_list[month - 1][0]
        
        r_cdf_series = (cdf_series - truncated_value) / (1 - truncated_value)
        r_cdf_series[r_cdf_series < 0] = 0
        
        name = body[0]
        dis_param = body[1]
        # minmum = np.min(self.ppf_transform(name, dis_param, r_cdf_series))
        
        output = self.ppf_transform(name, dis_param, r_cdf_series)
        output[output <= 0.06] = 0
        
        return output  

    def get_precipitation_sampling_v3(self, month):
        cdf_series = self.get_monthly_cdf_series_sampling_nt(month)
        body = self.rainfall_distribution_list[month - 1]
        truncated_value = 1 - self.rainy_statproperty_list[month - 1][0]
        
        r_cdf_series = (cdf_series - truncated_value) / (1 - truncated_value)
        r_cdf_series[r_cdf_series < 0] = 0
        
        name = body[0]
        dis_param = body[1]
        # minmum = np.min(self.ppf_transform(name, dis_param, r_cdf_series))
        
        output = self.ppf_transform(name, dis_param, r_cdf_series)
        output[output <= 0.06] = 0
        
        return output

    def check_precipitation_sampling_mean(self, month, times = 1000):
        test = 0
        for i in range(times):
            temp = np.mean(self.get_precipitation_sampling_v1(month))
            test = test + temp
        return test / times
    
    def check_precipitation_sampling_acf(self, month, times = 1000):
        array = []
        for i in range(1000):
            temp = self.get_precipitation_sampling_v1(month)
            array.append(temp)
        array = np.array(array)
        array = array.flatten()
        return tsa.pacf(array)
    
    def check_sampling_stat(self, times = 1000):
        mean = []
        acf = []
        for i in range(12):
            mean.append(self.check_precipitation_sampling_mean(i + 1))
            acf.append(self.check_precipitation_sampling_acf(i + 1 )[0 : 4])
        return mean, acf            
    
    def generate_monthly_precipitation_simulation(self, time = 1000):
        output_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(len(output_list)):
            month_array = []
            for j in range(time):
                temp = self.get_precipitation_sampling_v1(i + 1)
                month_array.append(temp)
            
            month_array = np.array(month_array)
            month_array = month_array.flatten()
            
            output_list[i] = month_array
        
        return output_list

    def generate_monthly_precipitation_simulation_v2(self, time = 1000):
        output_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(len(output_list)):
            month_array = []
            for j in range(time):
                temp = self.get_precipitation_sampling_v2(i + 1)
                month_array.append(temp)
            
            month_array = np.array(month_array)
            month_array = month_array.flatten()
            
            output_list[i] = month_array
        
        return output_list        
        
class inflow_estimate :
    def __init__(self, date_series, precipitation_series, temperature_series):
        self.date_series = date_series
        self.precipitation_series = precipitation_series
        self.temperature_series = temperature_series
        
        self.sat_u = 3
        self.sat_s = 7
        self.H_array = [10.65,11.15,11.85,12.6,13.15,13.50,13.35,12.85,12.1,11.4,10.8,10.5]
        self.Kc_array = [0.976,0.975,0.984,0.998,1.007,1.002,0.98,0.982,0.988,1.002,1.009,0.998]
        
        # 0.95, 0.85, 0.99,0.95, 0.8, 0.95,0.75 , 0.98, 0.68
        
        self.cs_1 = 1
        self.cs_2 = 1
        self.u_initial = 0.05 * self.sat_u
        self.s_initial = 0.002 * self.sat_s
        self.reverlution = 100
        
        self.cn_list = [50.5,50.5,50.5,45.55,40.6,30.71,30.71,30.71,55.45,30.7,65.35,45.55]
        self.lamda_list = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0,0.1,0.05,0.1,0.05]
        self.initial_uc_list = [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]
        self.initial_sc_list = [0.00025,0,0.0025,0.00025,0.001,0.00075,0.00475,0.00475,0,0,0,0]

    def partial_physical_model_v2(self, date, 
                               precipitation, 
                               temperature,
                               u,
                               s,
                               cn,
                               lamda
                               ):
        H_array = self.H_array
        Kc_array = self.Kc_array
        sat_u = self.sat_u
        
        if precipitation <= 8 :   ## 依照氣象局指定暴雨狀況以及歷史資料中推估r值以及條件選擇
            r = 0.15
        else:
            r = 0.356
    
        H = H_array[date.month -1 ]   ### 寫入日期中的月份str  並對應讀取日照強度序列的值
        kc = Kc_array[date.month - 1] ### 寫入日其中的月份str 並對應讀取植被覆蓋序列的值

        if u > 0.5*sat_u:             ### 判斷ks值
            ks = 1
        else:
            ks = u/(0.5*sat_u)
    
        ss = (2540/cn) -25.4           ##判斷S
        qr = math.pow((precipitation - lamda*ss),2)/(precipitation + (1-lamda)*ss )   ##水保局地表逕流公式
    
    
        i = max(precipitation - qr, 0)
        e0 = 6.11*math.exp((17.27*temperature)/(temperature + 237.3))
        pet = (0.021*math.pow(H,2)*e0)/(temperature + 273.3)
        ev = min(ks*kc*pet,u+i)
        ## ev cm / day
        
        pc = max(0,u + i-ev-sat_u)
    
    
        g = r*s                                      ##基流
        s_next = s + pc - g                    ## 明天的飽和層含水量
        u_next = u + i - pc - ev               ## 明天的非飽和層含水量
    
        if u_next > sat_u:                   ##含水量不可能超過其最大值
            u_next = sat_u
        elif u_next < 0 :                    ## 含水量不可能為負
            u_next = 0
    
        if s_next < 0 :                      ## 含水量不可能為負
            s_next = 0

        q_f = g + qr                         ##集水區出流量
    
        if q_f < 0:                         ##集水區出流量不可能為負
            q_f = 0
    
        return u_next,s_next,q_f,ev          ## 回傳值為一個tuple    

    def Mega_physical_model_v3(self):
        date_series = self.date_series
        temperature_series = self.temperature_series
        precipitation_series = self.precipitation_series
        
        u_initial_list = self.initial_uc_list
        s_initial_list = self.initial_sc_list
        cn_list = self.cn_list
        lamda_list = self.lamda_list
        
        len_number = len(temperature_series)
        q = []
        u_series = []
        s_series = []
        ev_s = []
        
        def get_cs(month):
            if month <= 11 and month >= 5:
                return self.cs_1
            else:
                return self.cs_2
        
        for i in range(len_number):
            date = date_series[i]
            temperature = temperature_series[i]
            month = date.month
            cn2 = cn_list[int(month) - 1]
            cs = get_cs(month)
            lamda = lamda_list[month - 1]

            precipitation = precipitation_series[i]
            if i == 0:
                u = u_initial_list[month - 1]
                s = s_initial_list[month - 1]
                
                amc_cls = self.AMS_classification_v2(date, precipitation_series)
                cn = self.CN_caculation(amc_cls, cn2)
                
                y = self.partial_physical_model_v2(date, precipitation,
                                                   temperature, u, s, cn, lamda)
                qs = round(cs*y[2]*763.4*10000/86400,7)
                q.append(qs)
                u_series.append(y[0])
                s_series.append(y[1])
                ev_t = round(y[3] * 763.4 * 10000 / 86400, 5)
                ev_s.append(ev_t)
                
            else:
                u = u_series[i-1] 
                s = s_series[i-1]
                
                amc_cls = self.AMS_classification_v2(date = date,
                                                     precipitation_series = precipitation_series)
                cn = self.CN_caculation(amc_cls,cn2)
                
                y = self.partial_physical_model_v2(date,precipitation,
                                                   temperature,u,s,cn,lamda)
                qs = round(cs*y[2]*763.4*10000/86400,7)
                q.append(qs)
                u_series.append(y[0])
                s_series.append(y[1])
                ev_t = round(y[3] * 763.4 * 10000 / 86400, 5)
                ev_s.append(ev_t)
                
                
        q = np.asarray(q, dtype = np.float64)
        u_series = np.asarray(u_series, dtype = np.float64)
        s_series = np.asarray(s_series, dtype = np.float64)
        ev_s = np.asarray(ev_s, dtype = np.float64)
            
            
        
        return q, ev_s      

    def generate_month_date_series_optimization(self, month, temperature_series):
        length = len(temperature_series)
        core = "2020-" + str(month) + "-1"
        date_list = []
        for i in range(length):
            date_list.append(core)
        output = pd.to_datetime(date_list)
        return output

    def Mega_physical_model_optimization(self, cn, lamda, month,
                                         precipitation_series,
                                         temperature_series,
                                         initial_uc,
                                         initial_sc):
        u_initial = initial_uc * self.sat_u * 0.1
        s_initial = initial_sc * self.sat_u
        
        cn2 = cn
        len_number = len(temperature_series)
        q = []
        u_series = []
        s_series = []
        ev_s = []
        
        date_series = self.generate_month_date_series_optimization(month, 
                                                                   temperature_series)
    
        def get_cs(month):
            if month <= 11 and month >=5 :
                return self.cs_1
            else:
                return self.cs_2
        
        for i in range(len_number):
            date = date_series[i]
            temperature = temperature_series[i]
            month = date.month
            cs = get_cs(month)
            lamda = lamda

            precipitation = precipitation_series[i]
            
            if i == 0:
                u = u_initial
                s = s_initial
                
                amc_cls = self.AMS_classification_opt(date, precipitation_series, i)
                cn = self.CN_caculation(amc_cls, cn2)
                
                y = self.partial_physical_model_v2(date, precipitation,
                                                   temperature, u, s, cn, lamda)
                qs = round(cs * y[2]*763.4*10000/86400,7)
                q.append(qs)
                u_series.append(y[0])
                s_series.append(y[1])
                ev_t = round(y[3] * 763.4 * 10000 / 86400, 5)
                ev_s.append(ev_t)
            else :
                u = u_series[i-1] 
                s = s_series[i-1]
                
                amc_cls = self.AMS_classification_opt(date = date,
                                                     precipitation_series = precipitation_series,
                                                     i = i)
                cn = self.CN_caculation(amc_cls,cn2)
                
                y = self.partial_physical_model_v2(date,precipitation,
                                                   temperature,u,s,cn, lamda)
                qs = round(cs*y[2]*763.4*10000/86400,7)
                q.append(qs)
                u_series.append(y[0])
                s_series.append(y[1])
                ev_t = round(y[3] * 763.4 * 10000 / 86400, 5)
                ev_s.append(ev_t)               
                
        q = np.asarray(q, dtype = np.float64)
        u_series = np.asarray(u_series, dtype = np.float64)
        s_series = np.asarray(s_series, dtype = np.float64)
        ev_s = np.asarray(ev_s, dtype = np.float64)
            
        return q, ev_s 
            
    def AMS_classification_v2(self, date, precipitation_series):
        if date.month <= 10 and date.month >= 5:
            grow_time = True
        else:
            grow_time = False
    
        date_number = self.date_series_to_date_number_v2(date)
        p5 = self.caculate_permutation_five_days(precipitation_series, date_number)
        
        if grow_time is True:
            if p5 < 1.4*2.54:
                
                return "AMC_1"
            
            if p5 >= 1.4*2.54 and p5 <= 2.1*2.54:
                
                return "AMC_2"
            else:
                return "AMC_3"
        else:
            if p5 < 0.5*2.54:
                return "AMC_1"
            if p5 >= 0.5*2.54 and p5 <= 1.1*2.54:
                return "AMC_2"
            else:
                return "AMC_3"

    def AMS_classification_opt(self, date, precipitation_series, i ):
        if date.month <= 10 and date.month >= 5:
            grow_time = True
        else:
            grow_time = False
    
        date_number = i
        p5 = self.caculate_permutation_five_days(precipitation_series, date_number)
        

    
        if grow_time is True:
            if p5 < 1.4*2.54:
                
                return "AMC_1"
            
            if p5 >= 1.4*2.54 and p5 <= 2.1*2.54:
                
                return "AMC_2"
            else:
                return "AMC_3"
        else:
            if p5 < 0.5*2.54:
                return "AMC_1"
            if p5 >= 0.5*2.54 and p5 <= 1.1*2.54:
                return "AMC_2"
            else:
                return "AMC_3"

    def date_series_to_date_number_v2(self, date):
        year = date.year
        month = date.month
        day = date.day
        date_series = self.date_series
        y0 = date_series[0].year
        m0 = date_series[0].month
        d0 = date_series[0].day
        
        d1 = datetime.datetime(year,month, day)
        d2 = datetime.datetime(y0,m0,d0)
        
        return int((d1 - d2).days)
                
    def CN_caculation(self,AMS_CLS,cn):
        if AMS_CLS == 'AMC_1':
            ans = 4.2*(cn)/(10-0.058*cn)#<<<<<<<<<<<<<<<<<<<<<<<<<<<單位議題#####
            return ans
        elif AMS_CLS == 'AMC_3':
            ans = 23*cn/(10 + 0.13*cn)#<<<<<<<<<<<<<<<<<<<<<<<<<<<單位議題#####
            return ans
        else:
            return cn  
        
    def caculate_permutation_five_days(self,precipitation_series,date_number):
        if date_number < 4:
            answer = precipitation_series[date_number] * 0.7 + 0.05 #<<<<<<<<<<<< 數值處理議題
        else:
            answer = 0
            for i in range(4):
                answer += precipitation_series[date_number - i]
        return answer

class climate_hydrological_simulation:
    def __init__(self,file_path_data_sheet,
                 file_path_tem,
                 file_path_total_data,
                 file_path_ar
                 ):
        rd = read_inflow_estimate_data(file_path_total_data,
                                       file_path_tem, 
                                       file_path_data_sheet, 
                                       file_path_ar)
        self.precipitation_record_list = rd.get_month_daily_precipitation_data()
        p_data_body = rd.get_month_daily_precipitation_data_v2()
        
        self.precipitation_binirization = p_data_body[0]
        self.precipitation_excluded_zero = p_data_body[1]
        
        self.temperature_record_list = rd.get_month_temperature_data()
        
        self.p_ar = precipitation_ar_model(self.precipitation_binirization,
                                           self.precipitation_excluded_zero)
        
        self.t_ar = temperature_ar_model(self.temperature_record_list)

    def generate_date_series(self, month):
        day_list = [31,28,31,30,31,30,31,31,30,31,30,31]
        year = datetime.datetime.now().year
        start = str(year) + '/' + str(month) +'/'+ '1'
        end = str(year) + '/' + str(month) +'/'+ str(day_list[month - 1])
        
        date_range = pd.date_range(start,end, freq = 'D')
        
        date_series = pd.Series(date_range)
        
        return date_series
    
    def hydrological_simulation(self, month):
        date_series = self.generate_date_series(month)
        precipitation_series = self.p_ar.get_precipitation_sampling_v1(month)/10
        temperature_series = self.t_ar.get_temperature_sampling(month)
        
        model = inflow_estimate(date_series,
                                precipitation_series, 
                                temperature_series)
        
        streamflow, evapotranspiration = model.Mega_physical_model_v3()
        
        return temperature_series, precipitation_series, streamflow, evapotranspiration
        
    def hydrological_simulation_v2(self, month):
        date_series = self.generate_date_series(month)
        precipitation_series = self.p_ar.get_precipitation_sampling_v2(month)/10
        temperature_series = self.t_ar.get_temperature_sampling(month)
        
        model = inflow_estimate(date_series,
                                precipitation_series, 
                                temperature_series)
        
        streamflow, evapotranspiration = model.Mega_physical_model_v3()
        
        return temperature_series, precipitation_series, streamflow, evapotranspiration        
        
class catcharea_state:
    def __init__(self ,up_target_storage_limit, resilience, bier):
        self.up_target_storage_limit = up_target_storage_limit
        self.resilience = resilience
        self.bier = bier
        storage_low_bond = [17500,15000,12500,10000,8000,8000,8000,10000,12500,15000,17500, 17500]
        storage_elow_bond = [9000,7500,6000,5000,4000,4000,4000,5000,6000,7500,9000,9000]
        
        self.storage_low_bond = np.array(storage_low_bond)/86400 * 10000
        self.storage_elow_bond = np.array(storage_elow_bond)/86400 * 10000
        
    def get_cta(self, consum, inflow, storage_state, water_translate = 0):
        inflow = np.sum(inflow) + water_translate
        return np.round(np.divide(consum, (inflow + 2* storage_state)), 5)
      
    def get_WDI(self, cta):
        return np.round(np.reciprocal((1 + (1/0.01 - 1) * np.exp(-1*self.resilience * cta))) , 5)
    
    def get_RFD(self, consum, wdi, ev, month):
        ev = np.sum(ev)
        return np.round((consum - ev * self.bier[month-1]) * wdi, 5) 
    
    def get_cta_mdp(self, consum, inflow, storage_state, water_translate = 0):
        inflow = inflow + water_translate
        return np.round(np.divide(consum, (inflow + 2* storage_state)), 5)
  
    def get_WDI_mdp(self, cta):
        return np.round(np.reciprocal((1 + (1/0.01 - 1) * np.exp(-1*self.resilience * cta))) , 5)
    
    def get_RFD_mdp(self, consum, wdi, ev, month):
        return np.round((consum - ev * self.bier[month-1]) * wdi, 5)    
    
    def reservoir_operation(self, inflow_array, es_array, 
                            past_storage, consum, correct,
                            discharge):
        sum_es = np.sum(es_array)
        sum_inflow = np.sum(inflow_array)
        total_change = (sum_inflow - consum - discharge + correct)
        overflow = 0
        
        if total_change + past_storage < 0:
            total_change = -1 * past_storage
            end_storage = max(past_storage + total_change, 0)
        
        elif total_change + past_storage > self.up_target_storage_limit:
            end_storage = self.up_target_storage_limit
            overflow = total_change + past_storage - self.up_target_storage_limit
            
        else:
            end_storage = max(past_storage + total_change, 0)
        
        eff_consum = consum
        storage_state = (end_storage + past_storage)/2
        
        return end_storage, eff_consum, sum_inflow, overflow, sum_es, storage_state

    def reservoir_operation_under_rule(self, inflow_array, es_array, past_storage, consum, month, 
                                       correct, discharge):
        end_storage, consum, inflow, overflow, sum_es = self.reservoir_operation(inflow_array,
                                                                                 es_array,
                                                                                 past_storage,
                                                                                 consum, 
                                                                                 correct,
                                                                                 discharge)
        if end_storage < self.storage_low_bond[int(month - 1)]:
            consum = 0.75*consum
        
        elif end_storage < self.storage_elow_bond[int(month - 1)]:
            consum = 0.5*consum
        
        end_storage, consum, inflow, overflow, sum_es = self.reservoir_operation(inflow_array, 
                                                                                 es_array,
                                                                                 past_storage, 
                                                                                 consum)
        storage_state = (end_storage + past_storage)/2
        
        return end_storage, consum, inflow, overflow, sum_es, storage_state
        
    def continue_base_state_update(self, 
                                   past_storage, 
                                   es_array, 
                                   inflow_array,
                                   consum,
                                   month,
                                   correct,
                                   discharge):
        end_storage, consum, inflow, overflow, sum_es, storage_state = self.reservoir_operation(inflow_array,
                                                                                                es_array, 
                                                                                                past_storage, 
                                                                                                consum, 
                                                                                                correct,
                                                                                                discharge)
        sum_inflow = np.sum(inflow_array)            
        sum_inflow = sum_inflow - overflow
            
        cta = self.get_cta(consum, sum_inflow, storage_state)
        wdi = self.get_WDI(cta)
        rfd = self.get_RFD(consum, wdi, sum_es, month)
            
        update_state = tuple([cta, wdi, rfd, end_storage, overflow])
            
        return update_state

    def continue_base_state_update_under_rule(self, 
                                              past_storage, 
                                              es_array, 
                                              inflow_array,
                                              consum,
                                              month,
                                              correct,
                                              discharge):
        end_storage, consum, inflow, overflow, sum_es, storage_state = self.reservoir_operation_under_rule(inflow_array,
                                                                                                           es_array, 
                                                                                                           past_storage, 
                                                                                                           consum, 
                                                                                                           month,
                                                                                                           correct,
                                                                                                           discharge)
        sum_inflow = np.sum(inflow_array)            
        sum_inflow = sum_inflow - overflow
        eff_consum = consum 
            
        cta = self.get_cta(eff_consum, sum_inflow, storage_state)
        wdi = self.get_WDI(cta)
        rfd = self.get_RFD(consum, wdi, sum_es, month)
            
        update_state = tuple([cta, wdi, rfd, end_storage, overflow])
            
        return update_state

class pair_modeling:
    def __init__(self, array_list):
        self.array_list = array_list
        self.array_mean_list = np.mean(array_list, axis = 2)
        
        regress_params_list = []
        noise_list = []
        pearsonr_list =[]
        for i in range(12):
            regress_params_list.append(self.get_month_regression_storage_to_correction(i+1))
            noise_list.append(self.regression_noise(i+1))
            pearsonr_list.append(self.get_pearsonr(i+1))
            
        self.regress_params = regress_params_list
        self.noise_list = noise_list
        self.pearsonr_list = pearsonr_list
        
        hyf = hydraulic_freq_analyse(noise_list)
        self.noise_distribution = hyf.get_suitable_distribution_list()
        
    def get_pearsonr(self, month):
        x = self.array_list[1][month - 1]
        y = self.array_list[0][month - 1]
        return st.pearsonr(x,y)[0]
    
    def get_month_regression_storage_to_correction(self, month):
        x = self.array_list[1][month - 1]
        y = self.array_list[0][month - 1]        
        slope, intercept, r, p, se = st.linregress(x, y)
        return slope, intercept
    
    def regression_noise(self, month):
        x = self.array_list[1][month - 1]
        y = self.array_list[0][month - 1]
        slope, intercept = self.get_month_regression_storage_to_correction(month)
        y_p = slope * x + intercept
        noise = -1 * abs(y-y_p)
        return noise

class dual_system_simulation:
    def __init__(self, file_path_data_sheet,
                 file_path_tem,
                 file_path_total_data,
                 file_path_ar,
                 up_target_storage_limit = 3000):
        rd = read_inflow_estimate_data(file_path_total_data, 
                                       file_path_tem, 
                                       file_path_data_sheet, 
                                       file_path_ar)
        self.read_data = rd
        self.bier = self.read_data.get_BIER()
        self.rfd_threshold = self.read_data.get_RFD_threshold()
        
        self.hydrological_model = climate_hydrological_simulation(file_path_data_sheet,
                                                                  file_path_tem, 
                                                                  file_path_total_data, 
                                                                  file_path_ar)
        
        cta_series = self.read_data.get_cta_data_frame()["cta"]
        self.resilience = np.log(99) / np.median(cta_series)
        self.dual_system_update = catcharea_state(up_target_storage_limit,
                                                  self.resilience, 
                                                  self.bier)
        
        self.storage_capacity = up_target_storage_limit

        self.inflow_simulation_monthly = self.get_annual_inflow_simulation()
        self.monthly_inflow_record = self.read_data.get_monthly_inflow_data()
        
        self.inflow_simulation_monthly = self.get_annual_inflow_simulation()
        infs_hyf = hydraulic_freq_analyse(self.inflow_simulation_monthly)
        self.inflow_s_dis_list = infs_hyf.get_suitable_distribution_list()
        self.es_simulation_monthly = self.get_annual_es_simulation()
        ess_hyf = hydraulic_freq_analyse(self.es_simulation_monthly)
        self.es_s_dis_list = ess_hyf.get_suitable_distribution_list()

        mean_discharge_array, discharge_inflow_pair = rd.get_monthly_discharge_correction()
        self.discharge_array = mean_discharge_array
        self.discharge_inflow_pair = discharge_inflow_pair
        
        discharge_modeling = pair_modeling(self.discharge_inflow_pair)
        self.discharge_modeling = discharge_modeling
        self.discharge_noise_distribution_list = discharge_modeling.noise_distribution
        self.discharge_regress_inflow = discharge_modeling.regress_params

        mean_correct_array, correct_pair = rd.get_monthly_correction_storage()
        self.storage_correct_array = mean_correct_array
        self.correct_pair = correct_pair
        
        storage_correct = pair_modeling(self.correct_pair)
        self.storage_correct = storage_correct
        self.noise_distribution_list = storage_correct.noise_distribution
        self.regress_s_to_correction = storage_correct.regress_params

    def storage_change_correction(self, month, inflow):
        slope, intercept = self.regress_s_to_correction[month - 1]
        noise = self.random_correct_noise(month, self.noise_distribution_list)
        return (inflow * slope + intercept  + noise)

    def discharge_simulation(self, month, inflow):
        slope, intercept = self.discharge_regress_inflow[month - 1]
        noise = self.random_correct_noise(month, self.discharge_noise_distribution_list)
        return abs(inflow * slope + intercept  + noise)

    def random_correct_noise(self, month, noise_distribution_list):
        name =  noise_distribution_list[month - 1][0]
        params = noise_distribution_list[month - 1][1]
       
        if name == 'norm':
            noise = st.norm.rvs(loc = params[0], scale = params[1])
        
        elif name == 'gamma':
            noise = st.gamma.rvs(a = params[0], loc = params[1], scale = params[2])
    
        elif name == 'gumbel_r':
            noise = st.gumbel_r.rvs(loc = params[0], scale = params[1])
            
        elif name == 'gumbel_l':
            noise = st.gumbel_l.rvs(loc = params[0], scale = params[1])     
        
        elif name == 'lognorm':
            noise = st.lognorm.rvs(s = params[0], loc = params[1], scale = params[2])          

        elif name == 'pearson3':
            noise = st.pearson3.rvs(skew = params[0], loc = params[1], scale = params[2])        

        elif name == 'loggamma':
            noise = st.pearson3.rvs(skew = params[0], loc = params[1], scale = params[2])          

        return noise            
        
    def mcdbdi_drought_real_time(self, month, consum, past_storage,
                                 simulation_time = 1000, wdi_c = 0.85):
        count_rfd = 0
        count_cwdi = 0
        count_fail = 0
        threshold = self.rfd_threshold[month - 1]
        rfd_record = []
        wdi_record = []
        bier = self.dual_system_update.bier
        
        for i in range(simulation_time):
            climate_body = self.hydrological_model.hydrological_simulation(month)
            inflow = climate_body[2]
            ev = climate_body[3]
            discharge = self.discharge_simulation(month, np.sum(inflow))
            correct = self.storage_change_correction(month, np.sum(inflow))
            cta, cwdi, rfd, end_storage, overflow = self.dual_system_update.continue_base_state_update(past_storage, 
                                                                                                      ev, 
                                                                                                      inflow, 
                                                                                                      consum, 
                                                                                                      month,
                                                                                                      correct,
                                                                                                      discharge)
            rfd_record.append(rfd)
            wdi_record.append(cwdi)
            er = sum(ev) * bier[month - 1]
            
            if rfd > threshold:
                count_rfd = count_rfd + 1
            
            if cwdi > wdi_c:
                count_cwdi = count_cwdi + 1
            
            if (rfd > threshold or consum <= threshold / cwdi + er) and cwdi > wdi_c :
                count_fail = count_fail + 1
            
            else:
                continue

        prob_rfd = count_rfd / simulation_time
        prob_cwdi = count_cwdi / simulation_time
        prob_fail = count_fail / simulation_time
        return prob_rfd, prob_cwdi, prob_fail

    def mcdbdi_drought_real_time_v2(self, month, consum, past_storage,
                                 simulation_time = 1000, wdi_c = 0.85):
        count_rfd = 0
        count_cwdi = 0
        count_fail = 0
        threshold = self.rfd_threshold[month - 1]
        
        rfd_record = []
        wdi_record = []
        bier = self.dual_system_update.bier
        for i in range(simulation_time):
            climate_body = self.hydrological_model.hydrological_simulation_v2(month)
            inflow = climate_body[2]
            ev = climate_body[3]
            discharge = self.discharge_simulation(month, np.sum(inflow))
            correct = self.storage_change_correction(month, np.sum(inflow))           
            cta, cwdi, rfd, end_storage, overflow = self.dual_system_update.continue_base_state_update(past_storage, 
                                                                                                      ev, 
                                                                                                      inflow, 
                                                                                                      consum, 
                                                                                                      month,
                                                                                                      correct,
                                                                                                      discharge)
            rfd_record.append(rfd)
            wdi_record.append(cwdi)
            er = sum(ev) * bier[month - 1]
            if rfd > threshold :
                count_rfd = count_rfd + 1
            
            if cwdi > wdi_c:
                count_cwdi = count_cwdi + 1
            
            if (rfd > threshold or consum <= threshold / cwdi + er) and cwdi > wdi_c :
                count_fail = count_fail + 1
            
            else:
                continue  
        prob_rfd = count_rfd / simulation_time
        prob_cwdi = count_cwdi / simulation_time
        prob_fail = count_fail / simulation_time
        return prob_rfd, prob_cwdi, prob_fail

    def get_seasonal_risk_map(self, 
                              month,
                              consumption_limit,
                              max_storage,
                              min_storage,
                              resolution):
        rfd_m = np.ones(shape = [resolution, resolution])
        cwdi_m = np.ones(shape = [resolution, resolution])
        fail_m = np.ones(shape = [resolution, resolution])        

        consumption_array = np.zeros(shape = resolution)
        storage_array = np.zeros(shape = resolution)
        c_step = consumption_limit / resolution
        s_step = (max_storage - min_storage) / resolution        
        
        for i in range(resolution):
            storage_array[i] = min_storage + i * s_step
            for j in range(resolution):
                consumption_array[j] = c_step * (j + 1)                
                rfd_m[i][j],cwdi_m[i][j],fail_m[i][j] = self.mcdbdi_drought_real_time(month, 
                                                                                      consumption_array[j],
                                                                                      storage_array[i]
                                                                                      )
        return rfd_m, cwdi_m, fail_m, consumption_array, storage_array

    def get_seasonal_risk_map_v2(self, 
                                 month,
                                 consumption_limit,
                                 max_storage,
                                 min_storage,
                                 resolution):
        rfd_m = np.ones(shape = [resolution, resolution])
        cwdi_m = np.ones(shape = [resolution, resolution])
        fail_m = np.ones(shape = [resolution, resolution])        

        consumption_array = np.zeros(shape = resolution)
        storage_array = np.zeros(shape = resolution)
        c_step = consumption_limit / resolution
        s_step = (max_storage - min_storage) / resolution        
        
        for i in range(resolution):
            storage_array[i] = min_storage + i * s_step
            for j in range(resolution):
                consumption_array[j] = c_step * (j + 1)  
                rfd_m[i][j],cwdi_m[i][j],fail_m[i][j] = self.mcdbdi_drought_real_time_v2(month, 
                                                                                         consumption_array[j],
                                                                                         storage_array[i]
                                                                                         )
        return rfd_m, cwdi_m, fail_m, consumption_array, storage_array
                        
    def get_annual_inflow_simulation(self, sample_size = 200):
        monthly_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(12):
            for j in range(sample_size):
                inflow_array = self.hydrological_model.hydrological_simulation(i +1)[2]
                inflow = np.sum(inflow_array)
                monthly_list[i].append(inflow)
        return np.array(monthly_list)

    def get_annual_es_simulation(self, sample_size = 200):
        monthly_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(12):
            for j in range(sample_size):
                es_array = self.hydrological_model.hydrological_simulation(i +1)[3]
                es = np.sum(es_array)
                monthly_list[i].append(es)
        
        return np.array(monthly_list)

def risk_map_read(file_path_riskmap_p):
    output = []
    for i in range(12):
        file_path = file_path_riskmap_p+"riskmap_"+str(i+1)+".npy"
        temp = np.load(file_path)
        output.append(temp)
    return np.array(output)
            
dual_system = dual_system_simulation(file_path_data_sheet,
                              file_path_tem, 
                              file_path_total_data,
                              file_path_ar)

annual_riskmap = risk_map_read(file_path_riskmap_p)

# print(dual_system.mcdbdi_drought_real_time(8, 1000, 500))
