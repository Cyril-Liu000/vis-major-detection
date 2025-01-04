6# -*- coding: utf-8 -*-
"""
Created on Fri May 24 22:38:44 2024

@author: Cyril Liu
"""

import scipy.stats as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib.cm as cm
import learn_ar_model as lm
from matplotlib import gridspec
import seaborn as sns


file_path_data_sheet = 'data_sheet.xlsx'
file_path_tem = '石門站溫度資料.txt'
file_path_total_data = '石門水庫雨量與流量資料.xlsx'
file_path_ar = '日雨量溫度資料.xlsx'
file_path_climate_change = '雨量與溫度變化預估.xlsx'
file_path_riskmap_p = "C:/Users/Cyril Liu/Desktop/Papper-Socioeconomic Drought/Paper of MCDBDI/"
risk_map = lm.annual_riskmap
consumption_limit = 1500
max_storage = 3000
min_storage = 150
resolution = 20
discount_factors = [0.215, 0.465, 0.681]

df_r = lm.dual_system.read_data.get_storage_correction_inflow_consum_dataframe()

def remake_df(df_r):
    date_series = df_r["date"]
    inflow_series = df_r["inflow"]
    storage_series = df_r["storage"]
    correction_series = df_r["correction"]
    consumption_series = df_r["consumption"]
    rfd_series = df_r["rfd"]
    threshold_series = df_r["threshold"]
    cwdi_series = df_r["cwdi"]
    imsrri_6 = df_r["imsrri_6"]
    pre_storage_series = df_r["pre_storage"]    
    df_out = pd.DataFrame()
    df_out.insert(loc = 0, column = "date", value = date_series)
    df_out.insert(loc = 1, column = "inflow", value = inflow_series)
    df_out.insert(loc = 2, column = "pre_storage", value = pre_storage_series)
    df_out.insert(loc = 3, column = "storage", value = storage_series)
    df_out.insert(loc = 4, column = "correction", value = correction_series)
    df_out.insert(loc = 5, column = "consumption", value = consumption_series)
    df_out.insert(loc = 6, column = "rfd", value = rfd_series)
    df_out.insert(loc = 7, column = "threshold",value = threshold_series)
    df_out.insert(loc = 8, column = "cwdi", value = cwdi_series)
    df_out.insert(loc = 9, column = "imsrri_6", value = imsrri_6)
    df_out = df_out.dropna()
    df_out = df_out.reset_index(drop = True)
    return df_out       

def model_setting(consumption_limit, max_storage, min_storage, resolution):
    
    consumption_array = np.zeros(shape = resolution)
    storage_array = np.zeros(shape = resolution)
    c_step = consumption_limit / resolution
    s_step = (max_storage - min_storage) / resolution        
    
    for i in range(resolution):
            
        consumption_array[i] = c_step * (i + 1)
        storage_array[i] = min_storage + s_step * i

    return consumption_array, storage_array

model_set = model_setting(consumption_limit, max_storage, min_storage, resolution)

def get_demand(file_path_data_sheet):
    
    data_frame = pd.DataFrame(pd.read_excel(file_path_data_sheet))
    date_series = pd.Series(data_frame["date_month"])
    
    c_series = pd.Series(data_frame["總引水量(C)"])
    record = pd.Series(data_frame["乾旱紀錄"])
    
    dataframe_1 = pd.DataFrame()
    
    dataframe_1.insert(loc = 0, column = "date", value = date_series)
    dataframe_1.insert(loc = 1, column = "C", value = c_series)
    dataframe_1.insert(loc = 2, column = "record", value = record)
    
    dataframe_1 = dataframe_1.dropna(thresh = 1)
    
    month_demand = np.zeros(shape = [12, 1])
    month_count = np.zeros(shape = [12, 1])
    
    for i in range(len(dataframe_1)):
        
        for j in range(len(month_demand)):
            
            if j + 1 == dataframe_1["date"][i].month:
                if dataframe_1["record"][i] != 1:
                    month_demand[j] = month_demand[j] + dataframe_1["C"][i]
                    month_count[j] = month_count[j] + 1
                    
                else:
                    continue
    return (month_demand/ month_count).reshape([12]) * 1.25

demand_list = get_demand(file_path_data_sheet)

def get_annual_risk_map(consumption_limit, 
                        max_storage,
                        min_storage, 
                        resolution):
    
    output = []
    for i in range(12):
        temp = lm.dual_system.get_seasonal_risk_map(i+1,
                                                    consumption_limit, 
                                                    max_storage, 
                                                    min_storage, 
                                                    resolution)[2]
        temp = np.rot90(temp.T)
        output.append(temp)
    return np.array(output)

def get_policy(policy_name):
    file_path = file_path_riskmap_p + "/policy/" + policy_name + ".npy"
    output = np.load(file_path)
    return output
            
class env_setting:
    
    def __init__(self, annual_risk_map, model_set, initial_storage, 
                 initial_month, demand_list ,c_punish_factor = 0, s_gain_factor = 0):
        
        self.consumptions = model_set[0]
        self.storages = model_set[1]
        self.annual_risk_map = annual_risk_map
        self.statecount = len(model_set[0])*len(model_set[1])
        self.actioncount = len(self.consumptions)
        self.initial_storage = initial_storage
        self.initial_month = initial_month
        self.c_punish_factor = c_punish_factor
        self.s_gain_factor = s_gain_factor
        self.demand_list = demand_list
        
        es_expect_list = []
        for i in range(12):
            es_expect_list.append(self.es_expected_value(i+1))
        
        self.es_expect_array = np.array(es_expect_list)
        
        inflow_expect_list = []
        for i in range(12):
            inflow_expect_list.append(self.inflow_expected_value(i+1))
        
        self.inflow_expect_array = np.array(inflow_expect_list)
        self.correct_array = lm.dual_system.storage_correct_array
        self.capacity = lm.dual_system.storage_capacity
        self.s_to_correct_params = lm.dual_system.regress_s_to_correction
        self.noise_distribution_list = lm.dual_system.noise_distribution_list
    
    def random_correct_noise(self, month):
   
        name =  self.noise_distribution_list[month - 1][0]
        params = self.noise_distribution_list[month - 1][1]
       
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
    
    def storage_change_correction(self, month, storage):
        
        slope, intercept = self.s_to_correct_params[month - 1]
        noise = self.random_correct_noise(month)
        
        return -1 * abs(storage * slope + intercept  + noise)
          
    def es_expected_value(self, month):
        
        name = lm.dual_system.es_s_dis_list[month - 1][0]
        params = lm.dual_system.es_s_dis_list[month - 1][1]
        
        if name == 'norm':
        
            es = st.norm.mean(loc = params[0], scale = params[1])
        
        elif name == 'gamma':
        
            es = st.gamma.mean(a = params[0], loc = params[1], scale = params[2])
    
        elif name == 'gumbel_r':
            
            es = st.gumbel_r.mean(loc = params[0], scale = params[1])    
      
        elif name == 'gumbel_l':
            
            es = st.gumbel_l.mean(loc = params[0], scale = params[1])        
        
        elif name == 'lognorm':

            es = st.lognorm.mean(s = params[0], loc = params[1], scale = params[2])          
        
        elif name == 'pearson3':

            es = st.pearson3.mean(skew = params[0], loc = params[1], scale = params[2])        

        elif name == 'loggamma':
            
            es = st.loggamma.mean(c = params[0], loc = params[1], scale = params[2])          
    
        
        return max(es, 0)
   
    def inflow_expected_value(self, month):
        
        name = lm.dual_system.inflow_s_dis_list[month - 1][0]
        params = lm.dual_system.inflow_s_dis_list[month - 1][1]
        
        if name == 'norm':
        
            inflow = st.norm.mean(loc = params[0], scale = params[1])
        
        elif name == 'gamma':
        
            inflow = st.gamma.mean(a = params[0], loc = params[1], scale = params[2])
    
        elif name == 'gumbel_r':
            
            inflow = st.gumbel_r.mean(loc = params[0], scale = params[1])    
      
        elif name == 'gumbel_l':
            
            inflow = st.gumbel_l.mean(loc = params[0], scale = params[1])        
        
        elif name == 'lognorm':

            inflow = st.lognorm.mean(s = params[0], loc = params[1], scale = params[2])          
        
        elif name == 'pearson3':

            inflow = st.pearson3.mean(skew = params[0], loc = params[1], scale = params[2])        

        elif name == 'loggamma':
            
            inflow = st.loggamma.mean(c = params[0], loc = params[1], scale = params[2])          
    
        
        return max(inflow, 0)

    def reward_supply_capacity(self, consumption, probability_s):
        
        return probability_s * consumption
    
    def reward_demand_satisfied(self, consumption, demand, probability_s):
        
        output = abs((consumption - demand + 0.12) / demand )
        
        return probability_s / output * consumption

    def reward_counter_shortage(self, consumption, demand):
        
        output =  abs((consumption - demand + 0.12) / demand)
        
        return 1 / output * demand

# The potential supply capacity is the measurement to evaluate
# water resource value in dual system. The counter shortage is maximum 

    def storage_update(self, month, pre_storage, consumption, inflow, change_correct):
   
        capacity = self.capacity
        consumption_boundary = min(max(pre_storage + inflow, 0), consumption)
        
        if pre_storage + inflow - consumption_boundary + change_correct > capacity:
            return max(capacity, 0 )
        
        else:
            return max(pre_storage + inflow - consumption_boundary + change_correct, 0)

    def random_inflow(self, month):
   
        name =  lm.dual_system.inflow_s_dis_list[month - 1][0]
        params = lm.dual_system.inflow_s_dis_list[month - 1][1]
       
        if name == 'norm':
        
            inflow = st.norm.rvs(loc = params[0], scale = params[1])
        
        elif name == 'gamma':
        
            inflow = st.gamma.rvs(a = params[0], loc = params[1], scale = params[2])
    
        elif name == 'gumbel_r':
            
            inflow = st.gumbel_r.rvs(loc = params[0], scale = params[1])
            
        elif name == 'gumbel_l':
            
            inflow = st.gumbel_l.rvs(loc = params[0], scale = params[1])     
        
        elif name == 'lognorm':

            inflow = st.lognorm.rvs(s = params[0], loc = params[1], scale = params[2])          

        elif name == 'pearson3':

            inflow = st.pearson3.rvs(skew = params[0], loc = params[1], scale = params[2])        

        elif name == 'loggamma':
            
            inflow = st.pearson3.rvs(skew = params[0], loc = params[1], scale = params[2])          

        return inflow


    def random_es(self, month):
   
        name =  lm.dual_system.es_s_dis_list[month - 1][0]
        params = lm.dual_system.es_s_dis_list[month - 1][1]
       
        if name == 'norm':
        
            es = st.norm.rvs(loc = params[0], scale = params[1])
        
        elif name == 'gamma':
        
            es = st.gamma.rvs(a = params[0], loc = params[1], scale = params[2])
    
        elif name == 'gumbel_r':
            
            es = st.gumbel_r.rvs(loc = params[0], scale = params[1])
            
        elif name == 'gumbel_l':
            
            es = st.gumbel_l.rvs(loc = params[0], scale = params[1])     
        
        elif name == 'lognorm':

            es = st.lognorm.rvs(s = params[0], loc = params[1], scale = params[2])          

        elif name == 'pearson3':

            es = st.pearson3.rvs(skew = params[0], loc = params[1], scale = params[2])        

        elif name == 'loggamma':
            
            es = st.pearson3.rvs(skew = params[0], loc = params[1], scale = params[2])          

        return es

    
    def randant_choice(self, storage, inflow, month):
        
        consumption_boundary = max(storage + inflow, 0)
        max_indice = np.argmin(abs(consumption_boundary - self.consumptions))
        
        return np.random.choice(self.consumptions[range(max_indice + 1)])
        
        
    def reset(self):
        
        initial_indice_s = np.argmin(abs(self.initial_storage - self.storages))
        es = self.random_es(self.initial_month)
        inflow = self.random_inflow(self.initial_month)

        return  self.storages[initial_indice_s], self.initial_month, inflow, es

    def reset_random(self):
        
        initial_indice_s = np.random.choice(range(20))
        es = self.random_es(self.initial_month)
        inflow = self.random_inflow(self.initial_month)
        
        return  self.storages[initial_indice_s], self.initial_month, inflow, es  

    def step_update_mdp(self, consumption, storage, month, reward_type = 0):
        
        threshold = lm.dual_system.rfd_threshold[month - 1]
        inflow = self.random_inflow(month)
        es = self.random_es(month)
        change_correct = self.storage_change_correction(month, storage)
        consumption_boundary = min(max(inflow + storage,0), consumption)
        demand = self.demand_list[month - 1]
        
        next_storage = self.storage_update(month, storage, consumption_boundary, inflow, change_correct)
        storage_state = (storage + next_storage)/2
     
        cta = lm.dual_system.dual_system_update.get_cta_mdp(consumption_boundary, inflow, storage_state)
        wdi = lm.dual_system.dual_system_update.get_WDI_mdp(cta)
        rfd = lm.dual_system.dual_system_update.get_RFD_mdp(consumption_boundary, wdi, es, month)

        def get_next_month(month):
            
            if month % 12 == 0:
                
                return 12
            
            else:
                
                return month % 12
        
        next_month = get_next_month(month + 1)        
        storage_gain = (next_storage - storage)*self.s_gain_factor
        consumption_punish = min(consumption_boundary - consumption, 0)  * self.c_punish_factor
        reward_con = storage_gain + consumption_punish
        
        if rfd >= threshold and wdi > 0.85:
            reward_c = 0
        else:
            reward_c = consumption_boundary
        
        if reward_type == 0:
            
            reward = reward_c + reward_con
        
        elif reward_type == 1:
            
            reward = abs((demand/(demand - consumption_boundary))) * reward_c + reward_con
        
        elif reward_type == 2:
            
            reward = abs(demand/(demand - consumption_boundary)) * demand + reward_con
            
        return next_storage, reward, consumption_boundary, storage, next_month, inflow, es        
        
class Q_learning:
    
    def __init__(self, epochs, count_step, eplison, decay, learning_rate
                 ,annual_risk_map, model_set, initial_storage, initial_month, 
                 discount_factor):
        
        self.epochs = epochs
        self.count_step = count_step
        self.eplison = eplison
        self.decay = decay
        self.env = env_setting(annual_risk_map, 
                               model_set, 
                               initial_storage, 
                               initial_month,
                               demand_list,
                               s_gain_factor = 0
                               )

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.storage_elimit = lm.dual_system.dual_system_update.storage_elow_bond
        self.storage_limit = lm.dual_system.dual_system_update.storage_low_bond
        
    def initial_annual_q_table(self, model_set):
        
        annual_q_table = []
        
        for i in range(12):
            
            q_table = 0.1*st.uniform.rvs(size = [len(model_set[1]), len(model_set[0])])
#            q_table = np.zeros(shape = [len(model_set[1]), len(model_set[0])])
            annual_q_table.append(q_table)
        
        return np.array(annual_q_table)

    def count_annual_table(self, model_set):
        
        annual_q_table = []
        
        for i in range(12):
            
            q_table = np.zeros(shape = [len(model_set[1]), len(model_set[0])])
            annual_q_table.append(q_table)
        
        return np.array(annual_q_table)

    def consumption_indice(self, consumption):
        
        return np.argmin(abs(consumption - self.env.consumptions))

    def storage_indice(self, storage):
        
        return np.argmin(abs(storage - self.env.storages))

    def inflow_ppf(self, month, p):
   
        name =  lm.dual_system.inflow_s_dis_list[month - 1][0]
        params = lm.dual_system.inflow_s_dis_list[month - 1][1]
       
        if name == 'norm':
        
            inflow = st.norm.ppf(q = p,loc = params[0], scale = params[1])
        
        elif name == 'gamma':
        
            inflow = st.gamma.ppf(q = p, a = params[0], loc = params[1], scale = params[2])
    
        elif name == 'gumbel_r':
            
            inflow = st.gumbel_r.ppf(q = p,loc = params[0], scale = params[1])
            
        elif name == 'gumbel_l':
            
            inflow = st.gumbel_l.ppf(q = p,loc = params[0], scale = params[1])     
        
        elif name == 'lognorm':

            inflow = st.lognorm.ppf(q = p,s = params[0], loc = params[1], scale = params[2])          

        elif name == 'pearson3':

            inflow = st.pearson3.ppf(q = p,skew = params[0], loc = params[1], scale = params[2])        

        elif name == 'loggamma':
            
            inflow = st.pearson3.ppf(q = p,kew = params[0], loc = params[1], scale = params[2])          

        return inflow

    def policy_during_water_deficit(self, month, water_storage, p_value = 0.025):
        
        inflow_p = self.inflow_ppf(month, p_value)
        threshold = lm.dual_system.rfd_threshold[int(month - 1)]
        bier = lm.dual_system.bier[int(month - 1)]
        robustness = lm.dual_system.resilience
        cwdi_limit = 0.85
        expected_e = self.env.es_expect_array[int(month-1)]
        
        coeff_cwdi_limit = np.log(99*cwdi_limit/(1-cwdi_limit))/robustness
        const_rfd_limit = expected_e * bier
        
        c_cwdi = coeff_cwdi_limit * (inflow_p + 2 * water_storage)
        c_rfd = threshold / cwdi_limit + const_rfd_limit
        
        sensitivity = [0,0] # [cwdi, rfd]
        
        if c_cwdi <= c_rfd:
            sensitivity = np.array([1,0])
            criteria = np.array([c_cwdi, c_rfd])
        
        else:
            sensitivity = [0,1]
            criteria = np.array([c_rfd, c_cwdi]) 
        
        return min(c_cwdi, c_rfd), sensitivity, criteria


    def training_mdp(self, reward_type = 0):
        
        annual_q_table = self.initial_annual_q_table(model_set)
        step = 0
        annual_count = self.count_annual_table(model_set)
        loss_array = np.zeros(shape = self.count_step * self.epochs + 1)
        
        def get_month_indice(month):
            
            if month % 12 == 0:
                return 11
            
            else:
                return month % 12 - 1
        
        for i in range(self.epochs):
            
            storage, month, inflow, es = self.env.reset_random()
            eplison = self.eplison * self.decay ** (i)
            
            while step < self.count_step* self.epochs:
                
                storage_elimit = self.storage_elimit[int(month - 1)]
                storage_limit = self.storage_limit[int(month - 1)]                
                step += 1
                q_table = annual_q_table[get_month_indice(month)]
                q_table_next = annual_q_table[get_month_indice(month + 1)]
                s_i = self.storage_indice(storage)
                count_table = annual_count[get_month_indice(month)]
                consumption_boundary = max(storage + inflow, 0)
                c_i_boundary = np.argmin(abs(consumption_boundary - self.env.consumptions))
                
                if storage < storage_elimit:
                    consumption_elimit = self.policy_during_water_deficit(month, storage)[0]
                    c_i = np.argmin(abs(consumption_elimit - self.env.consumptions))
                    c_i = min(c_i, c_i_boundary)
                    consumption = self.env.consumptions[c_i]
                
                elif storage < storage_limit and storage > storage_elimit:
                    consumption_limit = self.policy_during_water_deficit(month, storage, p_value = 0.15)[0]
                    c_i = np.argmin(abs(consumption_limit - self.env.consumptions))
                    c_i = min(c_i, c_i_boundary)
                    consumption = self.env.consumptions[c_i]                  
                    
                else:
                    if np.random.uniform() < eplison:
                        consumption = self.env.randant_choice(inflow, storage, month)
                        c_i = self.consumption_indice(consumption)
                
                    else:
                        c_i = min(np.argmax(q_table[s_i]), c_i_boundary)
                        consumption = self.env.consumptions[c_i]
                
                next_storage, reward, consumption, storage, next_month, inflow, es = self.env.step_update_mdp(consumption, 
                                                                                           storage, 
                                                                                           month,
                                                                                           reward_type = reward_type)
                n_s_i = self.storage_indice(next_storage)

                temp_record = q_table[s_i][c_i] * (1-self.learning_rate)
                temp_update = (reward +  self.discount_factor*np.max(q_table_next[n_s_i])) * self.learning_rate
                
                q_table[s_i][c_i] = temp_record + temp_update
                annual_q_table[get_month_indice(month)] = q_table
                count_table[s_i][c_i] = count_table[s_i][c_i] + 1
                
                annual_count[get_month_indice(month)] = count_table
                storage = next_storage
                month = next_month
                loss_array[step] = np.mean(q_table)
                
        return annual_q_table, annual_count, loss_array
    
    def policy_mdp_based(self, size = 100, reward_type = 0):
    
        record_mean = [[],[],[],[],[],[],[],[],[],[],[],[]]
        record_max = [[],[],[],[],[],[],[],[],[],[],[],[]]
        record = [[],[],[],[],[],[],[],[],[],[],[],[]]
        record_qmean = [[],[],[],[],[],[],[],[],[],[],[],[]]
        record_median = [[],[],[],[],[],[],[],[],[],[],[],[]]
        record_std = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(size):
        
            temp = self.training_mdp(reward_type = reward_type)[0]
        
            for j in range(len(temp)):
    
                temp_month = temp[j]
                record_max[j].append(np.argmax(temp_month, axis = 1))
                record[j].append(temp_month)
    
        for i in range(12):
        
            record_mean[i] = np.round(np.mean(np.array(record_max[i]).T, axis = 1))
            record_median[i] = np.round(np.median(np.array(record_max[i]).T, axis = 1))
            record[i] = np.array(record[i])
            record_std[i] = np.std(np.array(record_max[i]).T, axis = 1)
            record_qmean[i] = np.argmax(np.mean(record[i], axis = 0), axis = 1)
    
        return record_mean, record_median, record_qmean, record_std    


### from 12000 to 6000#######
df_r = remake_df(df_r)
opt_s = Q_learning(50, 12000, 0.9, 0.2, 0.1, risk_map, model_set, 1000, 1, 
                 discount_factors[0])
opt_m = Q_learning(50, 12000, 0.9, 0.2, 0.1, risk_map, model_set, 1000, 1, 
                 discount_factors[1])
opt_l = Q_learning(50, 12000, 0.9, 0.2, 0.1, risk_map, model_set, 1000, 1, 
                 discount_factors[2])

### training policy 
# =============================================================================
# traing_l_ri = opt_l.policy_mdp_based(reward_type= 0)
# np.save("policy_l_ri", traing_l_ri[0])
# np.save("policy_l_ri_std", traing_l_ri[3])
# 
# training_l_riii = opt_l.policy_mdp_based(reward_type = 2)
# np.save("policy_l_riii", training_l_riii[0])
# np.save("policy_l_riii_std", training_l_riii[3])
# 
# training_m_ri = opt_m.policy_mdp_based(reward_type = 0)
# np.save("policy_m_ri", training_m_ri[0])
# np.save("policy_m_ri_std", training_m_ri[3])
# 
# training_m_riii = opt_m.policy_mdp_based(reward_type = 2)
# np.save("policy_m_riii", training_m_riii[0])
# np.save("policy_m_riii_std", training_m_riii[3])
# 
# training_s_ri = opt_s.policy_mdp_based(reward_type = 0)
# np.save("policy_s_ri", training_s_ri[0])
# np.save("policy_s_ri_std", training_s_ri[3])
# 
# training_s_riii = opt_s.policy_mdp_based(reward_type = 2)
# np.save("policy_s_riii", training_s_riii[0])
# np.save("policy_s_riii_std", training_s_riii[3])
# =============================================================================

policy_l_ri, policy_l_ri_count, policy_l_ri_loss = opt_l.training_mdp(0)
policy_l_riii, policy_l_riii_count, policy_l_riii_loss = opt_l.training_mdp(1)

policy_m_ri, policy_m_ri_count, policy_m_ri_loss = opt_m.training_mdp(0)
policy_m_riii, policy_m_riii_count, policy_m_riii_loss = opt_m.training_mdp(1)

policy_s_ri, policy_s_ri_count, policy_s_ri_loss = opt_s.training_mdp(0)
policy_s_riii, policy_s_riii_count, policy_s_riii_loss = opt_s.training_mdp(1)


#######save policy#############

def default_water_resoruce_policy():
    
    elimit = lm.dual_system.dual_system_update.storage_elow_bond
    limit = lm.dual_system.dual_system_update.storage_low_bond
    consumptions = model_set[0]
    storages = model_set[1]
    
    policy = [[],[],[],[],[],[],[],[],[],[],[],[]]
    
    for j in range(12):
        
        for i in range(len(storages)):
            storage = storages[i]
        
            if storage <= elimit[j]:
                consumption = 0.5 * demand_list[j]
    
            elif storage > elimit[j] and storage < limit[j]:
                consumption = 0.75 * demand_list[j]
    
            else:
                consumption = demand_list[j]
        
            c_i = int(np.argmin(abs(consumptions - consumption)))
            policy[j].append(c_i)
    
    for i in range(12):
        
        policy[i] = np.array(policy[i])
    
    return np.array(policy)
    
class water_management_evaluation:
    
    def __init__(self, annual_policy, demand_list):
        
        self.annual_policy = annual_policy
        self.demand_list = demand_list
        self.capacity = lm.dual_system.storage_capacity
        self.correct_array = lm.dual_system.storage_correct_array
        env = env_setting(risk_map, model_set, 2000,
                         1, demand_list, )
        self.env = env
    
    def dual_state_update(self, month, pre_storage, consumption, inflow, es):
        
        threshold = lm.dual_system.rfd_threshold[month - 1]

        demand = self.demand_list[month - 1]
        change_correct = self.env.storage_change_correction(month, pre_storage)
        consumption_boundary = min(max(inflow + pre_storage,0), consumption)        
        next_storage = self.env.storage_update(month, pre_storage, consumption_boundary, 
                                               inflow, change_correct)
        storage_state = (pre_storage + next_storage)/2
        
        cta = lm.dual_system.dual_system_update.get_cta_mdp(consumption_boundary, inflow, storage_state)
        wdi = lm.dual_system.dual_system_update.get_WDI_mdp(cta)
        rfd = lm.dual_system.dual_system_update.get_RFD_mdp(consumption_boundary, wdi, es, month)        
        
        dual_system_state = [0,0] # [over rfd, over cwdi]
        status = [month, storage_state, cta, wdi, rfd, (consumption_boundary - demand)/demand]
        
        if rfd < threshold and wdi < 0.85:
            dual_system_state = [0,0]
        elif rfd > threshold and wdi < 0.85:
            dual_system_state = [1,0]
        elif rfd < threshold and wdi > 0.85:
            dual_system_state = [0,1]
        else:
            dual_system_state = [1,1]

        def get_month(month):
            
            if month % 12 == 0:
                return 12
            else:
                return month % 12
        
        update_state = [get_month(month + 1), next_storage]
        
        return update_state, dual_system_state, status, inflow, es, change_correct
    
    def policy_reward_update(self, month, pre_storage, consumption,inflow, es):
        
        change_correct = self.env.storage_change_correction(month, pre_storage)
        consumption_boundary = min(max(inflow + pre_storage,0), consumption)
        next_storage = self.env.storage_update(month ,pre_storage, consumption_boundary,
                                               inflow, change_correct)
        demand = self.demand_list[int(month - 1)]
        demand = model_set[0][np.argmin(abs(demand - model_set[0]))]
        
        p_s_i = np.argmin(abs(pre_storage - model_set[1]))
        c_i = np.argmin(abs(consumption_boundary - model_set[0]))
        pf = risk_map[int(month - 1)][int(p_s_i)][int(c_i)]
        
        return (1-pf)*consumption_boundary, (consumption_boundary)/demand, next_storage
        
    def decision_under_policy(self, month, pre_storage, inflow):
        
        storage_indice = np.argmin(abs(model_set[1] - pre_storage))
        policy = self.annual_policy[int(month) - 1]
        consumption_indice = policy[storage_indice]
        consumption = model_set[0][int(consumption_indice)]
        consumption_boundary = min(max(inflow + pre_storage, 0), consumption)

        return consumption_boundary

    def decision_under_given_policy(self, month, pre_storage, inflow, policy,
                                    change_correct):
        storage_indice = np.argmin(abs(model_set[1] - pre_storage))
        policy = policy[int(month) - 1]
        consumption_indice = policy[storage_indice]
        consumption = model_set[0][int(consumption_indice)]
        consumption_boundary = min(max(inflow + pre_storage + change_correct, 0),
                                   consumption)
        return consumption_boundary

    def twenty_year_simulation(self, month, initial_storage):
        
        def get_month(month):
            
            if month % 12 == 0:
                return 12
            else:
                return month % 12
        
        update_state_record = []
        dual_system_state_record = []
        status_record = []
        
        for i in range(20*12):
            
            if i == 0:
                month = get_month(month)
                inflow = self.env.random_inflow(month)
                es = self.env.random_es(month)
                storage = initial_storage
                consum = self.decision_under_policy(month, storage, inflow)
                update_state, dual_system_state, status, inflow, es, change_correct = self.dual_state_update(month,
                                                                                        storage,
                                                                                        consum, 
                                                                                        inflow, 
                                                                                        es)
                update_state_record.append(update_state)
                dual_system_state_record.append(dual_system_state)
                status_record.append(status)
                next_month, next_storage = update_state
            
            else:
                month = next_month
                inflow = self.env.random_inflow(month)
                es = self.env.random_es(month)
                storage = next_storage
                consum = self.decision_under_policy(month, storage, inflow)
                update_state, dual_system_state, status, inflow, es, change_correct = self.dual_state_update(month,
                                                                                        storage,
                                                                                        consum,
                                                                                        inflow,
                                                                                        es)
                update_state_record.append(update_state)
                dual_system_state_record.append(dual_system_state)
                status_record.append(status)
                next_month, next_storage = update_state
        
        return np.array(update_state_record), np.array(dual_system_state_record), np.array(status_record)              

    def sampling_every_twenty_year(self, 
                                   month, 
                                   initial_storage, 
                                   simulation_time = 1000):
        
        update_state_records = []
        dual_system_state_records = [] 
        status_records = []
        
        for i in range(simulation_time):
            
            update_state_record, dual_system_state_record, status_record = self.twenty_year_simulation(month, 
                                                                                             initial_storage)
            update_state_records.append(update_state_record)
            dual_system_state_records.append(dual_system_state_record)
            status_records.append(status_record)
        
        return update_state_records, dual_system_state_records, status_records
            
    def sequential_simulation(self, month, initial_storage, sample_len):
        def get_month(month):
            
            if month % 12 == 0:
                return 12
            else:
                return month % 12
        
        reward_i_record = []
        reward_iii_record = []
        storage_record = []
        for i in range(sample_len):
            if i == 0:
                month = get_month(month)
                inflow = self.env.random_inflow(month)
                es = self.env.random_es(month)
                storage = initial_storage
                consum = self.decision_under_policy(month, storage, inflow)
                reward_i, reward_iii, next_storage = self.policy_reward_update(month,
                                                                               storage,
                                                                               consum,
                                                                               inflow,
                                                                               es)
                reward_i_record.append(reward_i)
                reward_iii_record.append(reward_iii)
                storage_record.append(next_storage)
                month = get_month(month + 1)
            else:
                inflow = self.env.random_inflow(month)
                es = self.env.random_es(month)
                storage = next_storage
                consum = self.decision_under_policy(month, storage, inflow)
                reward_i, reward_iii, next_storage = self.policy_reward_update(month,
                                                                               storage,
                                                                               consum,
                                                                               inflow,
                                                                               es)
                reward_i_record.append(reward_i)
                reward_iii_record.append(reward_iii)
                storage_record.append(next_storage)
                month = get_month(month + 1)
        return np.array(storage_record), np.array(reward_i_record), np.array(reward_iii_record) 

    def resample_sequential_simulation(self, month, initial_storage, sample_len,
                                       simulation_time = 1000):
        ws_record = []
        ri_record = []
        riii_record = []
        for i in range(simulation_time):
            ws, ri, riii = self.sequential_simulation(month,
                                                      initial_storage, 
                                                      sample_len)
            ws_record.append(ws)
            ri_record.append(ri)
            riii_record.append(riii)
        return np.array(ws_record), np.array(ri_record), np.array(riii_record)
    
    def wave_plus_modeling_simulation(self, initial_month, pre_initial_storage,
                                      policy, sample_year = 40):
        def get_month(month):
            if month % 12 == 0:
                return 12
            else:
                return month % 12
            
        cta_record = []
        consumption_record = []
        month_record = []
        es_record = []
        for i in range(12 * sample_year):
            if i == 0:
                month = get_month(initial_month)
                inflow = self.env.random_inflow(month)
                es = self.env.random_es(month)
                storage = pre_initial_storage
                correct = self.env.storage_change_correction(month, storage)
                consumption = self.decision_under_given_policy(month, storage,
                                                               inflow, policy, correct)
                next_storage = self.env.storage_update(month, storage,
                                                       consumption, 
                                                       inflow, correct)
                storage_state = (next_storage + storage) / 2
                cta_record.append(consumption / (inflow + 2 * storage_state))
                consumption_record.append(consumption)
                month_record.append(month)
                es_record.append(es)
            else:
                month = get_month(month + 1)
                inflow = self.env.random_inflow(month)
                es = self.env.random_es(month)
                storage = pre_initial_storage
                correct = self.env.storage_change_correction(month, storage)
                consumption = self.decision_under_given_policy(month, storage,
                                                               inflow, policy, correct)
                next_storage = self.env.storage_update(month, storage,
                                                       consumption, 
                                                       inflow, correct)
                storage_state = (next_storage + storage) / 2
                cta_record.append(consumption / (inflow + 2 * storage_state))
                consumption_record.append(consumption)
                month_record.append(month)
                es_record.append(es)
        consumption_record = np.array(consumption_record)
        es_record = np.array(es_record)
        cta_record = np.array(cta_record)
        month_record = np.array(month_record)
        
        rfd_record = []
        cwdi_record = []
        wave_model = lm.catcharea_state(3000, np.log(99) / np.median(cta_record), lm.dual_system.bier)
        for i in range(len(consumption_record)):
            cwdi = wave_model.get_WDI(cta_record[i])
            rfd = wave_model.get_RFD(consumption_record[i], cwdi, 
                                     es_record[i], 
                                     month_record[i])
            cwdi_record.append(cwdi)
            rfd_record.append(rfd)
        return month_record, np.log(99) / np.median(cta_record), np.array(cwdi_record), np.array(rfd_record)
    
    def resampling_wave_plus(self, initial_month, pre_initial_storage, policy, 
                             sampling_time = 1000):
        
        rfd_simulations = [[],[],[],[],[],[],[],[],[],[],[],[]]
        cwdi_simulations = [[],[],[],[],[],[],[],[],[],[],[],[]]
        alpha_simulations = []
        for i in range(sampling_time):
            months, alpha, cwdis, rfds = self.wave_plus_modeling_simulation(initial_month, 
                                                                            pre_initial_storage, 
                                                                            policy)
            for j in range(len(months)):
                for k in range(12):
                    if k + 1 == months[j]:
                        cwdi_simulations[k].append(cwdis[j])
                        rfd_simulations[k].append(rfds[j])
            alpha_simulations.append(alpha)
        return  np.array(cwdi_simulations), np.array(rfd_simulations), np.array(alpha_simulations)        
            
    def lamda_vis_calculation_under_policy(self, 
                                           initial_month, 
                                           initial_storage,
                                           inflow_series,
                                           policy,
                                           correction_series = 0,
                                           correction_random = True):
        def get_month(month):
            if month % 12 == 0:
                return 12
            else:
                return month % 12
        policy_index_array = []
        storage_index_array = []
        failure_probability = []
        policy_array = []
        storage_array = []
        m_p_ws = np.gradient(risk_map, axis = 1).T
        m_p_c = np.gradient(risk_map, axis = 2)
        m_p_t = np.gradient(risk_map, axis = 0)
        lamda_array = []
        vis_array = []
        dpf_array = []
        for i in range(len(inflow_series) - 1):
            if i == 0:
                month = get_month(initial_month + i)
                inflow = inflow_series[i]
                storage = initial_storage
                if correction_random == True:
                    change_correct =  self.env.storage_change_correction(month, storage)
                else:
                    change_correct = correction_series[i]
                consum = self.decision_under_given_policy(month, 
                                                          storage, 
                                                          inflow,
                                                          policy,
                                                          change_correct)
                af_storage = self.env.storage_update(month, 
                                                     storage, 
                                                     consum, 
                                                     inflow, 
                                                     change_correct)
                c_i = np.argmin(abs(consum - model_set[0]))
                ##modified storage
                s_i = np.argmin(abs(af_storage - model_set[1]))
                policy_index_array.append(c_i)
                storage_index_array.append(s_i)
                policy_array.append(consum)
                storage_array.append(storage)
            else:
                month = get_month(initial_month + i)
                pre_month = get_month(initial_month + i - 1)
                inflow = inflow_series[i]
                if correction_random == True:
                    change_correct =  self.env.storage_change_correction(month, storage)
                else:
                    change_correct = correction_series[i]
                consum = self.decision_under_given_policy(month, 
                                                          storage, 
                                                          inflow,
                                                          policy,
                                                          change_correct)
                storage = af_storage
                af_storage = self.env.storage_update(month, 
                                                     storage, 
                                                     consum, 
                                                     inflow, 
                                                     change_correct)
                c_i = np.argmin(abs(consum - model_set[0]))
                s_i = np.argmin(abs(af_storage - model_set[1]))
                policy_index_array.append(c_i)
                storage_index_array.append(s_i)
                policy_array.append(consum)
                storage_array.append(storage)
    
                b_c_i = policy_index_array[i-1]
                b_s_i = storage_index_array[i-1]
                p_c = m_p_c[int(month)-1]
                p_ws = m_p_ws[int(month)-1]
                c_s_i_c = np.argmin(abs(consum - model_set[1]))
                c_s_i = max(min(b_s_i- c_s_i_c,19),0)
                pf = risk_map[int(month - 1)][b_s_i][c_i]
                
                if c_s_i - b_s_i == 0:
                    c_s_unit = 0
                else:
                    c_s_unit = 1 / (c_s_i - b_s_i)
                if c_i - b_c_i == 0:
                    c_c_unit = 0
                else:
                    c_c_unit = 1 / (c_i - b_c_i)
                
                lamda_policy = np.sign(b_s_i - c_s_i) * np.sum(p_ws[b_c_i][min(c_s_i, b_s_i): max(c_s_i, b_s_i)+1]) * c_s_unit           
                lamda_policy_change = np.sign(c_i - b_c_i) * np.sum(p_c[b_s_i][min(c_i,b_c_i): max(c_i,b_c_i)+1]) * c_c_unit
                lamda = lamda_policy_change + lamda_policy
                lamda_array.append(lamda)
                
                p_t = m_p_t[int(pre_month - 1)]                  
                vis_season = p_t[b_s_i][b_c_i]           
                vis_eff_in = np.sign(s_i - b_s_i) * np.sum(p_ws[c_i][min(s_i, b_s_i): max(s_i, b_s_i)+1]) - lamda_policy
                vis = vis_season + vis_eff_in
                vis_array.append(vis)
                failure_probability.append(pf)
                dpf_array.append(vis + lamda)
                
        storage_array = np.array(storage_array)
        failure_probability = np.array(failure_probability)
        lamda_array = np.array(lamda_array)
        dpf = np.array(dpf_array)
        vis_array = np.array(vis_array)
        return lamda_array, vis_array, failure_probability, dpf
        
    def lamda_vis_calculation(self, initial_month,
                              initial_storage,
                              inflow_series, 
                              consum_series,
                              correction_series = 0,
                              correct_random = True):
        
        def get_month(month):
            if month % 12 == 0:
                return 12
            else:
                return month%12
        
        policy_index_array = []
        storage_index_array = []
        failure_probability = []
        policy_array = []
        storage_array = []
        m_p_ws = np.gradient(risk_map, axis = 1).T
        m_p_c = np.gradient(risk_map, axis = 2)
        m_p_t = np.gradient(risk_map, axis = 0)

        lamda_array = []
        vis_array = []
        for i in range(len(inflow_series) - 1):
            if i == 0:
                month = get_month(initial_month + i)
                inflow = inflow_series[i]
                storage = initial_storage
                consum = consum_series[i]
                if correct_random == True:
                    change_correct =  self.env.storage_change_correction(month, storage)
                else:
                    change_correct = correction_series[i]
                af_storage = self.env.storage_update(month, 
                                                     storage, 
                                                     consum, 
                                                     inflow, 
                                                     change_correct)
                c_i = np.argmin(abs(consum - model_set[0]))
                s_i = np.argmin(abs(af_storage - model_set[1]))
                policy_index_array.append(c_i)
                storage_index_array.append(s_i)
                policy_array.append(consum)
                storage_array.append(storage)
            else:
                month = get_month(initial_month + i)
                pre_month = get_month(initial_month + i - 1)
                inflow = inflow_series[i]
                storage = af_storage
                consum = consum_series[i]
                if correct_random == True:
                    change_correct =  self.env.storage_change_correction(month, storage)
                else:
                    change_correct = correction_series[i]
                af_storage = self.env.storage_update(month, 
                                                     storage, 
                                                     consum, 
                                                     inflow, 
                                                     change_correct)
                c_i = np.argmin(abs(consum - model_set[0]))
                s_i = np.argmin(abs(af_storage - model_set[1]))
                policy_index_array.append(c_i)
                storage_index_array.append(s_i)

                policy_array.append(consum)
                storage_array.append(storage)
                
                b_c_i = policy_index_array[i-1]
                b_s_i = storage_index_array[i-1]
                pf = risk_map[int(month - 1)][b_s_i][c_i]
                p_c = m_p_c[int(month)-1]
                p_ws = m_p_ws[int(month)-1]
                c_s_i_c = np.argmin(abs(consum - model_set[1]))
                c_s_i = max(min(b_s_i - c_s_i_c , 19),0)

                if c_s_i - b_s_i == 0:
                    c_s_unit = 0
                else:
                    c_s_unit = 1 / (c_s_i - b_s_i)
                if c_i - b_c_i == 0:
                    c_c_unit = 0
                else:
                    c_c_unit = 1 / (c_i - b_c_i)

                lamda_policy = np.sign(b_s_i - c_s_i) * np.sum(p_ws[b_c_i][min(c_s_i, b_s_i): max(c_s_i, b_s_i)+1]) * c_s_unit        
                lamda_policy_change = np.sign(c_i - b_c_i) * np.sum(p_c[b_s_i][min(c_i,b_c_i): max(c_i,b_c_i)+1]) * c_c_unit
                lamda = lamda_policy_change + lamda_policy
                lamda_array.append(lamda)

                p_t = m_p_t[int(pre_month - 1)]
                vis_season = p_t[b_s_i][b_c_i]
                vis_eff_in = np.sign(s_i - b_s_i) * np.sum(p_ws[b_c_i][min(s_i, b_s_i): max(s_i, b_s_i)+1]) - lamda_policy
                vis = vis_season + vis_eff_in
                vis_array.append(vis)
                failure_probability.append(pf)
                
        storage_array = np.array(storage_array)
        failure_probability = np.array(failure_probability)
        lamda_array = np.array(lamda_array)
        dpf = np.gradient(failure_probability)
        vis_array = np.array(vis_array)
        
        return lamda_array, vis_array, failure_probability, dpf
    
    def lamda_and_vis_simulation_under_policy(self, 
                                              initial_month, 
                                              initial_storage,
                                              policy,
                                              sample_len = 2,
                                              simulation_time = 1000):
        
        def get_month(month):
            if month % 12 == 0:
                return 12
            else:
                return month % 12
        
        inflow_simulation = []
        date_record = []
        for i in range(sample_len):
            month = get_month(initial_month + i)
            date_record.append(month)
            
        for i in range(simulation_time):
            temp_inflow = []
            for j in range(sample_len):
                month = get_month(initial_month + j)
                temp_inflow.append(self.env.random_inflow(month))
            inflow_simulation.append(temp_inflow)
        
        inflow_simulation = np.array(inflow_simulation)
        dpf_simulation = []
        lamda_simulation = []
        vis_simulation = []
        pf_simulation = []
        for i in range(len(inflow_simulation)):
            inflow_series = inflow_simulation[i]
            lamda_array, vis_array, pf, dpf = self.lamda_vis_calculation_under_policy(initial_month, 
                                                                                      initial_storage, 
                                                                                      inflow_series,
                                                                                      policy)
            dpf_simulation.append(dpf)
            lamda_simulation.append(lamda_array)
            vis_simulation.append(vis_array)
            pf_simulation.append(pf)
        return np.array(lamda_simulation), np.array(vis_simulation), np.array(pf_simulation), date_record
            
    def lamda_and_vis_simulation(self, 
                                 initial_month,
                                 initial_storage, 
                                 consum_series,
                                 simulation_time = 1000):
        
        def get_month(month):
            if month % 12 == 0:
                return 12
            else:
                return month % 12
        
        inflow_simulation = []
        date_record = []
        for i in range(len(consum_series)):
            month = get_month(initial_month + i)
            date_record.append(month)
        
        for i in range(simulation_time):
            temp_inflow = []
            for j in range(len(consum_series)):
                month = get_month(initial_month + j)
                temp_inflow.append(self.env.random_inflow(month))
            inflow_simulation.append(temp_inflow)
        
        inflow_simulation = np.array(inflow_simulation)
        dpf_simulation = []
        lamda_simulation = []
        vis_simulation = []
        pf_simulation = []
        for i in range(len(inflow_simulation)):
            inflow_series = inflow_simulation[i]
            lamda_array, vis_array, pf, dpf = self.lamda_vis_calculation(initial_month, 
                                                                         initial_storage, 
                                                                         inflow_series,
                                                                         consum_series)
            dpf_simulation.append(dpf)
            lamda_simulation.append(lamda_array)
            vis_simulation.append(vis_array)
            pf_simulation.append(pf)
        return np.array(lamda_simulation), np.array(vis_simulation), np.array(pf_simulation), date_record                   

    def evolution_historical_state(self,
                                   date_series,
                                   inflow_series, 
                                   storage_series,
                                   correction_series,
                                   consumption_series):

        m_p_t = np.gradient(risk_map, axis = 0)
        m_p_ws = np.gradient(risk_map, axis = 1)
        m_p_c = np.gradient(risk_map, axis = 2)
        
        def get_month(month):
            if month % 12 == 0:
                return 12
            else:
                return int(month % 12)

        lamda_record = []
        vis_record = []
        pf_record = []
        c_i_record = []
        s_i_record = []
        dpf_record = []
    
        for i in range(len(date_series) - 1):
            if i == 0:
                storage = storage_series[i + 1]
                consum = consumption_series[i]
                month = date_series[i].month
                c_i = np.argmin(abs(consum - model_set[0]))
                s_i = np.argmin(abs(storage - model_set[1]))
                c_i_record.append(c_i)
                s_i_record.append(s_i)
            else:
                storage = storage_series[i + 1]
                consum = consumption_series[i]
                month = date_series[i].month
                pre_month = date_series[i - 1].month
                c_i = np.argmin(abs(consum - model_set[0]))     
                s_i = np.argmin(abs(storage - model_set[1]))
                c_i_record.append(c_i)
                s_i_record.append(s_i)
            
                b_s_i = s_i_record[i-1]
                b_c_i = c_i_record[i-1]              
                c_s_i_c = np.argmin(abs(consum - model_set[1]))
                c_s_i = max(min(b_s_i - c_s_i_c ,19),0)

                if c_s_i - b_s_i == 0:
                    c_s_unit = 0
                else:
                    c_s_unit = 1 / (c_s_i - b_s_i)
                if c_i - b_c_i == 0:
                    c_c_unit = 0
                else:
                    c_c_unit = 1 / (c_i - b_c_i)
                           
                p_t = m_p_t[pre_month - 1][b_s_i][b_c_i]
                p_ws = m_p_ws[month - 1].T
                p_c = m_p_c[month - 1]
                vis_season = p_t

                lamda_policy = np.sign(b_s_i - c_s_i) * np.sum(p_ws[b_c_i][min(b_s_i, c_s_i): max(b_s_i, c_s_i)+1]) * c_s_unit
                lamda_policy_change = np.sign(c_i - b_c_i) * np.sum(p_c[b_s_i][min(b_c_i,c_i): max(b_c_i,c_i)+1]) * c_c_unit
                vis_eff_in = np.sign(s_i - b_s_i) * np.sum(p_ws[b_c_i][min(s_i, b_s_i): max(s_i, b_s_i)+1]) - lamda_policy

                vis = vis_season + vis_eff_in
                lamda = lamda_policy_change + lamda_policy
                lamda_record.append(lamda)
                vis_record.append(vis)
                pf_record.append(risk_map[int(month)-1][int(b_s_i)][int(c_i)])
                dpf_record.append(lamda + vis)
    
        pf_record = np.array(pf_record)
        dpf = np.array(dpf_record)
        lamda_record = np.array(lamda_record)
        vis_record = np.array(vis_record)       
        return lamda_record, vis_record, pf_record, dpf

    def historical_datum_of_lamda_vis(self, 
                                      date_series,
                                      inflow_series,
                                      correct_series,
                                      storage_series,
                                      datum_policy
                                      ):
        expected_lamda_datum = []
        lamda_datum = []
        vis_datum = []
        for i in range(len(date_series)-1):
            initial_month = int(date_series[i].month)
            initial_storage = storage_series[i]
            temp_inflow = [inflow_series[i], inflow_series[i + 1]]
            temp_correct = [correct_series[i], correct_series[i + 1]]
            body = self.lamda_vis_calculation_under_policy(initial_month = initial_month,
                                                           initial_storage = initial_storage, 
                                                           inflow_series = temp_inflow, 
                                                           policy = datum_policy,
                                                           correction_series = temp_correct,
                                                           correction_random = False)
            lamda_datum.append(body[0][0])
            body_vis = self.lamda_and_vis_simulation_under_policy(initial_month = initial_month,
                                                                  initial_storage = initial_storage,
                                                                  policy = datum_policy,
                                                                  simulation_time = 100)
            lamda = body_vis[0]
            vis = body_vis[1]
            vis_datum.append(np.mean(vis))
            expected_lamda_datum.append(np.mean(lamda))
        return np.array(lamda_datum), np.array(vis_datum), np.array(expected_lamda_datum)


policy_l_ri = np.argmax(policy_l_ri, axis = 2)
policy_l_riii = np.argmax(policy_l_riii, axis = 2)
policy_s_ri = np.argmax(policy_s_ri, axis = 2)
policy_s_riii = np.argmax(policy_s_riii, axis = 2)
policy_m_ri = np.argmax(policy_m_ri, axis = 2)
policy_m_riii = np.argmax(policy_m_riii, axis = 2)
policy_default = default_water_resoruce_policy()
# =============================================================================
# policy_l_ri = get_policy("policy_l_ri")
# policy_l_riii = get_policy("policy_l_riii")
# policy_default = get_policy("policy_default")
# policy_m_ri = get_policy("policy_m_ri")
# policy_m_riii = get_policy("policy_m_riii")
# policy_s_ri = get_policy("policy_s_ri")
# policy_s_riii = get_policy("policy_s_riii")
# =============================================================================






      

    

            













    



    
        



          


    
    



