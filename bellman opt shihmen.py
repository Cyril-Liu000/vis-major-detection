import scipy.stats as st
import numpy as np
import statsmodels.tsa.api as tsa
import statsmodels.tsa as stastsa
import math as mt
import pandas as pd
import matplotlib.pyplot as plot
import datetime
import scipy.optimize as opt
import scipy.fft as fft
import math
import matplotlib.cm as cm

import learn_ar_model as lm
from matplotlib import gridspec

import json


def model_setting(consumption_limit, max_storage, min_storage, resolution):
    
    consumption_array = np.zeros(shape = resolution)
    storage_array = np.zeros(shape = resolution)
    c_step = consumption_limit / resolution
    s_step = (max_storage - min_storage) / resolution        
    
    for i in range(resolution):
            
        consumption_array[i] = c_step * (i + 1)
        storage_array[i] = min_storage + s_step * i

    return consumption_array, storage_array

# model_set = model_setting(consumption_limit, max_storage, min_storage, resolution)


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
    return (1.3*month_demand/ month_count).reshape([12])

# demand_list = get_demand(file_path_data_sheet)
# demand_list = get_demand(configs['files']['data_sheet'])

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
        inflow = self.random_inflow(month)
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
            
#            q_table = 0.1*st.uniform.rvs(size = [len(model_set[1]), len(model_set[0])])
            q_table = np.zeros(shape = [len(model_set[1]), len(model_set[0])])
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

    def policy_during_water_deficit(self, month, water_storage, p_value = 0.05):
        
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
                    consumption_limit = self.policy_during_water_deficit(month, storage, p_value = 0.25)[0]
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

        return annual_q_table, annual_count
    
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

        
    def decision_under_policy(self, month, storage, inflow):
        
        storage_indice = np.argmin(abs(model_set[1] - storage))
        policy = self.annual_policy[month - 1]
        consumption_indice = policy[storage_indice]
        consumption = model_set[0][int(consumption_indice)]
        consumption_boundary = min(max(inflow + storage, 0), consumption)

        return consumption_boundary
    
    def decision_under_given_policy(self, month, storage, inflow, policy,
                                    change_correct):
        storage_indice = np.argmin(abs(model_set[1] - storage))
        policy = policy[month - 1]
        consumption_indice = policy[storage_indice]
        consumption = model_set[0][int(consumption_indice)]
        consumption_boundary = min(max(inflow + storage + change_correct, 0),
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
                consum = self.decision_under_given_policy(month, storage, inflow)
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
                consum = self.decision_under_given_policy(month, storage, inflow)
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
                consum = self.decision_under_given_policy(month, storage, inflow)
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
                consum = self.decision_under_given_policy(month, storage, inflow)
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
        for i in range(len(inflow_series)):
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
                s_i = np.argmin(abs(storage - model_set[1]))
                policy_index_array.append(c_i)
                storage_index_array.append(s_i)
                pf = risk_map[int(month - 1)][s_i][c_i]
                failure_probability.append(pf)
                policy_array.append(consum)
                storage_array.append(storage)
            else:
                month = get_month(initial_month + i)
                inflow = inflow_series[i]
                storage = af_storage

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
                s_i = np.argmin(abs(storage - model_set[1]))
                policy_index_array.append(c_i)
                storage_index_array.append(s_i)
                pf = risk_map[int(month - 1)][s_i][c_i]
                policy_array.append(consum)
                storage_array.append(storage)
    
                b_c_i = policy_index_array[i-1]
                b_s_i = storage_index_array[i-1]
                p_c = m_p_c[int(month)-1]
                p_ws = m_p_ws[int(month)-1]
                c_s_i_c = np.argmin(abs(consum - model_set[1]))
                c_s_i = min(min(c_s_i_c + b_s_i,19),0)
                
                c_i_change = policy_index_array[i] - policy_index_array[i-1]
                lamda_policy = np.sum(p_ws[b_c_i][min(c_s_i, b_s_i): max(c_s_i, b_s_i)+1])           
                lamda_policy_change = np.sign(c_i_change)*np.sum(p_c[b_s_i][min(c_i,b_c_i): max(c_i,b_c_i)+1])
                lamda = lamda_policy_change - lamda_policy
                lamda_array.append(lamda)
                
                p_t = m_p_t[int(month - 1)]
                eff_in = inflow + change_correct
                eff_in_s_i_c = np.argmin(abs(eff_in - model_set[1]))
                eff_in_s_i = max(min(b_s_i + eff_in_s_i_c,19),0)
                vis_season = p_t[s_i][c_i]
                vis_eff_in = np.sum(p_ws[b_c_i][min(eff_in_s_i, b_s_i): max(eff_in_s_i, b_s_i)+1])
                vis = vis_season + vis_eff_in
                vis_array.append(vis)
                failure_probability.append(pf)
                
        storage_array = np.array(storage_array)
        failure_probability = np.array(failure_probability)
        lamda_array = np.array(lamda_array)
        dpf = np.gradient(failure_probability)
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
        for i in range(len(inflow_series)):
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
                s_i = np.argmin(abs(storage - model_set[1]))
                policy_index_array.append(c_i)
                storage_index_array.append(s_i)
                pf = risk_map[int(month - 1)][s_i][c_i]
                failure_probability.append(pf)
                policy_array.append(consum)
                storage_array.append(storage)
            else:
                month = get_month(initial_month + i)
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
                s_i = np.argmin(abs(storage - model_set[1]))
                policy_index_array.append(c_i)
                storage_index_array.append(s_i)
                pf = risk_map[int(month - 1)][s_i][c_i]
                policy_array.append(consum)
                storage_array.append(storage)
    
                b_c_i = policy_index_array[i-1]
                b_s_i = storage_index_array[i-1]
                p_c = m_p_c[int(month)-1]
                p_ws = m_p_ws[int(month)-1]
                c_s_i_c = np.argmin(abs(consum - model_set[1]))
                c_s_i = min(min(c_s_i_c + b_s_i,19),0)
                
                c_i_change = policy_index_array[i] - policy_index_array[i-1]
                lamda_policy = np.sum(p_ws[b_c_i][min(c_s_i, b_s_i): max(c_s_i, b_s_i)+1])         
                lamda_policy_change = np.sign(c_i_change)*np.sum(p_c[b_s_i][min(c_i,b_c_i): max(c_i,b_c_i)+1])
                lamda = lamda_policy_change - lamda_policy
                lamda_array.append(lamda)
                
                p_t = m_p_t[int(month - 1)]
                eff_in = inflow + change_correct
                eff_in_s_i_c = np.argmin(abs(eff_in - model_set[1]))
                eff_in_s_i = max(min(b_s_i + eff_in_s_i_c,19),0)
                vis_season = p_t[s_i][c_i]
                vis_eff_in = np.sum(p_ws[b_c_i][min(eff_in_s_i, b_s_i): max(eff_in_s_i, b_s_i)+1])
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
    
        for i in range(len(date_series)):
            if i == 0:
                storage = storage_series[i]
                inflow = inflow_series[i]
                correct = correction_series[i]
                consum = consumption_series[i]
                eff_in = inflow + correct
                month = date_series[i].month
                c_i = np.argmin(abs(consum - model_set[0]))
                s_i = np.argmin(abs(storage - model_set[1]))
                c_i_record.append(c_i)
                s_i_record.append(s_i)
                pf_record.append(risk_map[int(month)-1][int(s_i)][int(c_i)])        
            else:
                storage = storage_series[i]
                inflow = inflow_series[i]
                correct = correction_series[i]
                consum = consumption_series[i]
                eff_in = inflow + correct
                month = date_series[i].month
                c_i = np.argmin(abs(consum - model_set[0]))     
                s_i = np.argmin(abs(storage - model_set[1]))
                c_i_record.append(c_i)
                s_i_record.append(s_i)
            
                b_s_i = s_i_record[i-1]
                b_c_i = c_i_record[i-1]            
            
                c_s_i_c = np.argmin(abs(consum - model_set[1]))
                c_s_i = max(min(c_s_i_c + b_s_i,19),0)
                eff_s_i_c = np.argmin(abs(eff_in - model_set[1]))
                eff_s_i = max(min(eff_s_i_c + b_s_i,19),0)
            
                p_t = m_p_t[month - 1][s_i][c_i]
                p_ws = m_p_ws[month - 1].T
                p_c = m_p_c[month - 1]
            
                vis_season = p_t
                vis_eff_in = np.sum(p_ws[b_c_i][min(b_s_i, eff_s_i): max(b_s_i, eff_s_i)+1])
                lamda_policy = np.sum(p_ws[b_c_i][min(b_s_i, c_s_i): max(b_s_i, c_s_i)+1])
                lamda_policy_change = np.sign(c_i-b_c_i)*np.sum(p_c[b_s_i][min(b_c_i,c_i): max(b_c_i,c_i)+1])

                vis = vis_season + vis_eff_in
                lamda = lamda_policy_change - lamda_policy
                lamda_record.append(lamda)
                vis_record.append(vis)
                pf_record.append(risk_map[int(month)-1][int(b_s_i)][int(b_c_i)])
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
    
   
    

def drought_event_collection(result_data, demand_list):
    
    event_occurrence = []
    event_severity = []
    event_consum = []
    event_start = []
    
    threshold_list = lm.dual_system.rfd_threshold
    demand_list = demand_list
    storage_record, event_record, status_record = result_data
    
    for i in range(len(event_record)):
        
        event_temp = event_record[i].T[0] *  event_record[i].T[1]
        rfd_temp = status_record[i].T[4]
        month_temp = status_record[i].T[0]
        shoratage_i_temp = status_record[i].T[5]
        
        occur_temp = []
        severity_temp = []
        consum_temp = []
        start_temp = []
        
        for j in range(len(event_temp)):
            if j == 0:
                continue
            
            else:
                if event_temp[j] == 1 and event_temp[j-1] == 0:
                    event_len = 1
                    consum = demand_list[int(month_temp[j]-1)]*(1+shoratage_i_temp[j])
                    severity = max(threshold_list[int(month_temp[j]-1)],rfd_temp[j])
                    start_temp.append(month_temp[j])
                    
                elif event_temp[j] == 1 and event_temp[j-1] == 1:
                    event_len += 1
                    severity += max(threshold_list[int(month_temp[j]-1)],consum) 
                    consum += demand_list[int(month_temp[j]-1)]*(1+shoratage_i_temp[j])
                    
                elif event_temp[j] == 0 and event_temp[j-1] == 1:
                    occur_temp.append(event_len)
                    severity_temp.append(severity)
                    consum_temp.append(consum)
                    event_len = 0
                    severity = 0
                    consum = 0
                    
                else:
                    continue
        event_occurrence.append(occur_temp)
        event_severity.append(severity_temp)
        event_consum.append(consum_temp)
        event_start.append(start_temp)
        
        event_occurrence[i] = np.array(event_occurrence[i])
        event_severity[i] = np.array(event_severity[i])          
        event_consum[i] = np.array(event_consum[i])          
        event_start[i] = np.array(event_start[i])

    return event_occurrence, event_start, event_severity, event_consum          
    

# =============================================================================
# d_e_default = drought_event_collection(result_default, demand_list)
# d_e_m_ri = drought_event_collection(result_m_ri, demand_list)
# d_e_m_rii = drought_event_collection(result_m_rii, demand_list)
# d_e_m_riii = drought_event_collection(result_m_rii, demand_list)
# =============================================================================

def plt_drought_event(drought_result):
    
    event_occur, event_start, event_severity, event_consum = drought_result

    avg_severity = []
    occur = []
    month_record = []
    
    for i in range(len(event_start)):
        temp_record = event_start[i]
        temp_month_record = np.zeros(shape = (12,1))
        
        for j in range(len(temp_record)):
            for k in range(12):
                if k + 1 == temp_record[j]:
                    temp_month_record[k] +=1
                else:
                    continue
        month_record.append(temp_month_record)
    
    duration = []
    for i in range(len(event_occur)):
        for j in range(len(event_occur[i])):
            duration.append(event_occur[i][j])
    
    for i in range(len(event_occur)):
        avg_severity.append(np.mean(event_severity[i]))
        occur.append(len(event_occur[i]))

    occur = np.array(occur)

    duration = np.array(duration)
    avg_severity = np.array(avg_severity)
    month_record = np.array(month_record).T 
    occur[np.isnan(occur)] = 0
    avg_severity[np.isnan(avg_severity)] = 0
    
    return occur, duration, avg_severity, month_record 
    

# =============================================================================
# plt_de_d = plt_drought_event(d_e_default)
# plt_de_m_ri = plt_drought_event(d_e_m_ri)
# plt_de_m_riii = plt_drought_event(d_e_m_riii)
# =============================================================================

def plot_drought_violin(plt_de_d, plt_de_m_ri, plt_de_m_riii):
    
    occur_d, duration_d, avg_severity_d, month_record_d = plt_de_d
    occur_m_ri, duration_m_ri, avg_severity_m_ri, month_record_m_ri = plt_de_m_ri
    occur_m_riii, duration_m_riii, avg_severity_m_riii, month_record_m_riii = plt_de_m_riii
    
    avg_severity_d = avg_severity_d*86400/1000000
    avg_severity_m_ri = avg_severity_m_ri*86400/1000000
    avg_severity_m_riii = avg_severity_m_riii*86400/1000000
    
    labels = ["policy I", "policy II", "current"]
#    month_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    draw_month_record_d = []
    draw_month_record_m_ri = []
    draw_month_record_m_riii = []
    
    for i in range(len(month_record_d[0])):
        draw_month_record_d.append(month_record_d[0][i])
        draw_month_record_m_ri.append(month_record_m_ri[0][i])
        draw_month_record_m_riii.append(month_record_m_riii[0][i])      
 
    fig = plot.figure(dpi = 600,figsize = [5,3])
    spec = gridspec.GridSpec( nrows = 2, ncols = 1)
    
    fig.add_subplot(spec[0])    
    data_cap = [occur_m_ri, occur_m_riii, occur_d]
    labels = np.array(["policy I", "policy II", "current"])
    colors = ["blue", "orange", "green"]
    violin_parts = plot.violinplot(data_cap, vert = False, showmedians = True)
    i = 0
    for pc in violin_parts["bodies"]:
        pc.set_facecolor(colors[i])
        i += 1
    plot.yticks([1,2,3], labels = labels)
    plot.xlabel("occurence times")
    
    fig.add_subplot(spec[1])      
    data_si = [avg_severity_m_ri, avg_severity_m_riii, avg_severity_d]
    labels = np.array(["policy I", "policy II", "current"])
    colors = ["blue", "orange", "green"]
    violin_parts = plot.violinplot(data_si, vert = False, showmedians = True)
    i = 0
    for pc in violin_parts["bodies"]:
        pc.set_facecolor(colors[i])
        i += 1
    plot.yticks([1,2,3], labels = labels)
    plot.xlabel("average severity (M$m^3$)")
    plot.tight_layout()

    plot.figure()
    plot.violinplot(draw_month_record_d)
    plot.title("event start current policy")
    plot.ylim(0,20)
    plot.show()
    plot.close()    

    plot.figure()
    plot.violinplot(draw_month_record_m_ri)
    plot.title("policy I")
    plot.ylim(0,20)
    plot.show()
    plot.close()    

    plot.figure()
    plot.violinplot(draw_month_record_m_riii)
    plot.ylim(0,20)
    plot.title("policy II")  
    plot.show()
    plot.close()    
    

# =============================================================================
# policy_ri_evaluation = eva_p_l_ri.resample_sequential_simulation(1, 2000, 12,
#                                        simulation_time = 1000)
# policy_riii_evaluation = eva_p_l_riii.resample_sequential_simulation(1, 2000, 12,
#                                        simulation_time = 1000)
# policy_default_evaluation = eva_p_default.resample_sequential_simulation(1, 2000, 12,
#                                        simulation_time = 1000)
# =============================================================================

def draw_evaluation(policy_ri_evaluation,
                    policy_riii_evaluation,
                    policy_default_evaluation):

    ri_storage, ri_capacity, ri_si = policy_ri_evaluation
    riii_storage, riii_capacity, riii_si = policy_riii_evaluation
    dt_storage, dt_capacity, dt_si = policy_default_evaluation
    
    ri_storage = ri_storage.T *86400/1000000
    ri_capacity = ri_capacity.T *86400/1000000
    riii_storage = riii_storage.T *86400/1000000
    riii_capacity = riii_capacity.T *86400/1000000
    dt_storage = dt_storage.T *86400/1000000
    dt_capacity = dt_capacity.T *86400/1000000
    ri_si = ri_si.T * 100
    riii_si = riii_si.T * 100
    dt_si = dt_si.T * 100
    
    ri_s_mean = np.mean(ri_storage, axis = 1)
    riii_s_mean = np.mean(riii_storage, axis = 1)
    dt_s_mean = np.mean(dt_storage, axis = 1)
    
    ri_cap_sum = np.sum(ri_capacity, axis = 0)
    riii_cap_sum = np.sum(riii_capacity, axis = 0)
    dt_cap_sum= np.sum(dt_capacity, axis = 0)    
    
    ri_si_mean = np.median(ri_si, axis = 0)
    riii_si_mean = np.median(riii_si, axis = 0)
    dt_si_mean = np.median(dt_si, axis = 0)
    
    def generate_draw_bound(month_data, up, low):
        
        output = []
        
        for i in range(len(month_data)):
            temp_array = month_data[i]
            temp_array.sort()
            
            output.append([temp_array[int(len(temp_array) * up)],
                           temp_array[int(len(temp_array) * low)]])
        
        output = np.array(output)
        output = output.T
        return output

    ri_s_bond = generate_draw_bound(ri_storage, up = 0.84, low = 0.16)
    riii_s_bond = generate_draw_bound(riii_storage, up = 0.84, low = 0.16)
    dt_s_bond = generate_draw_bound(dt_storage, up = 0.84, low = 0.16)
    
    x = np.array(["Feb", "Mar", "Apr", "May", "Jun", "Jul", "Agu", "Sep", "Oct", "Nov", "Dec", "Jan"])
    fig = plot.figure(dpi = 600)
    axe = fig.add_axes([0.1,0.1,0.9,0.9])    
    axe.set_ylabel("volume (Mm$^3$)")
    axe.set_xlabel("month")
    x_multiplelocator = plot.MultipleLocator(1)
    axe.xaxis.set_major_locator(x_multiplelocator)
    
    axe.plot(x, ri_s_mean, label = "policy: reward I", color = "blue", alpha = 0.5)
    axe.fill_between(x,
                     ri_s_bond[0],
                     ri_s_bond[1],
                     alpha = .05, color = "blue")
    
    
    axe.plot(x, riii_s_mean, label = "policy: reward II", color = "orange")
    axe.fill_between(x,
                     riii_s_bond[0],
                     riii_s_bond[1],
                     alpha = .15, color = "orange")
    
    axe.plot(x, dt_s_mean, label = "current policy", color = "green")
    axe.fill_between(x,
                     dt_s_bond[0],
                     dt_s_bond[1],
                     alpha = .09, color = "green")

    axe.legend(loc = "upper left",bbox_to_anchor = (0.02,1))

#################################################
    fig = plot.figure(dpi = 600)
    spec = gridspec.GridSpec( nrows = 2, ncols = 1)
    fig.add_subplot(spec[0])    
    
    data_cap = [ri_cap_sum, riii_cap_sum, dt_cap_sum]
    labels = np.array(["policy I", "policy II", "current policy"])
    colors = ["blue", "orange", "green"]
    violin_parts = plot.violinplot(data_cap, vert = True, showmedians = True)
    i = 0
    for pc in violin_parts["bodies"]:
        pc.set_facecolor(colors[i])
        i += 1
    plot.xticks([])
    plot.ylabel("capacity (Mm$^3$)")
    

    fig.add_subplot(spec[1])      
    data_si = [ri_si_mean, riii_si_mean, dt_si_mean]
    labels = np.array(["policy I", "policy II", "current policy"])
    colors = ["blue", "orange", "green"]
    violin_parts = plot.violinplot(data_si, vert = True, showmedians = True)
    i = 0
    for pc in violin_parts["bodies"]:
        pc.set_facecolor(colors[i])
        i += 1
    plot.xticks([1,2,3], labels = labels)
    plot.ylabel("satisfied demand (%)")
    

def get_historical_state_under_given_policy(date_series,
                                            storage_series,
                                            inflow_series,
                                            correct_series,
                                            policy):
    
    initial_storage = model_set[1][np.argmin(abs(storage_series[0]-model_set[1]))]
    s_series = []
    c_series = []

     
    for i in range(len(date_series)):
        if i == 0:
            s = initial_storage
            month = date_series[i].month
            consum = model_set[0][policy[int(month)-1][np.argmin(abs(s - model_set[1]))]]
            c_series.append(consum)
            inflow = inflow_series[i]
            correct = correct_series[i]
            next_s = min(max(s + inflow - consum + correct, 0), 3000)
#            s_series.append((next_s + s)/2)
            s_series.append(s)
        
        else:
            s = next_s
            month = date_series[i].month
            consum = model_set[0][policy[int(month)-1][np.argmin(abs(s - model_set[1]))]]
            c_series.append(consum)
            inflow = inflow_series[i]
            correct = correct_series[i]
            next_s = min(max(s + inflow - consum+ correct, 0), 3000)
#           s_series.append((next_s + s)/2)
            s_series.append(s)
    
    return np.array(c_series), np.array(s_series)
            
# =============================================================================
# pi_consum, pi_storage = get_historical_state_under_given_policy(date_series,
#                                             storage_series,
#                                             inflow_series,
#                                             correction_series,
#                                             policy_l_ri)
# 
# piii_consum, piii_storage = get_historical_state_under_given_policy(date_series,
#                                             storage_series,
#                                             inflow_series,
#                                             correction_series,
#                                             policy_l_riii)
# 
# pc_consum, pc_storage = get_historical_state_under_given_policy(date_series,
#                                             storage_series,
#                                             inflow_series,
#                                             correction_series,
#                                             policy_default)
# =============================================================================

def get_contour_diagram_subplot(riskmap, 
                                date_series, 
                                pi_consum,
                                pi_storage,
                                piii_consum,
                                piii_storage,
                                pc_consum,
                                pc_storage,
                                consumption_series,
                                storage_series):
    
    c_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
    s_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
    
    c_list_i = [[],[],[],[],[],[],[],[],[],[],[],[]]
    s_list_i = [[],[],[],[],[],[],[],[],[],[],[],[]]    
    
    c_list_iii = [[],[],[],[],[],[],[],[],[],[],[],[]]
    s_list_iii = [[],[],[],[],[],[],[],[],[],[],[],[]]    

    c_list_c = [[],[],[],[],[],[],[],[],[],[],[],[]]
    s_list_c = [[],[],[],[],[],[],[],[],[],[],[],[]]    
    
    unit = 86400/1000000

    for i in range(len(date_series)):
        storage = model_set[1][np.argmin(abs(storage_series[i]-model_set[1]))]*unit
        consum = model_set[0][np.argmin(abs(consumption_series[i]-model_set[0]))]*unit
        consum_i =  model_set[0][np.argmin(abs(pi_consum[i]-model_set[0]))]*unit
        storage_i = model_set[1][np.argmin(abs(pi_storage[i]-model_set[1]))]*unit
        consum_iii =  model_set[0][np.argmin(abs(piii_consum[i]-model_set[0]))]*unit
        storage_iii = model_set[1][np.argmin(abs(piii_storage[i]-model_set[1]))]*unit
        consum_c =  model_set[0][np.argmin(abs(pc_consum[i]-model_set[0]))]*unit
        storage_c = model_set[1][np.argmin(abs(pc_storage[i]-model_set[1]))]*unit
        
        for j in range(12):
            
            if j + 1 == date_series[i].month:
                c_list[j].append(consum)
                s_list[j].append(storage)
                c_list_i[j].append(consum_i)
                s_list_i[j].append(storage_i)
                c_list_iii[j].append(consum_iii)
                s_list_iii[j].append(storage_iii)
                c_list_c[j].append(consum_c)
                s_list_c[j].append(storage_c)
    
    for i in range(len(c_list)):
        
        c_list[i] = np.array(c_list[i])
        s_list[i] = np.array(s_list[i])
        c_list_i[i] = np.array(c_list_i[i])
        s_list_i[i] = np.array(s_list_i[i])        
        c_list_iii[i] = np.array(c_list_iii[i])
        s_list_iii[i] = np.array(s_list_iii[i])
        c_list_c[i] = np.array(c_list_c[i])
        s_list_c[i] = np.array(s_list_c[i])

    leng = len(riskmap)
    
    fig = plot.figure(dpi = 600, figsize = (15,10)) 
    spec = gridspec.GridSpec( nrows = 3, ncols = 4)
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for i in range(leng):
        fig.add_subplot(spec[i])
        plot.title(month[i])
        fail = riskmap[i]
        c, s = model_set
        c = c*unit
        s = s*unit
        C, S = np.meshgrid(c,s, indexing = "xy")
        plot.contourf(C,S, fail, 10, cmap = cm.get_cmap("Reds"))
        if (i <= 3 and i >= 1 ) or (i >= 5 and i <= 7)  :
            plot.xticks(color = "w")
            plot.yticks(color = "w")
        elif (i+1) % 4 == 1 and i != 8:
            plot.xticks(color = "w")
            plot.ylabel("storage (M$m^3$)")
        elif i > 8:
            plot.yticks(color = "w")
            plot.xlabel("consumption (M$m^3$)")       
            
        elif i == 8:
            plot.ylabel("storage (M$m^3$)")
            plot.xlabel("consumption (M$m^3$)")        
        
        plot.scatter(c_list[i], s_list[i], c = "black", marker = "D",s = 50)
    cbar_ax = fig.add_axes([0.92,0.1,0.02,0.8]) 
    fig.colorbar(plot.contourf(C,S, fail, 10, cmap = cm.get_cmap("Reds")),
                 cax = cbar_ax)

    fig = plot.figure(dpi = 600, figsize = (15,10)) 
    spec = gridspec.GridSpec( nrows = 3, ncols = 4)
    for i in range(leng):
        
        fig.add_subplot(spec[i])
        plot.title(month[i])
        fail = riskmap[i]
        c, s = model_set
        c = c*unit
        s = s*unit
        C, S = np.meshgrid(c,s, indexing = "xy")
        plot.contourf(C,S, fail, 10, cmap = cm.get_cmap("Reds"))
        if (i <= 3 and i >= 1 ) or (i >= 5 and i <= 7)  :
            plot.xticks(color = "w")
            plot.yticks(color = "w")
        elif (i+1) % 4 == 1 and i != 8:
            plot.xticks(color = "w")
            plot.ylabel("storage (M$m^3$)")
        elif i > 8:
            plot.yticks(color = "w")
            plot.xlabel("consumption (M$m^3$)")       
            
        elif i == 8:
            plot.ylabel("storage (M$m^3$)")
            plot.xlabel("consumption (M$m^3$)")        
        
        plot.scatter(c_list_i[i], s_list_i[i], c = "blue", marker = "D",s = 50)
    cbar_ax = fig.add_axes([0.92,0.1,0.02,0.8]) 
    fig.colorbar(plot.contourf(C,S, fail, 10, cmap = cm.get_cmap("Reds")),
                 cax = cbar_ax)


    fig = plot.figure(dpi = 600, figsize = (15,10)) 
    spec = gridspec.GridSpec( nrows = 3, ncols = 4)
    for i in range(leng):
        
        fig.add_subplot(spec[i])
        plot.title(month[i])
        fail = riskmap[i]
        c, s = model_set
        c = c*unit
        s = s*unit
        C, S = np.meshgrid(c,s, indexing = "xy")
        plot.contourf(C,S, fail, 10, cmap = cm.get_cmap("Reds"))
        if (i <= 3 and i >= 1 ) or (i >= 5 and i <= 7)  :
            plot.xticks(color = "w")
            plot.yticks(color = "w")
        elif (i+1) % 4 == 1 and i != 8:
            plot.xticks(color = "w")
            plot.ylabel("storage (M$m^3$)")
        elif i > 8:
            plot.yticks(color = "w")
            plot.xlabel("consumption (M$m^3$)")       
            
        elif i == 8:
            plot.ylabel("storage (M$m^3$)")
            plot.xlabel("consumption (M$m^3$)")        
        
        plot.scatter(c_list_iii[i], s_list_iii[i],c = "purple", marker = "D",s = 50)
    cbar_ax = fig.add_axes([0.92,0.1,0.02,0.8]) 
    fig.colorbar(plot.contourf(C,S, fail, 10, cmap = cm.get_cmap("Reds")),
                 cax = cbar_ax)

    fig = plot.figure(dpi = 600, figsize = (15,10)) 
    spec = gridspec.GridSpec( nrows = 3, ncols = 4)
    for i in range(leng):
        
        fig.add_subplot(spec[i])
        plot.title(month[i])
        fail = riskmap[i]
        c, s = model_set
        c = c*unit
        s = s*unit
        C, S = np.meshgrid(c,s, indexing = "xy")
        plot.contourf(C,S, fail, 10, cmap = cm.get_cmap("Reds"))
        if (i <= 3 and i >= 1 ) or (i >= 5 and i <= 7)  :
            plot.xticks(color = "w")
            plot.yticks(color = "w")
        elif (i+1) % 4 == 1 and i != 8:
            plot.xticks(color = "w")
            plot.ylabel("storage (M$m^3$)")
        elif i > 8:
            plot.yticks(color = "w")
            plot.xlabel("consumption (M$m^3$)")       
            
        elif i == 8:
            plot.ylabel("storage (M$m^3$)")
            plot.xlabel("consumption (M$m^3$)")        
        
        plot.scatter(c_list_c[i], s_list_c[i], c = "green",marker = "D",s = 50)
    cbar_ax = fig.add_axes([0.92,0.1,0.02,0.8]) 
    fig.colorbar(plot.contourf(C,S, fail, 10, cmap = cm.get_cmap("Reds")),
                 cax = cbar_ax)

    fig = plot.figure(dpi = 600, figsize = (15,10)) 
    spec = gridspec.GridSpec( nrows = 3, ncols = 4)
    for i in range(leng):
        
        fig.add_subplot(spec[i])
        plot.title(month[i])
        fail = riskmap[i]
        c, s = model_set
        c = c*unit
        s = s*unit
        C, S = np.meshgrid(c,s, indexing = "xy")
        plot.contourf(C,S, fail, 10, cmap = cm.get_cmap("Reds"))
        if (i <= 3 and i >= 1 ) or (i >= 5 and i <= 7)  :
            plot.xticks(color = "w")
            plot.yticks(color = "w")
        elif (i+1) % 4 == 1 and i != 8:
            plot.xticks(color = "w")
            plot.ylabel("storage (M$m^3$)")
        elif i > 8:
            plot.yticks(color = "w")
            plot.xlabel("consumption (M$m^3$)")       
            
        elif i == 8:
            plot.ylabel("storage (M$m^3$)")
            plot.xlabel("consumption (M$m^3$)")
        plot.scatter(c_list_c[i], s_list_c[i], c = "green",marker = "+",s = 100)
        plot.scatter(c_list_i[i], s_list_i[i], color = "blue", s = 50)
        plot.scatter(c_list_iii[i], s_list_iii[i], color = "orange", marker = "x", s = 100)
    cbar_ax = fig.add_axes([0.92,0.1,0.02,0.8]) 
    fig.colorbar(plot.contourf(C,S, fail, 10, cmap = cm.get_cmap("Reds")),
                 cax = cbar_ax)
    

def get_event(date_series, rfd_series, threshold_series, cwdi_series):
    fail_record = []
    event = False
    deficiency = []
    temp_def = 0
    event_record = []
    temp_event = []
    for i in range(len(date_series)):
        if rfd_series[i] > threshold_series[i] and cwdi_series[i] > 0.85 and event == False:
            fail_record.append(i)
            event = True
            temp_def += max(rfd_series[i], threshold_series[i])
            temp_event.append(i)
        elif rfd_series[i] <= threshold_series[i] and cwdi_series[i] <= 0.85 and event == True:
            event = False
            deficiency.append(temp_def)
            temp_def = 0
            event_record.append(np.array(temp_event))
            temp_event = []
        elif event == True:
            fail_record.append(i)
            temp_event.append(i)
            temp_def += max(rfd_series[i], threshold_series[i])
        else:
            continue
    return fail_record, deficiency, event_record

def s_index_dateframe(storage_series, inflow_series, consum_series ,date_series):
    
    s_record = [[],[],[],[],[],[],[],[],[],[],[],[]]
    i_record = [[],[],[],[],[],[],[],[],[],[],[],[]]
    c_record = [[],[],[],[],[],[],[],[],[],[],[],[]]

    for i in range(len(storage_series)):
        for j in range(len(s_record)):
            if date_series[i].month - 1 == j:
                s_record[j].append(storage_series[i])
                i_record[j].append(inflow_series[i])
                c_record[j].append(consum_series[i])
            else:
                continue
            
    s_mean = []
    s_std = []
    c_mean = []
    c_std = []
    i_mean = []
    i_std = []
    for i in range(len(s_record)):
        s_mean.append(np.mean(np.array(s_record[i])))
        s_std.append(np.std(np.array(s_record[i])))        
        i_mean.append(np.mean(np.array(i_record[i])))    
        i_std.append(np.std(np.array(i_record[i])))    
        c_mean.append(np.mean(np.array(c_record[i])))    
        c_std.append(np.std(np.array(c_record[i])))
    
    ssi = []
    sci = []
    sii = []
    for i in range(len(storage_series)):
        for j in range(len(s_mean)):
            if j == date_series[i].month - 1:
                ssi.append((storage_series[i] - s_mean[j])/s_std[j])
                sii.append((inflow_series[i] - i_mean[j])/i_std[j])                
                sci.append((consum_series[i] - c_mean[j])/c_std[j])
    
    df = pd.DataFrame()
    df.insert(loc = 0, column = "date", value = date_series)
    df.insert(loc = 1, column = "ssi", value = np.array(ssi))
    df.insert(loc = 2, column = "sci", value = np.array(sci))
    df.insert(loc = 3, column = "sii", value = np.array(sii))
    return df

def draw_vis_major_plot(vis_major_record,
                        rfd_series, 
                        threshold_series, 
                        cwdi_series,
                        date_series,
                        ssi,
                        sci,
                        sii,
                        his_lamda_datum, 
                        his_vis_datum,
                        start, end):
    
    fig = plot.figure(figsize = (10,8), dpi = 600)
    spec = gridspec.GridSpec( nrows = 4, ncols = 1)
    lamda = vis_major_record[0]
    vis = vis_major_record[1]
    pf = vis_major_record[2]
#    dpf = np.gradient(vis_major_record[2])
    dpf = vis_major_record[3]
    delta_lamda = lamda - his_lamda_datum
    delta_vis = vis - his_vis_datum
    
    fig.add_subplot(spec[0])   
    plot.plot(np.arange(len(lamda[start-1:end])),
            lamda[start-1:end], 
            label = chr(955), 
            color = "r") 
    plot.plot(np.arange(len(vis[start-1:end])),
            vis[start-1:end],
            label = "G",
            color = "g")
    plot.xticks([])
    plot.legend() 
    
    fig.add_subplot(spec[1])
    plot.bar(np.arange(len(delta_lamda[start - 1:end])),
              delta_lamda[start-1:end],
              label = "\u0394"+chr(955),
              color = "r",
              width = 0.25)
    plot.bar(np.arange(len(delta_vis[start - 1: end]))+0.25,
              delta_vis[start - 1: end],
              label = "\u0394G",
              color = "g",
              width = 0.25)
    plot.xticks([])
    plot.legend()
    
    fig.add_subplot(spec[2])
    plot.plot(np.arange(len(dpf[start-1:end])),
              dpf[start-1:end],
              label = "\u0394Pf",
              color = "black")       
    plot.bar(np.arange(len(pf[start:end])),
            pf[start:end],
            label = "Pf",
            color = "purple")
    plot.xticks([])
    plot.legend() 
    
    fig.add_subplot(spec[3])
    plot.plot(date_series[start:end],ssi[start:end],
              color = "orange", linestyle = "--",label = "ssi")
    plot.plot(date_series[start:end],sci[start:end], 
              color = "red", linestyle = "--", label = "sci")
    plot.plot(date_series[start:end],sii[start:end],
              color = "blue", linestyle = "--", label = "sii")         
    plot.xticks(rotation = -15)
    plot.legend()
    plot.ylabel("standardized index")
    
# =============================================================================
#     fig.add_subplot(spec[2])    
#     plot.bar(date_series[start+1:end],
#              (threshold_series[start+1:end] - rfd_series[start+1:end])/ threshold_series[start+1:end],
#              color = "r", width = 5, label = "\u0394RFD / Threshold")
#     plot.axhline(0, color = "black", alpha = 0.5, linewidth = 0.5)
#     plot.ylabel("ratio")
#     plot.legend(loc = "upper left", bbox_to_anchor = (0.65,1))
#     plot.ylim((-1.5,1.5))
#     plot.xticks([])
#     plot.legend()       
#     
#     fig.add_subplot(spec[3])
#     plot.plot(date_series[start+1:end], cwdi_series[start+1:end], color = "orange", label = "CWDI")
#     plot.axhline(0.85, color = "purple", linestyle = "--", label = "recovery criteria")
#     plot.xticks(rotation = -15)
#     plot.ylabel("CWDI")
#     plot.ylim((-0.1,1.1))
#     plot.legend(loc = "upper left", bbox_to_anchor = (0.02,0.75))  
# =============================================================================




if __name__ == "__main__":
    with open("./configs.json", encoding="utf-8") as f:
        configs = json.load(f)

    risk_map = lm.annual_riskmap
    consumption_limit = 1500
    max_storage = 3000
    min_storage = 150
    resolution = 20
    discount_factors = [0.215, 0.465, 0.681]

    df_r = lm.dual_system.read_data.get_storage_correction_inflow_consum_dataframe()

    model_set = model_setting(consumption_limit, max_storage, min_storage, resolution)

    demand_list = get_demand(configs['files']['data_sheet'])

    opt_s = Q_learning(50, 12000, 0.9, 0.2, 0.1, risk_map, model_set, 1000, 1, 
                 discount_factors[0])

    opt_m = Q_learning(50, 12000, 0.9, 0.2, 0.1, risk_map, model_set, 1000, 1, 
                    discount_factors[1])

    opt_l = Q_learning(50, 12000, 0.9, 0.2, 0.1, risk_map, model_set, 1000, 1, 
                    discount_factors[2])
    
    date_series = df_r["date"]
    inflow_series = df_r["inflow"]
    storage_series = df_r["storage"]
    correction_series = df_r["correction"]
    consumption_series = df_r["consumption"]
    rfd_series = df_r["rfd"]
    threshold_series = df_r["threshold"]
    cwdi_series = df_r["cwdi"]
    
    q_l_ri = opt_l.training_mdp(0)
    policy_l_ri = np.argmax(q_l_ri[0], axis = 2)