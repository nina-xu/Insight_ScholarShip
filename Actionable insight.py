# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:11:06 2019

@author: Ning
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# import data
data = pd.read_csv('Data/data 258.csv')
# convert integers to float so I don't have to see warnings anymore
data.loc[:, 'TUITION2'] = data.TUITION2.astype('float64')
data.loc[:, 'avg_award'] = data.avg_award.astype('float64')

features = ['GRNRALT_rate','TUITION2','diverse_ind','avg_award','pct_awarded']
# fit the model
scaler = StandardScaler()
scaler.fit(data[features])
transformed_linear = LinearRegression()
result = transformed_linear.fit(X = scaler.transform(data[features]), y = data.logit)

def get_percentile(feature, inst_value, data, extra_pct = 0.1):
    """
    don't put in 0 as extra_pct, don't work
    """
    feature_value = inst_value[feature].values[0]
    percent = np.mean(data[feature]< feature_value)
    new_percent = np.min([percent + extra_pct, 1])
    new_feature_value = np.percentile(data[feature], new_percent*100)
    # if new feature value is 0 (in the case of financial aid)
    if new_feature_value == 0:
        new_feature_value = np.min(data[data[feature]>0][feature])
        new_percent = np.mean(data[feature]< new_feature_value)
    return {'feature_value':feature_value, 
            'percent':percent, 
            'new_feature_value':new_feature_value, 
            'new_percent':new_percent}

def reverse_logit(logit):
    return np.exp(logit)/(1 + np.exp(logit))

def get_improvement(feature, inst_value, data, result, scaler, extra_pct = 0.1):
    input_update = get_percentile(feature, inst_value, data, extra_pct)
    old_prediction = result.predict(X = scaler.transform(inst_value))
    old_intl_pct = reverse_logit(old_prediction[0])
    new_inst_value = inst_value.copy()
    new_inst_value[feature] = input_update['new_feature_value']
    new_prediction = result.predict(X = scaler.transform(new_inst_value))
    new_intl_pct = reverse_logit(new_prediction[0])
    increase = (new_intl_pct - old_intl_pct)*100
    return round(np.min([increase, 100]),2)

# calculate the net gain in tuition if they increase scholarship
def net_gain_avg_aid(value, data, result, scaler, extra_pct = 0.1):
    inst_value = data[data.UNITID==value][features]
    update = get_percentile('avg_award', inst_value, data, extra_pct)
    imp = get_improvement('avg_award', inst_value, data, result, scaler, extra_pct)
    student_gain = data['EFYTOTLT'][data.UNITID==value]*imp/100
    tuition_gain = student_gain * data['TUITION2'][data.UNITID==value]
    scholarship_pay = data['EFYNRALT'][data.UNITID==value] * (update['new_feature_value']- update['feature_value'])
    net_gain = tuition_gain - scholarship_pay
    return net_gain.values[0]

def net_gain_pct_aid(value, data, result, scaler, extra_pct = 0.1):
    inst_value = data[data.UNITID==value][features]
    update = get_percentile('pct_awarded', inst_value, data, extra_pct)
    imp = get_improvement('pct_awarded', inst_value, data, result, scaler, extra_pct)
    student_gain = data['EFYTOTLT'][data.UNITID==value]*imp/100
    tuition_gain = student_gain * data['TUITION2'][data.UNITID==value]
    scholarship_pay = data['EFYNRALT'][data.UNITID==value] * (update['new_feature_value']- update['feature_value'])/100*data['avg_award'][data.UNITID==value]
    net_gain = tuition_gain - scholarship_pay
    return net_gain.values[0]

# generate a list of insitutions that can benefit from increasing the percent of scholarship, at all 
increments = np.array(range(1,10))/100
pct_yes = []
gained_institutions_pct = []
for increment in increments:
    gains = []
    for value in data.UNITID:
        gain = net_gain_pct_aid(value, data, result, scaler, extra_pct = increment)
        if gain > 0:
            gained_institutions_pct.append(data.UNITID[data.UNITID==value].values[0])
        gains.append(gain)
    yes = [gain>0 for gain in gains]
    pct_yes.append(np.mean(yes))

#print(pct_yes)
# 36 institutions benefit from raising % scholarship
gained_institutions_pct_unique = np.unique(gained_institutions_pct)

# get the recommendations for these institutions
def best_gain_pct_aid(value, data, result, scaler):
    increments = np.array(range(1,10))/100
    max_gain = 0
    for increment in increments:
        gain = net_gain_pct_aid(value, data, result, scaler, extra_pct = increment)
        if gain > max_gain:
            max_gain = gain
            extra_pct_recommended = increment
    rec = {}
    rec['UNITID'] = value
    rec['max_gain'] = max_gain
    rec['extra_pct_recommended'] = extra_pct_recommended
    return rec

recs_pct = []
for value in gained_institutions_pct_unique:
    rec = best_gain_pct_aid(value, data, result, scaler)
    recs_pct.append(rec)
pd.DataFrame(recs_pct).to_csv('Data/inst_gain_from_pct_aid.csv', index = False)

# generate a list of insitutions that can benefit from increasing the percent of scholarship, at all 
increments = np.array(range(1,10))/100
gained_institutions_avg = []
for increment in increments:
    gains = []
    for value in data.UNITID:
        gain = net_gain_avg_aid(value, data, result, scaler, extra_pct = increment)
        if gain > 0:
            gained_institutions_avg.append(data.UNITID[data.UNITID==value].values[0])
        gains.append(gain)

#print(pct_yes)
# only 12 schools benefit from raising average scholarship
gained_institutions_avg_unique = np.unique(gained_institutions_avg)
# get the recommendations for these institutions
def best_gain_avg_aid(value, data, result, scaler):
    increments = np.array(range(1,10))/100
    max_gain = 0
    for increment in increments:
        gain = net_gain_avg_aid(value, data, result, scaler, extra_pct = increment)
        if gain > max_gain:
            max_gain = gain
            extra_pct_recommended = increment
    rec = {}
    rec['UNITID'] = value
    rec['max_gain'] = max_gain
    rec['extra_pct_recommended'] = extra_pct_recommended
    return rec

recs_avg = []
for value in gained_institutions_avg_unique:
    rec = best_gain_avg_aid(value, data, result, scaler)
    recs_avg.append(rec)
pd.DataFrame(recs_avg).to_csv('Data/inst_gain_from_avg_aid.csv', index = False)