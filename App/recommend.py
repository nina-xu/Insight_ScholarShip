# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:18:20 2019

@author: Ning
"""
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
# from diversity import make_bar

avg_gain = pd.read_csv('inst_gain_from_avg_aid.csv')
pct_gain = pd.read_csv('inst_gain_from_pct_aid.csv')

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

def select_best_feature(value, features_actionable, inst_value, data, result, scaler, extra_pct = 0.1):
    improvements = {}
    for feature in features_actionable:
        improvements[feature] = get_improvement(feature, inst_value, data, result, scaler, extra_pct)
    best_feature = max(improvements, key = improvements.get)
    return([best_feature, improvements])

def select_best_feature_shortterm(value):
    if value in list(avg_gain.UNITID) and value in list(pct_gain.UNITID):
        maxgain_avg = avg_gain.max_gain[avg_gain.UNITID == value]
        maxgain_pct = pct_gain.max_gain[pct_gain.UNITID == value]
        if maxgain_avg.values[0] > maxgain_pct.values[0]:
            return 'avg_award'
        else:
            return 'pct_awarded'
    elif value in list(avg_gain.UNITID):
        return 'avg_award'
    elif value in list(pct_gain.UNITID):
        return 'pct_awarded'
    else:
        return None

def generate_recommendations_longterm(best_feature, value, inst_value, data):
    header = 'Long-term recommendation:'
    if best_feature[0] == 'GRNRALT_rate':
        update = get_percentile('GRNRALT_rate', inst_value, data)
        string = "Currently, your international student graduation rate is {}%, better than {}% of the institutions. As a reference, your overall graduation rate is {}%. If you boost your international graduation rate to {}% (better than {}% of the institutions), you should attract an additional {}% of international students."
        output_text = html.Div([
                html.P(
                string.format( 
                        str(round(update['feature_value'],1)), 
                        str(round(update['percent']*100,1)),
                        str(round(data.GRTOTLT_rate[data.UNITID==value].values[0]*100,1)),
                        str(round(update['new_feature_value'],1)),
                        str(round(update['new_percent']*100,1)),
                        str(best_feature[1]['GRNRALT_rate'])
                   )
                ),
                html.P('Additional resource: '),
                dcc.Link('https://www.nafsa.org/professional-resources/publications/retaining-international-students', 
                         href='https://www.nafsa.org/professional-resources/publications/retaining-international-students')]
    )
        #output_graph = make_bar(data[data.UNITID == value])
        return header, output_text# output_graph#,
    elif best_feature[0] == 'diverse_ind':
        update = get_percentile(best_feature[0], inst_value, data)
        string = "Currently, your racial diversity index is {}%, better than {}% of the institutions. If you recruit more domestic students of racial minorities, and boost your diversity index to {}% (better than {}% of the institutions), you should attract an additional {}% of international students."
        output_text = string.format( 
            str(round(update['feature_value'],1)), 
            str(round(update['percent']*100,1)),
            str(round(update['new_feature_value'],1)),
            str(round(update['new_percent']*100,1)),
            str(best_feature[1][best_feature[0]])
                   )
        #output_graph = make_bar(data[data.UNITID == value])
        return  header, output_text# output_graph#,

def generate_recommendations_shortterm(value, 
                                  features_actionable, # put in % aid & avg aid
                                  inst_value, data, result, scaler):
    best_feature = select_best_feature_shortterm(value)
    header = 'Short-term recommendation:'
    if best_feature is None:
        return '', ''
    elif best_feature == 'avg_award':
        increment = avg_gain.extra_pct_recommended[avg_gain.UNITID == value].values[0]
        update = get_percentile(best_feature, inst_value, data, extra_pct = increment)
        improvement = get_improvement(best_feature, inst_value, data, result, scaler, extra_pct = increment)
        string = "Currently, your average international student receives ${} in financial aid, better than {}% of the institutions. If you increase your average financial aid to ${} (better than {}% of the institutions), you should attract an additional {}% of international students. You net gain in tuition is projected to be ${} per year."
        output = string.format( 
            str(round(update['feature_value'])), 
            str(round(update['percent']*100,1)),
            str(round(update['new_feature_value'])),
            str(round(update['new_percent']*100,1)),
            str(improvement), 
            str(round(avg_gain.max_gain[avg_gain.UNITID == value].values[0], 1))
                   )
        
        return header, output
    elif best_feature == 'pct_awarded':
        increment = pct_gain.extra_pct_recommended[pct_gain.UNITID == value].values[0]
        update = get_percentile(best_feature, inst_value, data, extra_pct = increment)
        improvement = get_improvement(best_feature, inst_value, data, result, scaler, extra_pct = increment)
        string = "Currently, {}% of your international students receive financial aid, better than {}% of the institutions. If you award financial aid to {}% of the international students (better than {}% of the institutions), you should attract an additional {}% of international students. You net gain in tuition is projected to be ${} per year."
        output = string.format( 
            str(round(update['feature_value'])), 
            str(round(update['percent']*100,1)),
            str(round(update['new_feature_value'])),
            str(round(update['new_percent']*100,1)),
            str(improvement), 
            str(round(pct_gain.max_gain[pct_gain.UNITID == value].values[0], 1))
                   )
        
        return header, output