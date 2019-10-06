# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:45:33 2019

@author: Ning
"""


# you do dash, do the dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# import data
data = pd.read_csv('data 258.csv')

# organize institution names for drop down menu. Still needs to order alphabetically
data_dropdown = data[['INSTNM','UNITID']]
data_dropdown = data_dropdown.rename(columns={'INSTNM': 'label', 'UNITID': 'value'})
data_dropdown = data_dropdown.T.to_dict()
data_dropdown = list(data_dropdown.values())
features = ['GRNRALT_rate','TUITION2','diverse_ind','avg_award','pct_awarded']
features_actionable = ['GRNRALT_rate','diverse_ind','avg_award','pct_awarded']
# standardize features
features_std = []
for i in range(len(features)):
    features_std.append(features[i]+'_std')
#feat_normalize = scale(data[features],  axis=0)
#for i in range(len(features)):
#    data.loc[:,features[i]+'_std'] = feat_normalize[:,i]
## logit transformation of output
#logit = np.log(data.intl_pct/(100-data.intl_pct))
#data.loc[:,'logit'] = logit
    
feature_display_name = [
                        'International Graduation Rate',
                        'Tuition', 
                        'Diversity Index',
                        'Financial Aid $',
                        'Financial Aid %'
                        ]

# fit the model
scaler = StandardScaler()
scaler.fit(data[features])
transformed_linear = LinearRegression()
result = transformed_linear.fit(X = scaler.transform(data[features]), y = data.logit)
# give the range and increment for the histograms
hist_ranges = [{'start':0, 'end':1, 'size':0.05},
        {'start':0, 'end':45000, 'size':5000},
        {'start':0.15, 'end':0.8, 'size':0.05},
        {'start':0, 'end':55000, 'size':5000},
        {'start':0, 'end':1, 'size':0.05}
        ]

# create each subplot
traces = {}
for i in range(len(features)):
    traces['trace'+str(i)] = go.Histogram(
                    x=data[features[i]],
                    xbins=dict(
                      start=hist_ranges[i]['start'],
                      end=hist_ranges[i]['end'],
                      size=hist_ranges[i]['size']),
                    autobinx=False,
                    marker={
                            'opacity': 0.7,
                            'color': "#fa9fb5",
                    }
                )

# put the subplots together
fig = make_subplots(rows=2, cols=3, 
                    specs=[[{}, {}, {}], [{}, {}, {}]],
                    subplot_titles=feature_display_name,
                    shared_xaxes=False, shared_yaxes=False,
                    vertical_spacing=0.2)
for i in range(len(features)):
    fig.append_trace(traces['trace'+str(i)], int(i/3)+1, i%3+1)

fig['layout'].update(height=600, width=1000, title='All Institution Distribution',
   showlegend = False)

# main layout of webpage
app.layout = html.Div([
    html.H1(children='ScholarShip'),

    html.Div(children='''
        Bring higher international student enrollment 
    '''),
    
    dcc.Dropdown(
        id='my-dropdown',
        options=data_dropdown,
        placeholder="Select your institution",),
    
    html.Div(id='output-container'),
             

    dcc.Graph(figure=fig, id='histograms'),
    
    html.Div(id='insight-container')
    
])

## show institution's stats
@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('my-dropdown', 'value')])
def update_output(value):
    output = 'Your instituion snapshot:'
    for i in range(len(features)):
        newline = feature_display_name[i] + ':' + str(data[data.UNITID == value][features[i]].values[0])
        output = output + newline
    return output

# show institutions position on the distributions
@app.callback(
    dash.dependencies.Output("histograms", "figure"),
    [dash.dependencies.Input('my-dropdown', 'value')])
def update_hist(value):
    fig = make_subplots(rows=2, cols=3, 
                    specs=[[{}, {}, {}], [{}, {}, {}]],
                    subplot_titles=feature_display_name,
                    shared_xaxes=False, shared_yaxes=False,
                    vertical_spacing=0.2)
    for j in range(len(features)):
        fig.append_trace(traces['trace'+str(j)], int(j/3)+1, j%3+1)

    fig['layout'].update(height=600, width=1000, title='Your Institution Performance',
       showlegend = False)


    def create_bins(start, end, size):
        breaks = []
        quantity = int((end-start)/size)
        for i in range(quantity):
            breaks.append(start+size*i)
        bins = []
        for low in breaks:
            bins.append((low, low + size))
        return bins

    def find_bin(index, bins):
        for i in range(0, len(bins)-1):
            if bins[i][0] <= index < bins[i][1]:
                return i
        # last bin
        if bins[len(bins)-1][0] <= index <= bins[len(bins)-1][1]:
            return len(bins)-1
        return -1
    
    if value is not None:
        colors = {}
        for j in range(len(features)):
            bins = create_bins(start=hist_ranges[j]['start'],
                      end=hist_ranges[j]['end'],
                      size=hist_ranges[j]['size'])
            feature_inst = data[data.UNITID == value][features[j]].values[0]
            colors['trace'+str(j)] = []
            
            for i in range(0, len(bins)):
                if bins[i] == bins[find_bin(feature_inst, bins)]:
                    colors['trace'+str(j)].append("#dd1c77")
                else:
                    colors['trace'+str(j)].append("#fa9fb5")
        for j in range(len(features)):
                fig['data'][j].update(go.Histogram(marker={
                            'opacity': 0.7,
                            'color': colors['trace'+str(j)],
                    }))
        return fig

#show institution's action
#@app.callback(
#    dash.dependencies.Output('insight-container', 'children'),
#    [dash.dependencies.Input('my-dropdown', 'value')])
#def update_recommendation(value, features = features_actionable):
#    inst_value = data[data.UNITID==value][features]
#    def get_percentile(feature, inst_value):
#        feature_value = inst_value[feature].values[0]
#        percent = np.mean(data[feature]< feature_value)
#        new_percent = np.min([percent + 0.1, 1])
#        new_feature_value = np.percentile(data[feature], new_percent*100)
#        return {'feature_value':feature_value, 
#                'percent':percent, 
#                'new_feature_value':new_feature_value, 
#                'new_percent':new_percent}
#    def reverse_logit(logit):
#        return np.exp(logit)/(1 + np.exp(logit))
#    def get_improvement(feature, inst_value):
#        input_update = get_percentile(feature, inst_value)
#        old_prediction = result.predict(X = scaler.transform(inst_value))
#        old_intl_pct = reverse_logit(old_prediction[0])
#        new_inst_value = inst_value.copy()
#        new_inst_value[feature] = input_update['new_feature_value']
#        new_prediction = result.predict(X = scaler.transform(new_inst_value))
#        new_intl_pct = reverse_logit(new_prediction[0])
#        increase = (new_intl_pct - old_intl_pct)*100
#        return round(np.min([increase, 100]),2)
#    def select_best_feature(value, features):
#        improvements = {}
#        for feature in features:
#            improvements[feature] = get_improvement(feature, inst_value)
#        best_feature = max(improvements, key = improvements.get)
#        return([best_feature, improvements])
#    
#    def generate_recommendations(best_feature):
#        if best_feature[0] == 'GRNRALT_rate':
#            update = get_percentile('GRNRALT_rate', inst_value)
#            string = "Currently, your international student graduation rate is {}%, better than {}% of the institutions. As a reference, your domestic student graduation rate is X%. If you boost your international graduation rate to {} (better than {}% of the institutions), you should attract an additional {}% of international students."
#            output = string.format( 
#                str(round(update['feature_value']*100,1)), 
#                str(round(update['percent']*100,1)),
#                str(round(update['new_feature_value']*100,1)),
#                str(round(update['new_percent']*100,1)),
#                str(best_feature[1]['GRNRALT_rate'])
#                       )
#            return output
#        if best_feature[0] == 'diverse_ind':
#            update = get_percentile(best_feature[0], inst_value)
#            string = "Currently, your racial diversity index is {}%, better than {}% of the institutions. (*Also show a distribution of races.) If you recruit more domestic students of racial minorities, and boost your diversity index to {} (better than {}% of the institutions), you should attract an additional {}% of international students."
#            output = string.format( 
#                str(round(update['feature_value']*100,1)), 
#                str(round(update['percent']*100,1)),
#                str(round(update['new_feature_value']*100,1)),
#                str(round(update['new_percent']*100,1)),
#                str(best_feature[1][best_feature[0]])
#                       )
#            return output
#        if best_feature[0] == 'avg_award': 
#            update = get_percentile(best_feature[0], inst_value)
#            string = "Currently, your average international student receives ${} in financial aid, better than {}% of the institutions. If you increase your average financial aid to ${} (better than {}% of the institutions), you should attract an additional {}% of international students."
#            output = string.format( 
#                str(round(update['feature_value'])), 
#                str(round(update['percent']*100,1)),
#                str(round(update['new_feature_value'])),
#                str(round(update['new_percent']*100,1)),
#                str(best_feature[1][best_feature[0]])
#                       )
#            return output
#        if best_feature[0] == 'pct_awarded': 
#            update = get_percentile(best_feature[0], inst_value)
#            string = "Currently, {}% of your international students receive financial aid, better than {}% of the institutions. If you award financial aid to {}% of the international students (better than {}% of the institutions), you should attract an additional {}% of international students."
#            output = string.format( 
#                str(round(update['feature_value']*100,1)), 
#                str(round(update['percent']*100,1)),
#                str(round(update['new_feature_value']*100,1)),
#                str(round(update['new_percent']*100,1)),
#                str(best_feature[1][best_feature[0]])
#                       )
#            return output
#    best_feature = select_best_feature(value, features= features_actionable)
#    recommendations = generate_recommendations(best_feature)
#    return recommendations

if __name__ == '__main__':
    app.run_server(debug=True)
    
