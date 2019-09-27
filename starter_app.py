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
import statsmodels.api as sm
# import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# import data
data = pd.read_csv('C:/Users/Ning/Insight material/Insight_College_Ranking_for_International_Students/Data/Data national universities 274.csv')
#institution = 'Samford University'
#intl_pct = data.intl_pct[data.INSTNM == institution].tolist()[0]/100
#diverse_ind = data.diverse_ind[data.INSTNM == institution].tolist()[0]



# organize institution names for drop down menu. Still needs to order alphabetically
data_dropdown = data[['INSTNM','UNITID']]
data_dropdown = data_dropdown.rename(columns={'INSTNM': 'label', 'UNITID': 'value'})
data_dropdown = data_dropdown.T.to_dict()
data_dropdown = list(data_dropdown.values())
features = ['diverse_ind','GRNRALT_rate','tuition_US_news', 'ranking_US_news','good_for_intl_US_news']
feature_display_name = ['Diversity Index',
                        'International Graduation Rate',
                        'Tuition', 
                        'US News Ranking',
                        'US News List of Best Us for Intl'
                        ]

# fit the model
binomial_model=sm.GLM(
    data[['EFYNRALT','EFYUS']], #[success, failure]
    data[features],
    family = sm.families.Binomial())
result=binomial_model.fit()

# give the range and increment for the histograms
hist_ranges = [{'start':0.05, 'end':0.9, 'size':0.05},
        {'start':0.1, 'end':1, 'size':0.05},
        {'start':0, 'end':70000, 'size':5000},
        {'start':0, 'end':300, 'size':15},
        {'start':0, 'end':2, 'size':1}
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
                    vertical_spacing=0.001)
for i in range(len(features)):
    fig.append_trace(traces['trace'+str(i)], int(i/3)+1, i%3+1)

fig['layout'].update(height=600, width=1000, title='Relavant Institution Performance',
   showlegend = False)

# main layout of webpage
app.layout = html.Div([
    html.H1(children='Universal University'),

    html.Div(children='''
        Boosting your international student enrollment
    '''),
    
    dcc.Dropdown(
        id='my-dropdown',
        options=data_dropdown,
        placeholder="Select your institution",),
    
    html.Div(id='output-container'),
             

    dcc.Graph(figure=fig, id='histograms'),
    
    html.Div(id='insight-container')
    
])

# show institution's stats
@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('my-dropdown', 'value')])
def update_output(value):
    output = 'Your instituion snapshot:'
    for feature in feature_display_name:
        newline = feature + ':' + str(data[data.UNITID == value][feature].values[0])
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
                    vertical_spacing=0.001)
    for i in range(len(features)):
        fig.append_trace(traces['trace'+str(i)], int(i/3)+1, i%3+1)

    fig['layout'].update(height=600, width=1000, title='Relavant Institution Performance')


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
        for i in range(0, len(bins)):
            if bins[i][0] <= index < bins[i][1]:
                return i
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
        #for j in range(len(features)):
                fig['data'][j].update(go.Histogram(marker={
                            'opacity': 0.7,
                            'color': colors['trace'+str(j)],
                    }))
        return fig

#show institution's action
@app.callback(
    dash.dependencies.Output('insight-container', 'children'),
    [dash.dependencies.Input('my-dropdown', 'value')])
def update_insight(value):
    pred0 = result.predict(data[data.UNITID==value][features])
    new_x = data[data.UNITID==value][features]
    new_x['diverse_ind'] = new_x['diverse_ind']*1.1
    pred1 = result.predict(new_x)
    improvement = (pred1-pred0)* 100
    return 'If you increase your diversity index by 10%, your international student enrollment will increase by {}%'.format(improvement.values[0])  

if __name__ == '__main__':
    app.run_server(debug=True)
    
