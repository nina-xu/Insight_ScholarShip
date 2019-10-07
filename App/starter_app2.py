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
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import recommend

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# import data
data = pd.read_csv('data 258.csv')
# convert integers to float so I don't have to see warnings anymore
data.loc[:, 'TUITION2'] = data.TUITION2.astype('float64')
data.loc[:, 'avg_award'] = data.avg_award.astype('float64')

# organize institution names for drop down menu. Still needs to order alphabetically
data_dropdown = data[['INSTNM','UNITID']]
data_dropdown = data_dropdown.rename(columns={'INSTNM': 'label', 'UNITID': 'value'})
data_dropdown = data_dropdown.T.to_dict()
data_dropdown = list(data_dropdown.values())
features = ['GRNRALT_rate','TUITION2','diverse_ind','avg_award','pct_awarded']
features_actionable = ['GRNRALT_rate','diverse_ind','avg_award','pct_awarded']
feature_display_name = [
                        'International Graduation Rate',
                        #'Tuition', 
                        'Racial Diversity',
                        'Average International Financial Aid',
                        'Percent Interantional Financial Aid'
                        ]
# fit the model
scaler = StandardScaler()
scaler.fit(data[features])
transformed_linear = LinearRegression()
result = transformed_linear.fit(X = scaler.transform(data[features]), y = data.logit)

# give the range and increment for the histograms
hist_ranges = [{'start':0, 'end':1, 'size':0.05},
        #{'start':0, 'end':45000, 'size':5000},
        {'start':0.15, 'end':0.8, 'size':0.05},
        {'start':0, 'end':55000, 'size':5000},
        {'start':0, 'end':1, 'size':0.05}
        ]

# create each subplot
traces = {}
for i in range(len(features_actionable)):
    traces['trace'+str(i)] = go.Histogram(
                    x=data[features_actionable[i]],
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
fig = make_subplots(rows=2, cols=2, 
                    specs=[[{}, {}], [{}, {}]],
                    subplot_titles=feature_display_name,
                    shared_xaxes=False, shared_yaxes=False,
                    vertical_spacing=0.2)
for i in range(len(features_actionable)):
    fig.append_trace(traces['trace'+str(i)], int(i/2)+1, i%2+1)

fig['layout'].update(height=500, width=600, title='All Institution Distribution',
   showlegend = False)

# main layout of webpage
app.layout = html.Div([
    html.H1(children='ScholarShip'),

    html.Div(children='''
        Bringing higher international student enrollment 
    '''),
    html.Div([
            dcc.Graph(figure=fig, id='histograms'),
            ],
            style= {'width': '50%', 'display': 'inline-block',
                    'vertical-align': 'middle'}
    ),
    
    html.Div([
            html.H2(children = 'Select your institution:'),
            dcc.Dropdown(
                    id='my-dropdown',
                    options=data_dropdown,
                    placeholder="Select your institution",
                    ),
    
            html.Div(id='output-container'),
            html.H2(children = 'Short-term recommendation:'),
            html.Div(id='shortterm-text'),
            html.H2(children = 'Long-term recommendation:'),
            #html.Div(dcc.Graph( id = 'longterm-graph')),
            html.Div(id='longterm-text'),
            ],
            style= {'width': '49%', 'display': 'inline-block',
                    'vertical-align': 'top'})            
])

## show institution's stats, just the % of international student
@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('my-dropdown', 'value')])
def update_snapshot(value):
    if value is not None:
        return 'Currently, your institution has {}% international students'.format(
                round(data[data.UNITID == value]['intl_pct'].values[0],1))

## show institution's stats, long version
#@app.callback(
#    dash.dependencies.Output('output-container', 'children'),
#    [dash.dependencies.Input('my-dropdown', 'value')])
#def update_snapshot(value):
#    if value is not None:
#        return html.Div([
#                html.P('Your instituion snapshot:'),
#                
#                html.P(feature_display_name[0] + ': ' 
#                       + str(round(data[data.UNITID == value][features_actionable[0]].values[0]*100)) 
#                       + '%'),
#                html.P(feature_display_name[1] + ': ' + 
#                       str(round(data[data.UNITID == value][features_actionable[1]].values[0]*100))
#                       + '%'),
#                html.P(feature_display_name[2] + ': $' + str(data[data.UNITID == value][features_actionable[2]].values[0])),
#                html.P(feature_display_name[3] + ': ' 
#                       + str(round(data[data.UNITID == value][features_actionable[3]].values[0]*100))
#                       + '%')
#                ])


# show institutions position on the distributions
@app.callback(
    dash.dependencies.Output("histograms", "figure"),
    [dash.dependencies.Input('my-dropdown', 'value')])
def update_hist(unitid):
    fig = make_subplots(rows=2, cols=2, 
                    specs=[[{}, {}], [{}, {}]],
                    subplot_titles=feature_display_name,
                    shared_xaxes=False, shared_yaxes=False,
                    vertical_spacing=0.2)
    for j in range(len(features_actionable)):
        fig.append_trace(traces['trace'+str(j)], int(j/2)+1, j%2+1)

    fig['layout'].update(height=500, width=600, title='Your Institution Performance',
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
    
    if unitid is not None:
        colors = {}
        for j in range(len(features_actionable)):
            bins = create_bins(start=hist_ranges[j]['start'],
                      end=hist_ranges[j]['end'],
                      size=hist_ranges[j]['size'])
            feature_inst = data[data.UNITID == unitid][features_actionable[j]].values[0]
            colors['trace'+str(j)] = []
            
            for i in range(0, len(bins)):
                if bins[i] == bins[find_bin(feature_inst, bins)]:
                    colors['trace'+str(j)].append("#dd1c77")
                else:
                    colors['trace'+str(j)].append("#fa9fb5")
        #for j in range(len(features_actionable)):
                fig['data'][j].update(go.Histogram(marker={
                            'opacity': 0.7,
                            'color': colors['trace'+str(j)],
                    }))
        return fig

#show institution's action
@app.callback(
    #[
     #dash.dependencies.Output('shortterm-graph', 'figure'),
     dash.dependencies.Output('shortterm-text', 'children'),
    [dash.dependencies.Input('my-dropdown', 'value')])
def short_term_recommendation(value):
    if value is not None:
        inst_value = data[data.UNITID==value][features] # all features used in the model
        best_feature = recommend.select_best_feature(
                value, features_actionable[2:4], inst_value, data, result, scaler)
        recommendations = recommend.generate_recommendations(
                best_feature, value, inst_value, data)
        #histogram = recommend.generate_recommendations(
        #        best_feature, value, inst_value, data)
        return recommendations 
    
    
@app.callback(
    #[
     #dash.dependencies.Output('longterm-graph', 'figure'),
     dash.dependencies.Output('longterm-text', 'children'),
    [dash.dependencies.Input('my-dropdown', 'value')])
def longterm_recommendation(value):
    if value is not None:
        inst_value = data[data.UNITID==value][features] # all features used in the model
        best_feature = recommend.select_best_feature(
                value, features_actionable[:2], inst_value, data, result, scaler)
        recommendations = recommend.generate_recommendations(
                best_feature, value, inst_value, data)
        #histogram = recommend.generate_recommendations(
        #        best_feature, value, inst_value, data)
        return recommendations


            

if __name__ == '__main__':
    app.run_server(debug=True)
    
