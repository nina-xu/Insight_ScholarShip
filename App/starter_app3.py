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
                        'Intl Graduation Rate',
                        #'Tuition', 
                        'Racial Diversity',
                        'Average Intl Financial Aid',
                        'Percent Intl Financial Aid'
                        ]
xaxis_titles = ['Graduation Rate (%)','Diversity (%)','Avg Aid ($)','% Receiving Aid']
# fit the model
scaler = StandardScaler()
scaler.fit(data[features])
transformed_linear = LinearRegression()
result = transformed_linear.fit(X = scaler.transform(data[features]), y = data.logit)

# create each subplot
traces = {}
for i in range(len(features_actionable)):
    traces['trace'+str(i)] = go.Scatter(
                    x = data[features_actionable[i]],
                    y = data['intl_pct'],
                    mode='markers',
                    hovertext=data['INSTNM'],
                    showlegend = False,
                    name = feature_display_name[i],
                    #hoverinfo = 'text',
                    marker={
                            'opacity': 0.7,
                            'color': "#fa9fb5",
                    }
                )

# put the subplots together
fig = make_subplots(rows=2, cols=2, 
                    specs=[[{}, {}], [{}, {}]],
                    subplot_titles=feature_display_name,
                    shared_xaxes=False, shared_yaxes=True,
                    vertical_spacing=0.3)
for i in range(len(features_actionable)):
    fig.append_trace(traces['trace'+str(i)], int(i/2)+1, i%2+1)
    fig.update_xaxes(title_text=xaxis_titles[i], row = int(i/2)+1, col = i%2+1)
    if i % 2 ==0:
        fig.update_yaxes(title_text='% Intl Students', row = int(i/2)+1, col = i%2+1)

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
            style= {'width': '48%', 'display': 'inline-block',
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
            style= {'width': '51%', 'display': 'inline-block',
                    'vertical-align': 'top'})            
])

## show institution's stats, just the % of international student
#@app.callback(
#    dash.dependencies.Output('output-container', 'children'),
#    [dash.dependencies.Input('my-dropdown', 'value')])
#def update_snapshot(value):
#    if value is not None:
#        return 'Currently, your institution has {}% international students'.format(
#                round(data[data.UNITID == value]['intl_pct'].values[0],1))



# show institutions position on the distributions
@app.callback(
    dash.dependencies.Output("histograms", "figure"),
    [dash.dependencies.Input('my-dropdown', 'value')])
def update_scatter(unitid):
    fig = make_subplots(rows=2, cols=2, 
                        specs=[[{}, {}], [{}, {}]],
                        subplot_titles=feature_display_name,
                        shared_xaxes=False, shared_yaxes=True,
                        vertical_spacing=0.3)
    for i in range(len(features_actionable)):
        fig.append_trace(traces['trace'+str(i)], int(i/2)+1, i%2+1)
        fig.update_xaxes(title_text=xaxis_titles[i], row = int(i/2)+1, col = i%2+1)
        if i % 2 ==0:
            fig.update_yaxes(title_text='% Intl Students', row = int(i/2)+1, col = i%2+1)
    
```    fig['layout'].update(height=500, width=600, title='Your Institution Performance',
       showlegend = False)

    
    if unitid is not None:
#        inst_value = data[data.UNITID==unitid][features] # all features used in the model
#        best_feature_short = recommend.select_best_feature(
#                    unitid, features_actionable[2:4], inst_value, data, result, scaler)
#        recommendations_short = recommend.generate_recommendations(
#                    best_feature_short, unitid, inst_value, data)
#        best_feature_long = recommend.select_best_feature(
#                    unitid, features_actionable[:2], inst_value, data, result, scaler)
#        recommendations_long = recommend.generate_recommendations(
#                    best_feature_long, unitid, inst_value, data)
        colors = {}
        opacity = {}
        sizes = {}
        
        for j in range(len(features_actionable)):
            colors['trace'+str(j)] = ["#dd1c77" if ID == unitid else "#fa9fb5" for ID in data.UNITID]
            opacity['trace'+str(j)] = [1 if ID == unitid else 0.5 for ID in data.UNITID]
            sizes['trace'+str(j)] = [10 if ID == unitid else 7 for ID in data.UNITID]
            fig['data'][j]['marker']['color'] = colors['trace'+str(j)]
            fig['data'][j]['marker']['opacity'] = opacity['trace'+str(j)]
            fig['data'][j]['marker']['size'] = sizes['trace'+str(j)]
            #update hovering text into including recommendations
#            if features_actionable[j] == best_feature_short[0]:
#                fig['data'][j]['hovertext'] = recommendations_short
#            if features_actionable[j] == best_feature_long[0]:
#                fig['data'][j]['hovertext'] = recommendations_long
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
    
