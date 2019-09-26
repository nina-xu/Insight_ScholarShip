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
# import os
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

data = pd.read_csv('C:/Users/Ning/Insight material/Insight_College_Ranking_for_International_Students/Data/Data national universities 274.csv')
institution = 'Samford University'
intl_pct = data.intl_pct[data.INSTNM == institution].tolist()[0]/100
diverse_ind = data.diverse_ind[data.INSTNM == institution].tolist()[0]
# organize institution names for drop down menu. Still needs to order alphabetically
data_dropdown = data[['INSTNM','UNITID']]
data_dropdown = data_dropdown.rename(columns={'INSTNM': 'label', 'UNITID': 'value'})
data_dropdown = data_dropdown.T.to_dict()
data_dropdown = list(data_dropdown.values())
features = ['diverse_ind','GRNRALT_rate','tuition_US_news', 'ranking_US_news','good_for_intl_US_news']


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
             
    dcc.Graph(
        id='hist-diverse',
        figure={
            'data': [
                go.Histogram(
                    x=data.diverse_ind,
                    marker={
                            'opacity': 0.7,
                            'color': "#fa9fb5",
                    }
                ) 
            ],
            'layout': {
                    'title': 'Diversity Index'
                    }
        }
    )
])

@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('my-dropdown', 'value')])
def update_output(value):
    output = 'Your instituion snapshot:'
    for feature in features:
        newline = feature + ':' + str(data[data.UNITID == value][feature].values[0])
        output = output + newline
    return output

@app.callback(
    dash.dependencies.Output("hist-diverse", "figure"),
    [dash.dependencies.Input('my-dropdown', 'value')])
def update_hist(value):
    diverse_ind_inst = data.diverse_ind[data.UNITID == value].values[0]

    figure={
            'data': [
                go.Histogram(
                    x=data.diverse_ind,
                    marker={
                            'opacity': 0.7,
                            'color': "#fa9fb5",
                    }
                ) 
            ],
            'layout': {
                    'title': 'Diversity Index'
                    }
        }

    def create_bins(lower_bound, width, quantity):
        breaks = []
        for i in range(quantity):
            breaks.append(lower_bound+width*i)
        bins = []
        for low in breaks:
            bins.append((low, low + width))
        return bins

    bins = create_bins(lower_bound=0.05, width=0.05, quantity=17)

    def find_bin(index, bins):
        for i in range(0, len(bins)):
            if bins[i][0] <= index < bins[i][1]:
                return i
        return -1

    if value is not None:
        color = []
        for i in range(0, len(bins)):
            if bins[i] == bins[find_bin(diverse_ind_inst, bins)]:
                color.append("#dd1c77")
            else:
                color.append("#fa9fb5")
        figure["data"][0].update(go.Histogram(marker={
                            'opacity': 0.7,
                            'color': color,
                    }))
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
    
