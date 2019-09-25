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

app.layout = html.Div([
    html.H1(children='Universal University'),

    html.Div(children='''
        Boosting your international student acquisition 
    '''),
    
    dcc.Dropdown(
        id='my-dropdown',
        options=data_dropdown,
        value=data_dropdown[0]['value']),
    
    html.Div(id='output-container'),
             
    dcc.Graph(
        id='institution-graph',
        figure={
            'data': [
                go.Histogram(
                    x=data.diverse_ind,
                    histnorm='probability'
                ) 
            ],
            'layout': {
                    'title': institution
                    }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
    
