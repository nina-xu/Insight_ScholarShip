# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 18:31:16 2019

@author: Ning
"""
#import plotly.express as px
#import pandas as pd
import plotly.graph_objs as go

def make_bar(data):
    my_total = (data['EFYAIANT']+data['EFYASIAT']+data['EFYBKAAT']
    +data['EFYHISPT']+data['EFYWHITT']+data['EFYNHPIT']+data['EFY2MORT'])/100
    my_total = my_total.values[0]
    pct = {}
    pct['AIAN'] = data['EFYAIANT'].values[0]/my_total
    pct['ASIA'] = data['EFYASIAT'].values[0]/my_total
    pct['BKAA'] = data['EFYBKAAT'].values[0]/my_total
    pct['HISP'] = data['EFYHISPT'].values[0]/my_total
    pct['WHIT'] = data['EFYWHITT'].values[0]/my_total
    pct['NHPI'] = data['EFYNHPIT'].values[0]/my_total
    pct['2MOR'] = data['EFY2MORT'].values[0]/my_total
    name = {'AIAN':'American Indian or Alaska Native',
           'ASIA': 'Asian',
           'BKAA': 'Black',
            'HISP': 'Hispanic',
            'WHIT':'White',
            'NHPI': 'Native Hawaiian or Other Pacific Islander',
            '2MOR': 'Two or more races'
           }
    #df = pd.DataFrame.from_records([pct,name]).T
    #df.columns=['pct','name']
    return {
            'data': [
                    go.Bar(x = name, y = pct)
            ],
            'layout':[
                    go.Layout(
                            xaxis={
                                    'title': 'Race'
                                    },
                            yaxis={
                                    'title': 'Percent'
                                    })
                    ]
    
    }
            
    
