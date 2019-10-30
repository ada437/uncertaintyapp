#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 00:16:37 2019

@author: xavierochoa
"""

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, ClientsideFunction

import pandas as pd
import numpy
from sklearn import cluster
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import pickle

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

server = app.server
app.config.suppress_callback_exceptions = False

students = pd.read_csv("diabetes_student.csv")
semesters = pd.read_csv("diabetes_semester.csv")
students_train = pd.read_csv("students_train.csv")
semesters_train = pd.read_csv("semesters_train.csv")
clus_students = pickle.load(open("student_model.sav", 'rb'))
clus_semesters = pickle.load(open("semester_model.sav", 'rb'))
rf_model=pickle.load(open("rf_model.sav",'rb'))

semesters_test=semesters[semesters['ID'] > 350]
students_test=students[students['ID'].isin(semesters_test['ID'].tolist())]


def get_student_data(student_id):
    student_data=students_test[students_test['ID']==student_id]
    print(student_data)
    Insulin=student_data['Insulin'].values[0]
    SkinThickness=student_data['SkinThickness'].values[0]
    BloodPressure=student_data['BloodPressure'].values[0]
    Glucose=student_data['Glucose'].values[0]
    return Insulin,SkinThickness,BloodPressure,Glucose

def get_semester_data(student_id):
    semester_data=semesters_test[semesters_test['ID']==student_id]
    Pregnancies=semester_data['Pregnancies'].values[0]
    BMI=semester_data['BMI'].values[0]
    DiabetesPedigreeFunction=semester_data['DiabetesPedigreeFunction'].values[0]
    Age=semester_data['Age'].values[0]
    return Pregnancies,BMI,DiabetesPedigreeFunction,Age

def get_new_risk_and_uncertainty(Insulin,SkinThickness,BloodPressure,Glucose,Pregnancies, BMI,DiabetesPedigreeFunction, Age):
    columns_student=['Insulin','SkinThickness','BloodPressure','Glucose']
    columns_semester=['Pregnancies', 'BMI','DiabetesPedigreeFunction', 'Age']
    student_data = pd.DataFrame([[Insulin,SkinThickness,BloodPressure,Glucose]], columns=columns_student)
    semester_data = pd.DataFrame([[Pregnancies, BMI,DiabetesPedigreeFunction, Age]], columns=columns_semester)
    student_cluster=clus_students.predict(student_data)[0]
    semester_cluster=clus_semesters.predict(semester_data)[0]
    
    similar_students=students_train[students_train["cluster"]==student_cluster]
    similar_students_ids=similar_students["ID"].tolist()
    
    selected_semesters=semesters_train[semesters_train["ID"].isin(similar_students_ids)]
    selected_semesters=selected_semesters[selected_semesters['cluster']==semester_cluster]
    
    total_cases=len(selected_semesters)
    failed_cases=len(selected_semesters[selected_semesters['Outcome']==True])
    risk=failed_cases/total_cases
    return risk,total_cases

def get_forest_risk_and_uncertainty(Insulin,SkinThickness,BloodPressure,Glucose,Pregnancies, BMI,DiabetesPedigreeFunction, Age):
    df = pd.DataFrame([[Insulin, SkinThickness, BloodPressure,Glucose,Pregnancies, BMI,DiabetesPedigreeFunction, Age]], columns=['Insulin','SkinThickness','BloodPressure','Glucose','Pregnancies', 'BMI','DiabetesPedigreeFunction', 'Age'])
    prediction=rf_model.predict(df)[0]
    risk=0
    if (prediction):
        risk=1
    certainty=0.7355
    return risk,certainty
    
opt_st=[]
for student in students_test['ID'].values:
    opt_st.append({'label': student, 'value': student})

navbar = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand("Predictor Dashboard", className="ml-2")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

body = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Select Patient"),
                        dcc.Dropdown(
                           id='student',
                           options=opt_st,
                           value=opt_st[0]['value'],
                           ),
                        html.Br(),
                        html.H2("Select Model"),
                        dcc.Dropdown(
                           id='model',
                           options=[{'label': 'Cluster', 'value': 1},{'label': 'Random Forest', 'value': 2}],
                           value=1,
                           ),
                       
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        html.H2("Patient Data"),
                        html.H3("Glucose"),
                        daq.GraduatedBar(id='Glucose',
                                         color={"gradient":True,"ranges":{"red":[0,6],"yellow":[6,8],"green":[8,10]}},
                                         showCurrentValue=True,
                                         value=10
                                         ),
                        html.Br(),
                        html.H3("Factors"),
                        dcc.Graph(id='student_graph'),
                        html.H2("Lifestyle Data"),
                        dbc.Container([dbc.Row([
                                dbc.Col([
                                        daq.LEDDisplay(id="Pregnancies",
                                                       label="Pregnancies",
                                                       value="0",
                                                       size=64,
                                                       color="#FF5E5E"
                                                       ),
                                          ]),
                                dbc.Col([
                                        daq.LEDDisplay(id="BMI",
                                                       label="BMI",
                                                       value="0",
                                                       size=64,
                                                       color="#FF5E5E"
                                                       ),
                                          ]),
                                dbc.Col([
                                        daq.LEDDisplay(id="DiabetesPedigreeFunction",
                                                       label="DiabetesPedigreeFunction",
                                                       value="0",
                                                       size=64,
                                                       color="#FF5E5E"
                                                       ),
                                          ]),
                                dbc.Col([
                                        daq.LEDDisplay(id="Age",
                                                       label="Age",
                                                       value="0",
                                                       size=64,
                                                       color="#FF5E5E"
                                                       ),
                                           ]),
                                ])]),
                        html.Br(),
                        html.H2("Prediction"),
                        dbc.Container([dbc.Row([
                                dbc.Col([
                                        daq.Gauge(id='risk-gauge',
                                                  showCurrentValue=True,
                                                  color={"gradient":True,"ranges":{"red":[0,0.4],"yellow":[0.4,0.7],"green":[0.7,1]}},
                                                  label="Risk",
                                                  max=1,
                                                  min=0,
                                                  value=1
                                                  ),
                                        ]),
                                dbc.Col([
                                        daq.Gauge(id='certainty-gauge',
                                                  showCurrentValue=True,
                                                  color={"gradient":True,"ranges":{"red":[0,200],"yellow":[200,500],"green":[500,1000]}},
                                                  label="Certainty",
                                                  max=1000,
                                                  min=0,
                                                  value=1
                                                  ),
                                        ]),
                             ])]),
                    ],
                ),
            ]
        )
    ],
    className="mt-4",
)

app.layout = html.Div(children=[navbar,body]
)


@app.callback(
    [Output("student_graph", "figure"),
     Output("Glucose", "value"),
     Output("Pregnancies", "value"),
     Output("BMI", "value"),
     Output("DiabetesPedigreeFunction", "value"),
     Output("Age", "value"),
     Output("risk-gauge", "value"),
     Output("certainty-gauge", "value")],
    [Input("student", "value"),
     Input("model", "value")]
)
def update_plots(student_value,model_value):
    Insulin, SkinThickness, BloodPressure, Glucose = get_student_data(student_value)
    Pregnancies, BMI, DiabetesPedigreeFunction, Age=get_semester_data(student_value)
    
    if(model_value==1):
        risk,certainty=get_new_risk_and_uncertainty(Insulin,SkinThickness,BloodPressure,Glucose,Pregnancies, BMI,DiabetesPedigreeFunction, Age)
    else:
        risk,certainty=get_forest_risk_and_uncertainty(Insulin,SkinThickness,BloodPressure,Glucose,Pregnancies, BMI,DiabetesPedigreeFunction, Age)
        
    data_semester = [
        {
            "x": ['Insulin','SkinThickness','BloodPressure','Glucose'],
            "y": [Insulin, SkinThickness, BloodPressure, Glucose],
            #"y": [10,5,8,4,3,2],
            "text": ['Insulin','SkinThickness','BloodPressure','Glucose'],
            "type": "bar",
            "name": student_value,
        }
    ]
    layout_semester = {
        "autosize": True,
        "xaxis": {"showticklabels": True},
    }
    
    data_student = [{
            "type": 'scatterpolar',
            "r": [Insulin, SkinThickness, BloodPressure, Glucose, Insulin],
            "theta": ['Insulin','SkinThickness','BloodPressure','Glucose', 'Insulin'],
            "fill": 'toself'
            }]

    layout_student = {
            "polar": {
                    "radialaxis": {
                            "visible": True,
                            "range": [0, 10]
                            }
            },
            "showlegend": False
            }

    return {"data": data_student, "layout": layout_student},Glucose,Pregnancies,BMI,DiabetesPedigreeFunction, Age, risk, certainty


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)