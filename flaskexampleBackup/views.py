import numpy as np
import pandas as pd
import pickle
import plotly
import mapbox
import geojson
import psycopg2
import config
import plotly.offline as py_off
import base64
import matplotlib

from flask import render_template
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from flask import request
from flaskexample.a_Model import ModelIt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from mapboxgl.viz import *
from mapboxgl.utils import *
from mapboxgl.colors import *
from plotly.graph_objs import *
from plotly.graph_objs import *
from mapbox import Geocoder

global lotCounter, mtbMAPdf, targetAltitude, targetDistance, clusterTried
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#filenameModel = 'knn_model.sav'
#filenameScaling = 'scaling_model.sav'
filenameMTBdf = 'mtbDf.sav'
#knn = pickle.load(open(filenameModel, 'rb')) #knn model
#scaler = pickle.load(open(filenameScaling, 'rb')) #scaling model
mtbMAPdf = pickle.load(open(filenameMTBdf, 'rb')) #mtb dataframe
lotCounter = 5

@app.route('/')
def ride_input():
    global lotCounter, clusterTried
    lotCounter = 5
    clusterTried = []
    return render_template("input.html")

@app.route('/output')
def first_ride():
  global lotCounter, targetAltitude, targetDistance, mtbMAPdf, clusterTried
  lotCounter = lotCounter-1
  x_columns = ['totalDistance','netAltGain']
  y_column = ['id']
  X_train =mtbMAPdf[x_columns]
  y_train =mtbMAPdf[y_column]
    
  #Create scaling object and applly train, then train knn
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  knn = KNeighborsRegressor(n_neighbors=1)
  # Fit the model on the training data.
  knn.fit(X_train, y_train)

  #Setup dataframe for testing
  X_test = pd.DataFrame()
  targetAltitude = float(request.args.get('altitude')) #pull desired altitude
  targetDistance = float(request.args.get('distance')) #pull desired distance

  #Setup dataframe for prediction and make prediction
  X_test['totalDistance'] = [targetDistance] #pull desired distance
  X_test['netAltGain'] = [targetAltitude] #pull desired altitude
  X_test = scaler.transform(X_test) #scale prediction
  prediction = knn.predict(X_test) #make prediction
  
  #Matched distance
  matchedDis = mtbMAPdf.loc[mtbMAPdf.id ==
                            prediction[0][0],'totalDistance'].item()
  #Matched altitude
  matchedAlt = mtbMAPdf.loc[mtbMAPdf.id ==
                            prediction[0][0],'netAltGain'].item()
  #Add to the clusters tried
  clusterTried.append(mtbMAPdf[mtbMAPdf.id ==
                               prediction[0][0]]['centriodLabel'].item())

  myLat = mtbMAPdf.loc[mtbMAPdf.id == prediction[0][0],'lat'].item()
  myLon = mtbMAPdf.loc[mtbMAPdf.id == prediction[0][0],'long'].item()
  latCenter = np.mean(myLat)
  longCenter = np.mean(myLon)
  token = config.api_key
  graphs = [dict(
            data = Data([

                     (Scattermapbox(
                        mode='lines',
                        #lat=results_df.latitude,
                        #lon=results_df.longitude,       
                        lat=myLat,
                        lon=myLon,
                        marker=Marker(color='black',size=20))),

                    (Scattermapbox(
                        mode='markers',
                        lat=[myLat[0]],
                        lon=[myLon[0]],
                        marker=dict(
                            size=15,
                            color='blue',
                            opacity=0.5),
                        text='Start Location',
                        hoverinfo='text'
                    ))]),
            layout = Layout(
                margin=dict(t=0,b=0,r=0,l=0),
                autosize=True,
                hovermode='closest',
                showlegend=False,
                mapbox=dict(
                    accesstoken=token,
                    bearing=0,
                   # center=dict(
                   #     lat=np.mean(pd.to_numeric(results_df.latitude)),
                   #     lon=np.mean(pd.to_numeric(results_df.longitude))
                   # ),
                    center=dict(
                        lat=latCenter,
                        lon=longCenter
                    ),
                    pitch=0,
                    zoom=11,
                    style='outdoors'
                ),
            ))]
  graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)


 ########################################################
  indRoute = mtbMAPdf.index[mtbMAPdf.id == prediction[0][0]].tolist()[0]
  plt.plot(mtbMAPdf.distance[indRoute],mtbMAPdf.altitude[indRoute],color = 'darkgreen')
  plt.xlabel('distance (m)')
  plt.ylabel('elevation (ft)')
  ax = plt.gca()
  ax.fill_between(mtbMAPdf.distance[indRoute], 0, mtbMAPdf.altitude[indRoute],color = 'darkgreen')
  myFig = mysavefig()
#########################################################################
  return render_template("output.html", foundDistance = ('%0.1f'%matchedDis),
                         foundAltitude = ('%0.0f'%matchedAlt),graphJSON=graphJSON,
                         remainLots=lotCounter, plotVar = myFig)

@app.route('/different_lot')
def substitute_ride():
  global lotCounter, targetAltitude, targetDistance, mtbMAPdf, clusterTried
  lotCounter = lotCounter-1
  x_columns = ['totalDistance','netAltGain','centriodLabel']
  y_column = ['id','centriodLabel']
  X_train =mtbMAPdf[x_columns]
  y_train =mtbMAPdf[y_column]
  
  for clus in clusterTried:
    y_train = y_train.loc[y_train['centriodLabel']!=clus]
    X_train = X_train.loc[X_train['centriodLabel']!=clus]

  #Drop centroid label
  y_train = y_train.drop('centriodLabel', 1)
  X_train = X_train.drop('centriodLabel', 1)

  #Create scaling object and applly train, then train knn
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  knn = KNeighborsRegressor(n_neighbors=1)
  # Fit the model on the training data.
  knn.fit(X_train, y_train)

  #Setup dataframe for prediction and make prediction
  X_test = pd.DataFrame()
  X_test['totalDistance'] = [targetDistance] #pull desired distance
  X_test['netAltGain'] = [targetAltitude] #pull desired altitude
  X_test = scaler.transform(X_test) #scale prediction
  prediction = knn.predict(X_test) #make prediction
  
  #Matched distance
  matchedDis = mtbMAPdf.loc[mtbMAPdf.id ==
                            prediction[0][0],'totalDistance'].item()
  #Matched altitude
  matchedAlt = mtbMAPdf.loc[mtbMAPdf.id ==
                            prediction[0][0],'netAltGain'].item()
  #Add to the clusters tried
  clusterTried.append(mtbMAPdf[mtbMAPdf.id ==
                               prediction[0][0]]['centriodLabel'].item())

  myLat = mtbMAPdf.loc[mtbMAPdf.id == prediction[0][0],'lat'].item()
  myLon = mtbMAPdf.loc[mtbMAPdf.id == prediction[0][0],'long'].item()
  latCenter = np.mean(myLat)
  longCenter = np.mean(myLon)
  type(myLat[0])
  print(myLat[0])
  token = config.api_key
  graphs = [dict(
            data = Data([

                     (Scattermapbox(
                        mode='lines',
                        #lat=results_df.latitude,
                        #lon=results_df.longitude,       
                        lat=myLat,
                        lon=myLon,
                        marker=Marker(color='black',size=20))),

                    (Scattermapbox(
                        mode='markers',
                        lat=[myLat[0]],
                        lon=[myLon[0]],
                        marker=dict(
                            size=15,
                            color='blue',
                            opacity=0.5),
                        text='Start Location',
                        hoverinfo='text'
                    ))]),
            layout = Layout(
                margin=dict(t=0,b=0,r=0,l=0),
                autosize=True,
                hovermode='closest',
                showlegend=False,
                mapbox=dict(
                    accesstoken=token,
                    bearing=0,
                   # center=dict(
                   #     lat=np.mean(pd.to_numeric(results_df.latitude)),
                   #     lon=np.mean(pd.to_numeric(results_df.longitude))
                   # ),
                    center=dict(
                        lat=latCenter,
                        lon=longCenter
                    ),
                    pitch=0,
                    zoom=11,
                    style='outdoors'
                ),
            ))]
  graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
  return render_template("different_lot.html", foundDistance = ('%0.1f'%matchedDis),
                         foundAltitude = ('%0.0f'%matchedAlt),graphJSON=graphJSON,
                         remainLots=lotCounter)

def mysavefig():
    png_output = BytesIO()
    plt.savefig(png_output)
    png_output.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(png_output.getvalue()).decode('utf8')
    plt.clf()
    return figdata_png


