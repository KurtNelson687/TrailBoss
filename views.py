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
import plotly.graph_objs as go
import plotly.plotly as py

global lotCounter, mtbMAPdf, targetAltitude, targetDistance, clusterTried
matplotlib.use('Agg')
import matplotlib.pyplot as plt



filenameMTBdf = 'mtbDf.sav' #name of pickle containing datafram holding rides 
mtbMAPdf = pickle.load(open(filenameMTBdf, 'rb')) #mtb dataframe
lotCounter = 5

@app.route('/')
def ride_input():
    """
    Home page that shows input boxes for desired elevation and altitude
    """
    global lotCounter, clusterTried
    lotCounter = 5 #Counter for number of parking lots remaining
    clusterTried = [] #list storing ride parking lots already tried
    return render_template("input.html")

@app.route('/error')
def error_page():
    """
    Home page that shows input boxes for desired elevation and altitude
    """
    global lotCounter, clusterTried
    lotCounter = 5 #Counter for number of parking lots remaining
    clusterTried = [] #list storing ride parking lots already tried
    return render_template("error.html")

@app.route('/output')
def first_ride():
    """
    First output from homepage. Shows interactive map from found trail, and 
    elevation profile, and botton for showing additional rides
    """
    # Retrieve desired altitude and distance and check if they are empty
    pullElevation = request.args.get('altitude') #pull desired altitude
    pullDistance = request.args.get('distance') #pull desired distance
    if pullElevation == "" or pullDistance == "":
        return render_template("error.html")
    
    global lotCounter, targetAltitude, targetDistance, mtbMAPdf, clusterTried
    lotCounter = lotCounter-1 # Counter for number of parking lots remaining
    
    # Turn into floats
    targetAltitude = float(pullElevation)
    targetDistance = float(pullDistance) 
    
    # Setup train data
    X_train, y_train = getTrain(mtbMAPdf)
    y_train = y_train.drop('centriodLabel', 1)
    X_train = X_train.drop('centriodLabel', 1)

    # Find cloest route and extract distance and elevation
    prediction, matchedDis, matchedAlt = findRoute(X_train,y_train,targetDistance,targetAltitude)

    # Add to the clusters tried
    clusterTried.append(mtbMAPdf[mtbMAPdf.id ==
                                 prediction[0][0]]['centriodLabel'].item())
    
    # Latitudes and longitudes of ride
    myLat = mtbMAPdf.loc[mtbMAPdf.id == prediction[0][0],'lat'].item()
    myLon = mtbMAPdf.loc[mtbMAPdf.id == prediction[0][0],'long'].item()

    token = config.api_key #mapbox token 
    #create dictionary for plotly maps and store as JSON
    graphJSON = makeGraphJSON(myLat,myLon,token)
    elevationJSON = myElevationfig(mtbMAPdf,prediction) #plot elevation profile and save
    return render_template("output.html", foundDistance = ('%0.1f'%matchedDis),
                           foundAltitude = ('%0.0f'%matchedAlt),graphJSON=graphJSON,
                           remainLots=lotCounter, elevationJSON = elevationJSON)

@app.route('/different_lot')
def substitute_ride():
    """
    Page showing rides other than the first one suggested. Looks very similar
    to the inital output page, but has a different background.
    """
    global lotCounter, targetAltitude, targetDistance, mtbMAPdf, clusterTried
    # Counter to keep track of suggested lots
    lotCounter = lotCounter-1
    
    # Assign train data for knn
    X_train, y_train = getTrain(mtbMAPdf)
  
    # Drop train values that are from cluster already suggested
    for clus in clusterTried:
        y_train = y_train.loc[y_train['centriodLabel']!=clus]
        X_train = X_train.loc[X_train['centriodLabel']!=clus]
    y_train = y_train.drop('centriodLabel', 1)
    X_train = X_train.drop('centriodLabel', 1)

    # Find cloest route and extract distance and elevation
    prediction, matchedDis, matchedAlt = findRoute(X_train,y_train,targetDistance,targetAltitude)

    #Add to the clusters tried
    clusterTried.append(mtbMAPdf[mtbMAPdf.id ==
                                 prediction[0][0]]['centriodLabel'].item())

    # Latitudes and longitudes of ride
    myLat = mtbMAPdf.loc[mtbMAPdf.id == prediction[0][0],'lat'].item()
    myLon = mtbMAPdf.loc[mtbMAPdf.id == prediction[0][0],'long'].item()

    token = config.api_key #mapbox token 
    #create dictionary for plotly maps and store as JSON
    graphJSON = makeGraphJSON(myLat,myLon,token)
    elevationJSON = myElevationfig(mtbMAPdf,prediction) #plot elevation profile and save
    return render_template("different_lot.html", foundDistance = ('%0.1f'%matchedDis),
                           foundAltitude = ('%0.0f'%matchedAlt),graphJSON=graphJSON,
                           remainLots=lotCounter,elevationJSON = elevationJSON)

"""def myElevationfig(mtbMAPdf,prediction):
   
    Creates elevation profile map
    
    indRoute = mtbMAPdf.index[mtbMAPdf.id == prediction[0][0]].tolist()[0]
    plt.plot(mtbMAPdf.distance[indRoute],mtbMAPdf.altitude[indRoute],color = 'darkgreen')
    plt.xlabel('distance (m)')
    plt.ylabel('elevation (ft)')
    fig = plt.gcf()
    fig.patch.set_alpha(0.7)
    ax = plt.gca()
    ax.fill_between(mtbMAPdf.distance[indRoute], mtbMAPdf.altitude[indRoute],0,color = 'darkgreen')
    png_output = BytesIO()
    plt.savefig(png_output)
    png_output.seek(0)  # rewind to beginning of file
    figdata_png = base64.b64encode(png_output.getvalue()).decode('utf8')
    plt.clf()
    return figdata_png"""

@app.route('/About', methods=['GET', 'POST'])
def About():
    return render_template('About.html')

def myElevationfig(mtbMAPdf,prediction):
    """
    Creates elevation profile map
    """
    indRoute = mtbMAPdf.index[mtbMAPdf.id == prediction[0][0]].tolist()[0]
    
    # Get distance and elevation
    allDista = mtbMAPdf.distance[indRoute]
    allElevation = mtbMAPdf.altitude[indRoute]

    elevgraphs = [dict(
        data = Data([
            (go.Scatter(
                x=allDista,
                y=allElevation,
                fill='tozeroy',
                mode = 'none'
            )),
                ]),
        layout = go.Layout(
            title='Elevation Profile',
            xaxis=dict(
                title='Distance',
                titlefont=dict(
                    #family='Courier New, monospace',
                    size=18,
                    color='black'
                )
            ),
            yaxis=dict(
                title='Elevation',
                titlefont=dict(
                    #family='Courier New, monospace',
                    size=18,
                    color='black'
                )
            )
        )
        )]
    return json.dumps(elevgraphs, cls=plotly.utils.PlotlyJSONEncoder)

def makeGraphJSON(myLat,myLon,token):
    """
    Creates dictionary for interactive plotly map and saves it as
    a JSON file. 
    """
    latCenter = np.mean(myLat)
    longCenter = np.mean(myLon)
    graphs = [dict(
            data = Data([

                     (Scattermapbox(
                        mode='lines',   
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
                    center=dict(
                        lat=latCenter,
                        lon=longCenter
                    ),
                    pitch=0,
                    zoom=11,
                    style='outdoors'
                ),
            ))]
    return json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

def getTrain(mtbMAPdf):
    """
    Sets up train data
    """
    x_columns = ['totalDistance','netAltGain','centriodLabel']
    y_column = ['id','centriodLabel']
    X_train =mtbMAPdf[x_columns]
    y_train =mtbMAPdf[y_column]
    return X_train, y_train

def findRoute(X_train,y_train,targetDistance,targetAltitude):
    """
    Finds cloest match to desired ride characteristics in all 
    avilable lots. Returns the route identifer and ride distance and
    elevation.
    """
    #Create scaling object and applly train, then train knn
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    knn = KNeighborsRegressor(n_neighbors=1)
    # Fit the model on the training data.
    knn.fit(X_train, y_train)

    #Setup dataframe for testing
    X_test = pd.DataFrame()

    #Setup dataframe for prediction and make prediction
    X_test['totalDistance'] = [targetDistance] #pull desired distance
    X_test['netAltGain'] = [targetAltitude] #pull desired altitude
    X_test = scaler.transform(X_test) #scale prediction
    route = knn.predict(X_test) #make prediction
    
    #Matched distance
    matchedDis = mtbMAPdf.loc[mtbMAPdf.id ==
                              route[0][0],'totalDistance'].item()
    #Matched altitude
    matchedAlt = mtbMAPdf.loc[mtbMAPdf.id ==
                              route[0][0],'netAltGain'].item()
    return route, matchedDis, matchedAlt
