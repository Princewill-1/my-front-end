import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

#because streamlit always reruns every you change something you can use st.cache to make a code run just once.
#first we have to define a function for the code

@st.cache
def get_data(filename):
    taxi_data=pd.read_csv(filename)
    return taxi_data
    
with header:
    st.title('Welcome to my awesome project')
    st.text('In this project i looked into the transcations of taxis in NYC')
with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on kaggle.com')
    
    taxi_data = get_data('data/nyc_taxis.csv')
    st.write(taxi_data.head())
    
    st.subheader('Pickup location ID distribution on the NYC dataset')
    pickup_location_distribution = pd.DataFrame(taxi_data['pickup_location_code'].value_counts())
    st.bar_chart(pickup_location_distribution)


with features:
    st.header('The features I created')
    st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic')
    st.markdown('* **second feature:** I created this feature because of this... I calculated it using this logic')


with model_training:
    st.header('Time to train model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance changes!')
    
    sel_col, disp_col = st.columns(2)
    max_depth = sel_col.slider('What is the max_depth of the model', min_value=(10),max_value=(100),value=20,step=10)
    n_estimators = sel_col.selectbox('how many trees', options=[100,200,300,'No limit'],index=0)
    
    sel_col.text('Here is a list of features in my data:') 
    sel_col.write(taxi_data.columns)
    
    input_feature = sel_col.text_input('which feature should be used as the input feature?','pickup_location_code')
    
    #now we start training our models
    #the code in the if statement is done to accomodate the 'No limit' button.
    #if there was no 'No limit' button you dont need the if statement.

    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=(max_depth))
    else:
        regr = RandomForestRegressor(max_depth=(max_depth),n_estimators=n_estimators)
    
    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]
    
    regr.fit(X,y)
    prediction = regr.predict(y)
    #we have finished traning the model
    #next we try and display the MAE in the disp_col
    
    disp_col.subheader('Mean absolute error of this model is:')
    disp_col.write(mean_absolute_error(y, prediction))
    
    
    disp_col.subheader('Mean squared error of this model is:')
    disp_col.write(mean_squared_error(y, prediction))
    
    
    disp_col.subheader('R squared of the model is:')
    disp_col.write(r2_score(y, prediction))
    
    
    
