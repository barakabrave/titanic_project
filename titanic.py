import pandas as pd
import numpy as np
import pickle
import streamlit as smt
from PIL import Image
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import datasets
  
# loading in the model to predict on the data
smt.set_page_config(layout="wide")
smt.markdown(
    """
    <style>
    .reportview-container {
        background: url("background.jpeg")
    }
   .sidebar .sidebar-content {
        background: url("background.jpeg")
    }
    </style>
    """,
    unsafe_allow_html=True
)
pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)
#model=pd.read_csv("C:\Users\BRAVE BARAKA\Breast cancer prediction\data.csv")
#smt.title("Titanic Prediction Model")
html_temp = """
    <div style ="background-color:pink ;padding:10px">
    <h1 style ="color:black;text-align:center;">Titanic Prediction ML App</h1>
    </div>
    """

      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
smt.markdown(html_temp, unsafe_allow_html = True)
# import the required modules

import pydeck as pdk
import plotly.express as px
  
# Dataset we need to import
#DATA_URL = (
#    "train.csv"
#)
  
# Add title and subtitle of the map.
smt.title("TITANIC PREDICTION MODEL")
smt.markdown("This app analyzes and displays people who succumbed to Titanic accident")
  
"""
Here, we define load_data function,
to prevent loading the data everytime we made some changes in the dataset.
We use streamlit's cache notation.
"""
smt.write("This application will be used to predict the number of people succembed to Titanic accident")
#@smt.cache(persist = True)

data=pd.read_csv("virtual_env/titanic/train.csv")
smt.title('Welcome all')
#smt.write("Let's view our dataset first")
#smt.dataframe(data=data)
smt.write("The original dataset is as shown below:")
smt.write(data)
data2=data
smt.write("Now lets do some analysis and data preparation")
smt.write("We will start by removing unneccessary variables")
data.drop("PassengerId",axis=1,inplace=True)
data.drop("Ticket",axis=1,inplace=True)
data.drop("Cabin",axis=1,inplace=True)
data.drop("Name",axis=1,inplace=True)
smt.subheader("Below is the editted data:")
smt.write(data)
smt.write("Since the 'Age' Variable has some missing values, we will fill the missing values with '0' then display the columns")
#data.fillna("0",inplace=True)
data.Age=data.Age.replace(np.NAN,data.Age.mean())
smt.write(data.isnull().sum())
smt.write("Finally we will Encode the Sex and the Embarked variable as shown below:")
from sklearn.preprocessing import LabelEncoder
lbl=LabelEncoder()
data.Sex=lbl.fit_transform(data.Sex).astype(int)
data.Embarked=lbl.fit_transform(data.Embarked).astype(int)
smt.write(data.Sex,data.Embarked)
smt.subheader("Below is the fully data to be used for this model development")
smt.write(data)
# Plot : 1
# plot a streamlit map for accident locations.
#st.header("Where are the most people casualties in accidents in UK?")
# plot the slider that selects number of person died
#casualties = st.slider("Number of persons died", 1, int(data["number_of_casualties"].max()))
#st.map(data.query("number_of_casualties >= @casualties")[["latitude", "longitude"]].dropna(how ="any"))
  
# Plot : 2
# plot a pydeck 3D map for the number of accident's happen between an hour interval
#st.header("How many accidents occur during a given time of day?")
#hour = st.slider("Hour to look at", 0, 23)
#original_data = data
#data = data[data['date / time'].dt.hour == hour]
  
#st.markdown("Vehicle collisions between % i:00 and % i:00" % (hour, (hour + 1) % 24))
#midpoint = (np.average(data["latitude"]), np.average(data["longitude"]))
  
#st.write(pdk.Deck(
#    map_style ="mapbox://styles / mapbox / light-v9",
#    initial_view_state ={
#        "latitude": midpoint[0],
#        "longitude": midpoint[1],
#       "zoom": 11,
#        "pitch": 50,
#    },
#    layers =[
#        pdk.Layer(
#        "HexagonLayer",
#        data = data[['date / time', 'latitude', 'longitude']],
#        get_position =["longitude", "latitude"],
#        auto_highlight = True,
#        radius = 100,
#        extruded = True,
#        pickable = True,
#        elevation_scale = 4,
#        elevation_range =[0, 1000],
#        ),
#    ],
#))
  
# Plot : 3
# plot a histogram for minute of the hour atwhich accident happen
smt.subheader("Below is abit of data visualisation of the data. We will analyse the relationships between the major  independent variables to the independent variable.")
smt.write("Here I choose the Age,Fare and Sex variables as my independent variables for analysis and survived variable as the dependent variable")
filtered = data["Age"]
#    (data['date / time'].dt.hour >= hour) & (data['date / time'].dt.hour < (hour + 1))
#]
hist = np.histogram(filtered, bins = 60, range =(0, 60))[0]
#chart_data = pd.DataFrame({"Age": range(60) "Survived": hist})
fig1 = px.bar(data, x ='Age', y ='Survived', hover_data =['Age', 'Survived'])
fig2=px.bar(data, x ='Fare', y ='Survived', hover_data =['Age', 'Survived'])
fig3=px.bar(data, x ='Sex', y ='Survived', hover_data =['Age', 'Survived'])
smt.write(fig1,fig2,fig3)
fig=plt.figure()
x=data["Fare"]
y=data["Survived"]
#plt.scatter(x,y,color="blue")
#smt.pyplot(fig)
smt.subheader("Now lets go forward and predict if someone survived based on the user inputs")

def prediction(P_Class,Sex,Fare):  
   
    prediction=model.predict([[P_Class,Sex,Fare]])
    print(prediction)
    return prediction
def main():
    smt.sidebar.header("Choose your inputs")
    P_Class=smt.sidebar.slider("P_Class",1,3)
    smt.sidebar.write("0=female , 1=male")
    Sex=smt.sidebar.slider("Sex",0,1)
    
#    Age=smt.sidebar.slider("Age",0,200)
 #   SibSp=smt.sidebar.slider("SibSp",0,3)
  #  Parch=smt.sidebar.slider("Parch",0,2)
    Fare=smt.sidebar.slider("Fare",0,1000)
#    Embarked=smt.sidebar.slider("Embarked",0,3)
#    smt.write.sidebar("3=S, 2=O, 1=C")
#    Daily_People_Vaccinated =smt.sidebar.number_input("Daily_People_Vaccinated")
    
    
    user_data={'P_Class':P_Class, 'Sex':Sex,'Fare':Fare}
    features=pd.DataFrame(user_data,index=[0])
    result =""
    
    
    smt.write("## Your Chosen weightings: ")
    smt.write(features)
    
    smt.write("\n\n\n ### THE MODEL PREDICTS: ")
    
    prediction = model.predict(features)[0]
    smt.text(f"The person will {prediction}")
    smt.write("1=Survived ; 0=Not survived")
     
    
if __name__=='__main__':
    main()

# The code below uses checkbox to show raw data
#st.header("Condition of Road at the time of Accidents")
#select = st.selectbox('Weather ', ['Dry', 'Wet / Damp', 'Frost / ice', 'Snow', 'Flood (Over 3cm of water)'])
  
#if select == 'Dry':
#    st.write(original_data[original_data['road_surface_conditions']=="Dry"][["weather_conditions", "light_conditions", "speed_limit", "number_of_casualties"]].sort_values(by =['number_of_casualties'], ascending = False).dropna(how ="any"))
  
#elif select == 'Wet / Damp':
#    st.write(original_data[original_data['road_surface_conditions']=="Wet / Damp"][["weather_conditions", "light_conditions", "speed_limit", "number_of_casualties"]].sort_values(by =['number_of_casualties'], ascending = False).dropna(how ="any"))
#elif select == 'Frost / ice':
#    st.write(original_data[original_data['road_surface_conditions']=="Frost / ice"][["weather_conditions", "light_conditions", "speed_limit", "number_of_casualties"]].sort_values(by =['number_of_casualties'], ascending = False).dropna(how ="any"))
  
#elif select == 'Snow':
#    st.write(original_data[original_data['road_surface_conditions']=="Snow"][["weather_conditions", "light_conditions", "speed_limit", "number_of_casualties"]].sort_values(by =['number_of_casualties'], ascending = False).dropna(how ="any"))
  
#else:
#    st.write(original_data[original_data['road_surface_conditions']=="Flood (Over 3cm of water)"][["weather_conditions", "light_conditions", "speed_limit", "number_of_casualties"]].sort_values(by =['number_of_casualties'], ascending = False).dropna(how ="any"))
  
  
#if st.checkbox("Show Raw Data", False):
#    st.subheader('Raw Data')
#    st.write(data)
