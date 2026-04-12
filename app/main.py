import streamlit as st
import pandas as pd
import json
from datetime import datetime
import sys
import os
import pydeck as pdk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.lstm import ReadyLSTM

# cache the model so it is not constantly reloading it every time
@st.cache_resource
def load_model():
    return ReadyLSTM()

map_container = st.empty()

lstm_probability_model = load_model()
with open('Datasets/stops_on_route.json', "r") as f:
    stops_on_route_dict = json.load(f)
stops_info_df = pd.read_csv("Datasets/stops.txt")

def display_route(stop_ids):
    stop_ids['stop_id'] = stop_ids['stop_id'].astype(str)
    stops_lat_lon_df = stop_ids.merge(
    stops_info_df[["stop_id", "stop_lat", "stop_lon"]],
        on="stop_id",
        how="left"
    )
    locs = stops_lat_lon_df[["stop_lon", "stop_lat"]].dropna().values.tolist()    
    print(locs)
    layer = pdk.Layer(
        "PathLayer",
        data=[{"path": locs}],
        get_path="path",
        get_width=5,
        get_color=[255, 0, 0],
        width_min_pixels=4,
    )

  
    view_state = pdk.ViewState(
        latitude=sum(stops_lat_lon_df['stop_lat']) / len(stops_lat_lon_df['stop_lat']),
        longitude=sum(stops_lat_lon_df['stop_lon']) / len(stops_lat_lon_df['stop_lon']),
        zoom=11,
    )

    map_container.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view_state
        )
    )
    

st.title("Where are you going today?")

route = st.selectbox(
    "Line",
    ("Orange", "Red", "Blue"),
)

stops = stops_on_route_dict[route]
arr_stop = st.selectbox(
    "Starting Point Stop",
    stops,
)
dest_stop = st.selectbox(
    "Destination Stop",
    stops,
)

time_option = st.radio(
    "When:",
    ("Right now", "Future")
)

if time_option == "Right now":
    selected_datetime = datetime.now()
    st.write("Right now:", selected_datetime)

else:
    date = st.date_input("Select date")
    time = st.time_input("Select time")
    
    selected_datetime = datetime.combine(date, time)
    st.write("Selected datetime:", selected_datetime)

if st.button("Predict Delay Likelihood", type="primary"):
    input = [route, arr_stop, dest_stop, selected_datetime]

    # predicting probability of delay with lstm
    trip_sq_df, prediction = lstm_probability_model.predict(input)
    stop_ids = trip_sq_df[['stop_id']]
    display_route(stop_ids)
    st.write(prediction)
    
    # predicting probability of delay with hmm
    
if st.button("Predict Delay Duration", type="primary"):
    pass


