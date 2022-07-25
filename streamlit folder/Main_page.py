
from msilib.schema import CheckBox
import streamlit as st
from nbformat import write
import pandas as pd
import numpy as np

st.title('Netflix Movies Recommendation Engine')


get_data = (
    'https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies')


st.subheader('About The Project')

st.write("A Microsoft Capstone Project was given to test knowledge of the basics of Machine Learning and Data Science which was taught over a 30 Days Period.")

st.write("This is a Netflix Movie recommender Algorithm, a very simple algorithm that recommends movie based on some particular pattern and similarity.")

st.write("This system is built using Content filtering which is based solely on the item’s description and profile of the user’s interests, and recommends items based on the user’s past interests.")

st.write(f"Data Soure {get_data}")

st.write("The engine recommends 10 movies based on your input")

st.write("To use this Engine, click the recommendation page")
