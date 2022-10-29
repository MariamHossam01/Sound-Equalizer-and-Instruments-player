import streamlit as st
from email.mime import audio
import numpy as np  
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import librosa as lr
from streamlit_option_menu import option_menu
import functions


st.set_page_config(page_title="Equalizer studio",
                   page_icon="📶", layout="wide",
                   initial_sidebar_state="expanded",
                   )

# # styles from css file
# with open('style.css') as css:
#     st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# Styling of the sliders
ColorMinMax = st.markdown(''' <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {
    background: rgb(1 1 1 / 0%); } </style>''', unsafe_allow_html=True)


Slider_Cursor = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
    background-color: rgb(14, 38, 74); box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;} </style>''', unsafe_allow_html=True)


Slider_Number = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div
                                { color: rgb(14, 38, 74); 
                                  font-weight: bold
                                } </style>''', unsafe_allow_html=True)

# Styling of the buttons
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #191970;
    color:#ffffff;
    border-radius: 60px;
    font-weight: bold
}
div.stButton > button:hover {
    background-color:#B0C4DE;
    color:#191970;
    }
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 300px;
        heigth: 100;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
</style>""", unsafe_allow_html=True)
# ***************************************************************************#
freq_1 = st.sidebar.slider('Choose the first frequency', 0, 100, 0)
st.session_state["frequency_1"] = freq_1
freq_2 = st.sidebar.slider('Choose the second frequency', 0, 100, 0)
st.session_state["frequency_2"] = freq_2
freq_3 = st.sidebar.slider('Choose the third frequency', 0, 100, 0)
st.session_state["frequency_3"] = freq_3
freq_4 = st.sidebar.slider('Choose the fourth frequency', 0, 100, 0)
st.session_state["frequency_4"] = freq_4
freq_5 = st.sidebar.slider('Choose the fifth frequency', 0, 100, 0)
st.session_state["frequency_5"] = freq_5
freq_6 = st.sidebar.slider('Choose the sixth frequency', 0, 100, 0)
st.session_state["frequency_6"] = freq_6
freq_7 = st.sidebar.slider('Choose the seventh frequency', 0, 100, 0)
st.session_state["frequency_7"] = freq_7
freq_8 = st.sidebar.slider('Choose the eigth frequency', 0, 100, 0)
st.session_state["frequency_8"] = freq_8
freq_9 = st.sidebar.slider('Choose the ningth frequency', 0, 100, 0)
st.session_state["frequency_9"] = freq_9
freq_10 = st.sidebar.slider('Choose the tenth frequency', 0, 100, 0)
st.session_state["frequency_10"] = freq_10

Signal_type = st.selectbox(
    'what is the required type of the signal?',
    ('audio','Alphapetic', 'musical', 'Ecg' ,'else'))


Uploaded_fig= px.density_heatmap(data_frame=[{}])

uploaded_file =st.file_uploader('upload the signal file',['csv','ogg','wav'] , help='upload your signal file' )
if (uploaded_file):

    st.audio(uploaded_file, format='audio/wav',start_time=10)
    # analyising uploaded audio --> getting x ,y
    audio_amp,sampFreq= lr.load(uploaded_file)
    time_axis =np.arange(0,len(audio_amp))/sampFreq
    Uploaded_fig.add_trace(go.Scatter(x=time_axis[:1000], y=audio_amp[:1000],))
    # updating data in functions
    functions.variabls.uploaded_audio_time=time_axis
    functions.variabls.uploaded_audio_time=audio_amp   


# FRONT uii

st.title("Equalizer:")
fig,fig2 = st.columns((2))

with fig:
    st.write('audio signal')
    st.write(Uploaded_fig)
