from email.mime import audio
import streamlit as st
import numpy as np  
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import librosa as lr

import mariam2 as fn



with st.sidebar:
    
    factor1 =st.slider('bracardia :beats per minute ', step=1, max_value=60, min_value=30 )
    factor2 =st.slider('Tachycardia :beats per minute', step=1, max_value=170, min_value=100)
    factor3 =st.slider('Arrhythmia ', step=1, max_value=60, min_value=30 )

    # beat= 250 point
    # time= 10 sec 
    # factor/6 *beat per 10sec
uploaded_file1 =st.file_uploader('upload the signal file',['csv','ogg','wav'] , help='upload your signal file',key='1' )
if (uploaded_file1):

    df = pd.read_csv(uploaded_file1)
    inverseFourier, fourierTransform = fn.fourier_transform(df,factor1)
    fn.fourier_inverse_transform(inverseFourier,df,factor1)

uploaded_file2 =st.file_uploader('upload the signal file',['csv','ogg','wav'] , help='upload your signal file',key='2'  )
if (uploaded_file2):

    df = pd.read_csv(uploaded_file2)
    inverseFourier, fourierTransform = fn.fourier_transform(df,factor2)
    fn.fourier_inverse_transform(inverseFourier,df,factor2)



