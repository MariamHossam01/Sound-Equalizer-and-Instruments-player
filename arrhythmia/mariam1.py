from email.mime import audio
import streamlit as st
import numpy as np  
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import librosa as lr

import mariam2 as fn



# st.download_button(
#     label="Download data as CSV",
#     data=fn.save_signal(),
#     file_name='lol.csv',
#     mime='text/csv',
# )
with st.sidebar:
    
    bracardia =st.slider('bracardia :beats per minute ', step=1, max_value=60, min_value=30 )
    Tachycardia =st.slider('Tachycardia :beats per minute', step=1, max_value=170, min_value=100 ,)
    Arrhythmia =st.slider('Arrhythmia ', step=1, max_value=60, min_value=30 )

    # beat= 250 point
    # time= 10 sec 
    # factor/6 *beat per 10sec
uploaded_file1 =st.file_uploader('upload the signal file',['csv','ogg','wav'] , help='upload your signal file',key='1' )
if (uploaded_file1):

    df = pd.read_csv(uploaded_file1)
    inverseFourier, fourierTransform = fn.fourier_transform(df,bracardia,Tachycardia)
    fn.fourier_inverse_transform(inverseFourier,df)

# uploaded_file2 =st.file_uploader('upload the signal file',['csv','ogg','wav'] , help='upload your signal file',key='2'  )
# if (uploaded_file2):

#     df = pd.read_csv(uploaded_file2)
#     inverseFourier, fourierTransform = fn.fourier_transform(df,factor2)
#     fn.fourier_inverse_transform(inverseFourier,df,factor2)



