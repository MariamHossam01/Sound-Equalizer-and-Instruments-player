import streamlit as st
import pandas as pd
import functions as fn
import librosa
from scipy.io import wavfile as wav

st.set_page_config(layout="wide")

# ------------------------------------------------------------------------------------ uploaded_file Browsing Button
uploaded_file = st.file_uploader(label="Uploading Signal", type = ['csv',"wav"])

# ------------------------------------------------------------------------------------Calling Main Functions
if uploaded_file is not None:

    # USER OPTIONS
    radio_button = st.radio("",["Default Signal", "Music", "Vowels", "Arrhythima", "Optional"])

    if radio_button == "Default Signal":
        names_list = [('A', 100),('B', 150),('C', 75),('D', 25),('E', 150),('F', 60),('G', 86),('H', 150),('E', 150),('G', 25),('K', 99),('L', 150),
                        ('M', 150),('M', 55),('N', 150)]
        fn.creating_sliders(names_list)

    elif radio_button == "Music":
        names_list = [('Megzawy', 100),('Magdy', 150)]
        fn.creating_sliders(names_list)

    elif radio_button == "Vowels":
        names_list = [('Amr', 100),('Sameh', 150),]
        fn.creating_sliders(names_list)

    elif radio_button == "Arrhythima":
        names_list = [('Mariam', 100),('Taha', 150),]
        fn.creating_sliders(names_list)
    else:
        names_list = [('Ahmed', 100),('Youssef', 150),]
        fn.creating_sliders(names_list)

    file_name = uploaded_file.type
    file_extension = file_name[-3:]


    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
        inverseFourier, fourierTransform = fn.fourier_transform(df)
        fn.fourier_inverse_transform(inverseFourier,df)
        fn.wave_ranges(fourierTransform)
    else:
        # st.audio(uploaded_file, format='audio/ogg')         # displaying the audio player
        # amplitude, frequency = librosa.load(uploaded_file)  # getting audio attributes which are amplitude and frequency (number of frames per second)
        # fn.plotting(amplitude, frequency) #librosa feha moshkla f fourier bokra inshalah afhmhalko
        st.audio(uploaded_file, format='audio/wav') 
        xf,fft_out = fn.fourier_for_audio(uploaded_file)
        fn.plotting(xf,fft_out)

else:
    pass


