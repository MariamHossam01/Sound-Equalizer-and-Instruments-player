import streamlit as st
import functions as fn

st.set_page_config(layout="wide")
with open('main.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)
with st.sidebar:
    #------------------------------------------------------------------------------------- COLUMNS
    # ------------------------------------------------------------------------------------ Uploaded File Browsing Button
    uploaded_file = st.file_uploader(label="Uploading Signal", type = ['csv',".wav"])
    #------------------------------------------------------------------------------------- USER OPTIONS
    select_mode = st.selectbox("",[ "Uniform Range Mode","Music", "Vowels", "Arrhythima", "Optional"])
    #------------------------------------------------------------------------------------- Changing Between Plots
    show_spectrogram = st.checkbox("Show Spectrogram")
    # ------------------------------------------------------------------------------------ Calling Main Functions

column1,column2,column3=st.columns([1.5,2,2])
if uploaded_file is not None:
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    file_extension = file_type[-3:]
    if select_mode == "Uniform Range Mode":
        if file_extension == "wav":
            fn.uniform_range_mode(column2, column3, uploaded_file, show_spectrogram,file_name)

    elif select_mode == "Music":
         fn.music_control(column2, column3, uploaded_file, show_spectrogram,file_name)
    elif select_mode == "Vowels":
        if file_extension == "wav":
            fn.vowels_mode(uploaded_file,column2,column3)

    elif select_mode == "Arrhythima":
          if file_extension == "csv":
                fn.ECG_mode(uploaded_file, show_spectrogram)
    elif select_mode == "Optional":
        fn.voice_changer(uploaded_file, column1, column2, column3, show_spectrogram)


else:
    st.write('Please upload your signal and select mode.')

