import streamlit as st
import pandas as pd
import functions as fn

st.set_page_config(layout="wide")

# ------------------------------------------------------------------------------------ Uploaded File Browsing Button
uploaded_file = st.file_uploader(label="Uploading Signal", type = ['csv',".wav"])

# ------------------------------------------------------------------------------------ Calling Main Functions
if uploaded_file is not None:
    # Determining whether the file is csv or wav
    file_name = uploaded_file.type
    file_extension = file_name[-3:]

    # USER OPTIONS
    radio_button = st.radio("",["Uniform Range Mode", "Music", "Vowels", "Arrhythima", "Optional"])

    if radio_button == "Uniform Range Mode":
        fn.uniform_range_mode(uploaded_file)

    elif radio_button == "Music":
        pass
    elif radio_button == "Vowels":
        pass

    elif radio_button == "Arrhythima":
        if file_extension == "csv":
            pass
    else:
        pass
else:
    pass

