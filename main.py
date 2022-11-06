import streamlit as st
import functions as fn

st.set_page_config(layout="wide")
with open('main.css') as fileStyle:
    st.markdown(f'<style>{fileStyle.read()}</style>', unsafe_allow_html=True)

#------------------------------------------------------------------------------------- COLUMNS
column1,column2,column3=st.columns([1,3,3])

# ------------------------------------------------------------------------------------ Uploaded File Browsing Button
uploaded_file = column1.file_uploader(label="Uploading Signal", type = ['csv',".wav"])

#------------------------------------------------------------------------------------- USER OPTIONS
select_mode = column1.selectbox("",[ "Uniform Range Mode","Music", "Vowels", "Arrhythima", "Optional","Spectro"])

# ------------------------------------------------------------------------------------ Calling Main Functions
if uploaded_file is not None:
    file_name = uploaded_file.type
    file_extension = file_name[-3:]
    if select_mode == "Uniform Range Mode":
        if file_extension == "wav":
            fn.uniform_range_mode(column1, column2, column3,uploaded_file)

    elif select_mode == "Music":
        pass
    elif select_mode == "Vowels":
        pass

    elif select_mode == "Arrhythima":
        pass

else:
    pass

