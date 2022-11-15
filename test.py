import streamlit as st
import functions as fn

uploaded_file1 = st.file_uploader(label="Uploading Signal1", type = ['csv',".wav"])
uploaded_file2 = st.file_uploader(label="Uploading Signal2", type = ['csv',".wav"])


if uploaded_file1 is not None:
    print(uploaded_file1)
if uploaded_file2 is not None:
    print(uploaded_file2)