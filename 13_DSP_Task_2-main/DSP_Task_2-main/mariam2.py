import random 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit_vertical_slider  as svs
from scipy.fftpack import fft
from scipy.fft import fft, fftfreq, fftshift
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.io.wavfile import write
from scipy.signal import find_peaks
import wave
import IPython.display as ipd
import pylab
import os
import librosa
import librosa.display
import time
import altair as alt
import pandas as pd
import plotly.graph_objects as go

def plot_animation(df1,df2):
    
    lines1 = alt.Chart(df1).mark_line().encode(
        x=alt.X('time', axis=alt.Axis(title='date')),
        y=alt.Y('amplitude', axis=alt.Axis(title='value')),
    ).properties(
        width=600,
        height=300
    )
    lines2 = alt.Chart(df2).mark_line().encode(
        x=alt.X('time', axis=alt.Axis(title='date')),
        y=alt.Y('amplitude', axis=alt.Axis(title='value')),
    ).properties(
        width=600,
        height=300
    )
    return lines1,lines2

def Dynamic_graph(signal_x_axis, signal_y_axis, modified_signal, radio_button,
                    column2,column3):
    # start of dynamic plotting

    df1    = pd.DataFrame({'time': signal_x_axis[::1500], 'amplitude': signal_y_axis[:: 1500]}, columns=['time', 'amplitude'])
    lines1 = alt.Chart(df1).mark_line(width=50).encode( x=alt.X('0:T', axis=alt.Axis(title='time')),
                        y=alt.Y('1:Q', axis=alt.Axis(title='value'))).properties(width=600,height=300)

    df2    = pd.DataFrame({'time': signal_x_axis[::1500], 'amplitude': modified_signal[:: 1500]}, columns=['time', 'amplitude'])
    lines2 = alt.Chart(df2).mark_line(width=50).encode( x=alt.X('0:T', axis=alt.Axis(title='time')),
                        y=alt.Y('1:Q', axis=alt.Axis(title='value'))).properties(width=600,height=300)

    N = df1.shape[0]                   # number of elements in the dataframe
    if "size" not in st.session_state:
        st.session_state["size"] = 0   # size of the current dataset # number of elements (months) to add to the plot

    if "start_g" not in st.session_state:
        st.session_state["start_g"]= 0 

    # Plot Animation
    line_plot1 = column2.altair_chart(lines1)
    line_plot2 = column3.altair_chart(lines2)
    # start_btn1 = st.button(label=button_name)
    if radio_button == 'play':
        for i in range(1, N):
            # st.write(st.session_state.start_g)
            st.write(st.session_state.size)
            step_df1 = df1.iloc[st.session_state.start_g:st.session_state.size]
            step_df2 = df2.iloc[st.session_state.start_g:st.session_state.size]
            lines1,lines2 = plot_animation(step_df1,step_df2)


            line_plot1 = line_plot1.altair_chart(lines1)
            line_plot2 = line_plot2.altair_chart(lines2)

            st.session_state.size = i + 6
            if st.session_state.size >= N:
                st.session_state.size = N - 1
            time.sleep(0.000001)
    elif radio_button == 'pause':
        step_df1 = df1.iloc[0:st.session_state.size]  
        step_df2 = df2.iloc[0:st.session_state.size] 
        lines1,lines2 = plot_animation(step_df1,step_df2)
        line_plot1 = line_plot1.altair_chart(lines1)
        line_plot2 = line_plot2.altair_chart(lines2)
        # st.session_state.start_g = st.session_state.size 
        # st.write(st.session_state.start_g)
        # st.session_state.size += 6