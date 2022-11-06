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

class variabls:
    
    points_num=1000

#  ----------------------------------- TRIAL FOURIER TRANSFORM FUNCTION ---------------------------------------------------
#-------------------------------------- Fourier Transform on Audio ----------------------------------------------------
def uniform_range_mode(audio_file):
    st.audio(audio_file, format='audio/wav') # displaying the audio
    obj = wave.open(audio_file, 'rb')
    sample_rate = obj.getframerate()      # number of samples per second
    n_samples   = obj.getnframes()        # total number of samples in the whole audio
    duration    = n_samples / sample_rate # duration of the audio file
    signal_wave = obj.readframes(-1)      # amplitude of the sound

    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int16)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))

    yf = rfft(signal_y_axis) # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform

    st.write(sample_rate)
    points_per_freq = len(xf) / (xf[-1]) # NOT UNDERSTANDABLE 
    
    fig, axs = plt.subplots()
    fig.set_size_inches(14,5)
    plt.plot(signal_x_axis, signal_y_axis) #plotting fourier
    st.plotly_chart(fig)

    slider_range_1 = st.slider(label='Slider_1' , min_value=0, max_value=10, value=1, step=1, key="1")
    slider_range_2 = st.slider(label='Slider_2' , min_value=0, max_value=10, value=1, step=1, key="2")
    slider_range_3 = st.slider(label='Slider_3' , min_value=0, max_value=10, value=1, step=1, key="3")
    slider_range_4 = st.slider(label='Slider_4' , min_value=0, max_value=10, value=1, step=1, key="4")
    slider_range_5 = st.slider(label='Slider_5' , min_value=0, max_value=10, value=1, step=1, key="5")
    slider_range_6 = st.slider(label='Slider_6' , min_value=0, max_value=10, value=1, step=1, key="6")
    slider_range_7 = st.slider(label='Slider_7' , min_value=0, max_value=10, value=1, step=1, key="7")
    slider_range_8 = st.slider(label='Slider_8' , min_value=0, max_value=10, value=1, step=1, key="8")
    slider_range_9 = st.slider(label='Slider_9' , min_value=0, max_value=10, value=1, step=1, key="9")
    slider_range_10= st.slider(label='Slider_10', min_value=0, max_value=10, value=1, step=1, key="10")





    yf[int(points_per_freq*0)   :int(points_per_freq* 1000)] *= slider_range_1
    yf[int(points_per_freq*1000):int(points_per_freq* 2000)] *= slider_range_2
    yf[int(points_per_freq*2000):int(points_per_freq*3000)]  *= slider_range_3
    yf[int(points_per_freq*3000):int(points_per_freq*4000)]  *= slider_range_4
    yf[int(points_per_freq*4000):int(points_per_freq*5000)]  *= slider_range_5
    yf[int(points_per_freq*5000):int(points_per_freq*6000)]  *= slider_range_6
    yf[int(points_per_freq*6000):int(points_per_freq*7000)]  *= slider_range_7
    yf[int(points_per_freq*7000):int(points_per_freq*8000)]  *= slider_range_8
    yf[int(points_per_freq*8000):int(points_per_freq*9000)]  *= slider_range_9
    yf[int(points_per_freq*9000):int(points_per_freq*10000)] *= slider_range_10
    # fig2, axs2 = plt.subplots()
    # fig2.set_size_inches(14,5)
    # plt.plot(xf,np.abs(yf)) # ploting signal after modifying
    # st.plotly_chart(fig2,use_container_width=True)




    modified_signal         = irfft(yf)                 # returns the inverse transform after modifying it with sliders 
    modified_signal_channel = np.int16(modified_signal) # returns two channels 


    write   ("Equalized_Music.wav", sample_rate, modified_signal_channel) # creates the modified song
    st.audio("Equalized_Music.wav", format='audio/wav')

def ECG_mode(uploaded_file):
    # ECG Sliders  
    bracardia   =st.slider('bracardia :beats per minute '   , step=1, max_value=60 , min_value=0   ,value=80)
    Tachycardia =st.slider('Tachycardia :beats per minute'  , step=1, max_value=170, min_value=100 ,value=80)
    Arrhythmia  =st.slider('Arrhythmia '                    , step=1, max_value=60 , min_value=30  ,value=0 )
    # Reading uploaded_file
    df = pd.read_csv(uploaded_file)
    uploaded_xaxis=df['time']
    uploaded_yaxis=df['amp']
    # Slicing big data
    if (len(uploaded_xaxis)>variabls.points_num):
        uploaded_xaxis=uploaded_xaxis[:variabls.points_num]
    if (len(uploaded_yaxis)>variabls.points_num):
        uploaded_yaxis=uploaded_yaxis[:variabls.points_num]
    #Plotting
    uploaded_fig,uploaded_ax = plt.subplots()
    uploaded_ax.set_title('The Actual Data')
    uploaded_ax.plot(uploaded_xaxis,uploaded_yaxis)
    uploaded_ax.set_xlabel('Time')
    uploaded_ax.set_ylabel('Amplitude')
    st.plotly_chart(uploaded_fig)

    uploaded_xaxis= Tachycardia_barcardia (Tachycardia,bracardia,uploaded_xaxis)

    # fourier transorm
    y_fourier = np.fft.fft(uploaded_yaxis)
    x_fourier = fftfreq(len(uploaded_xaxis),(uploaded_xaxis[5]-uploaded_xaxis[0])/5) 
    abs_y_fourier=np.abs(y_fourier) 

    fourier_fig,fourier_ax = plt.subplots()
    fourier_ax.set_title('fourier')
    fourier_ax.plot(x_fourier,abs_y_fourier)
    fourier_ax.set_xlabel('Time')
    fourier_ax.set_ylabel('Amplitude')
    st.plotly_chart(fourier_fig)


    y_inverse_fourier = np.fft.ifft(y_fourier)

    # Plotting
    inverse_fig,inverse_ax = plt.subplots()
    inverse_ax.set_title('fourier')
    inverse_ax.plot(uploaded_xaxis,y_inverse_fourier)
    inverse_ax.set_xlabel('Time')
    inverse_ax.set_ylabel('Amplitude')
    st.plotly_chart(inverse_fig)

def Tachycardia_barcardia (Tachycardia,bracardia,uploaded_xaxis):
    scaling_factor=((Tachycardia+bracardia)/2-80)/80
    new_x=uploaded_xaxis
    if (scaling_factor>0):
        new_x = [item * scaling_factor for item in uploaded_xaxis]
    if (scaling_factor<0):
        new_x = [item * 1/scaling_factor for item in uploaded_xaxis]

    return new_x

    


