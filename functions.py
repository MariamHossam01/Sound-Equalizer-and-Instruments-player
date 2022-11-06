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

class variabls:
    
    points_num=1000
#------------------------------------------- Reading Audio ----------------------------------------------------------- 
def read_audio(audio_file):
    obj = wave.open(audio_file, 'r')
    sample_rate   = obj.getframerate()                           # number of samples per second
    n_samples     = obj.getnframes()                             # total number of samples in the whole audio
    signal_wave   = obj.readframes(-1)                           # amplitude of the sound
    duration      = n_samples / sample_rate                      # duration of the audio file
    sound_info    = pylab.fromstring(signal_wave, 'int16')
    signal_y_axis = np.frombuffer(signal_wave, dtype=np.int16)
    signal_x_axis = np.linspace(0, duration, len(signal_y_axis))
    return signal_x_axis, signal_y_axis, sample_rate,  sound_info

#-------------------------------------- UNIFORM RANGE MODE FUNCTION ----------------------------------------------------
def uniform_range_mode(column2, column3, audio_file, show_spectrogram):
    signal_x_axis, signal_y_axis, sample_rate ,sound_info = read_audio(audio_file)    # Read Audio File


    yf = rfft(signal_y_axis)                                               # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform
    
    points_per_freq = len(xf) / (xf[-1])  # duration

    if (show_spectrogram):
        pylab.specgram(sound_info, Fs=sample_rate)
        column2.pyplot(pylab)
    else:
        plotting_graphs(column2,signal_x_axis,signal_y_axis,False)

    columns=st.columns(10)
    index=0
    list_of_sliders_values = []
    while index < 10:
        with columns[index]:
            sliders = (st.slider(label="",key=index, min_value=0, max_value=10,value=1, step=1))
        index +=1
        list_of_sliders_values.append(sliders)

    for index2,value in enumerate(list_of_sliders_values):
        yf[int(points_per_freq * 1000 * index2)  : int(points_per_freq * 1000 * index2 + points_per_freq * 1000)] *= value
    else:
        pass
    
    modified_signal         = irfft(yf)                 # returns the inverse transform after modifying it with sliders 
    modified_signal_channel = np.int16(modified_signal) # returns two channels
    
    write(".Equalized_Music.wav", sample_rate, modified_signal_channel)   # creates the modified song
    
    if (show_spectrogram):
        # plot_spectrogram(column3,".Equalized_Music.wav")
        pylab.specgram(modified_signal_channel, Fs=sample_rate)
        column3.pyplot(pylab)
    else:
        plotting_graphs(column3,signal_x_axis,modified_signal,False)

    column2.audio  (audio_file, format='audio/wav') # displaying the audio before editing
    column3.audio(".Equalized_Music.wav", format='audio/wav')             # displaying the audio after editing
#-------------------------------------- ARRYTHMIA FUNCTION ----------------------------------------------------
def ECG_mode(uploaded_file):
    # ECG Sliders  
    # bracardia   =st.slider('bracardia :beats per minute '   , step=1, max_value=60 , min_value=0   ,value=80)
    # Tachycardia =st.slider('Tachycardia :beats per minute'  , step=1, max_value=170, min_value=100 ,value=80)
    Arrhythmia  =st.slider('Arrhythmia '                    , step=1, max_value=100 , min_value=0  ,value=0 )
    Arrhythmia/=100
    # Reading uploaded_file
    df = pd.read_csv(uploaded_file)
    uploaded_xaxis=df['time']
    uploaded_yaxis=df['amp']
    # Slicing big data
    if (len(uploaded_xaxis)>variabls.points_num):
        uploaded_xaxis=uploaded_xaxis[:variabls.points_num]
    if (len(uploaded_yaxis)>variabls.points_num):
        uploaded_yaxis=uploaded_yaxis[:variabls.points_num]


    # uploaded_xaxis= Tachycardia_barcardia (Tachycardia,bracardia,uploaded_xaxis)

    # fourier transorm
    y_fourier = np.fft.fft(uploaded_yaxis)
    
    y_fourier=arrhythmia (Arrhythmia,y_fourier)
    # abs_y_fourier=np.abs(y_fourier)
    # x_fourier = fftfreq(len(uploaded_xaxis),(uploaded_xaxis[5]-uploaded_xaxis[0])/5) 
    # length=(len( y_fourier ))//2


    y_inverse_fourier = np.fft.ifft(y_fourier)
    #Plotting
    uploaded_fig,uploaded_ax = plt.subplots()
    uploaded_ax.set_title('The Actual Data')
    uploaded_ax.plot(uploaded_xaxis,uploaded_yaxis)
    uploaded_ax.plot(uploaded_xaxis,y_inverse_fourier)
    uploaded_ax.set_xlabel('Time')
    uploaded_ax.set_ylabel('Amplitude')
    st.plotly_chart(uploaded_fig)


def Tachycardia_barcardia (Tachycardia,bracardia,uploaded_xaxis):
    scaling_factor=((Tachycardia+bracardia)/2-80)/80
    new_x=uploaded_xaxis
    if (scaling_factor>0):
        new_x = [item * scaling_factor for item in uploaded_xaxis]
    if (scaling_factor<0):
        new_x = [item * 1/scaling_factor for item in uploaded_xaxis]

    return new_x

def arrhythmia (arrhythmia,y_fourier):
    new_y=y_fourier
    df = pd.read_csv('arrhythmia_components.csv')
    sub=df['sub']
    abs_sub=df['abs_sub']
    result = [item * arrhythmia for item in abs_sub]
    new_y=np.add(y_fourier,result)

    return new_y

    
#------------------
#------------------------------------------ PLOTTING FUNCTION -------------------------------------------------------
def plotting_graphs(column,x_axis,y_axis,flag):
    fig, axs = plt.subplots()
    fig.set_size_inches(6,3)
    plt.plot(x_axis,y_axis)
    if flag == True:
        pass
        # plt.xlim(45, 55)
        # plt.xlabel("Time in s")
        # plt.ylabel("ECG in mV")     
    column.plotly_chart(fig)
    #--------------------------------------------- PLOTTING SPECTROGRAMS ---------------------------------
# def plot_spectrogram(column,audio_file):
#     signal_x_axis, signal_y_axis, sample_rate,  sound_info = read_audio(audio_file)
#     pylab.figure(num=None, figsize=(19, 12))
#     pylab.subplot(111)
#     # pylab.title('spectrogram of %r' % wav_file)
#     pylab.specgram(sound_info, Fs=sample_rate)
#     pylab.savefig('spectrogram.png')
#     column.pyplot(pylab)
    #----------------------------------------- Reading Audio
def optional_function(column2,column3,audio_file):
    signal_x_axis, signal_y_axis, sample_rate ,sound_info = read_audio(audio_file)    # Read Audio File

    yf = rfft(signal_y_axis)                                               # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform
    
    points_per_freq = len(xf) / (xf[-1]) # duration

    plotting_graphs(column2,signal_x_axis,signal_y_axis,False)

    sub_column1, sub_column2, sub_column3 = st.columns([1,1,1])
    slider_range   = sub_column1.slider(label='Bass Sound'   , min_value=0, max_value=10, value=1, step=1, key="Bass slider")
    
    yf[int(points_per_freq*0)   :int(points_per_freq* 1000)] *= slider_range
    modified_signal         = irfft(yf)                 # returns the inverse transform after modifying it with sliders
    modified_signal_channel = np.int16(modified_signal) # returns two channels 

    plotting_graphs(column2,signal_x_axis,modified_signal,False)

    write   (".Equalized_Music.wav", sample_rate, modified_signal_channel) # creates the modified song
    column2.audio('.piano_timpani_piccolo_out.wav', format='audio/wav')    # displaying the audio before editing
    column3.audio(".Equalized_Music.wav", format='audio/wav')              # displaying the audio after  editing