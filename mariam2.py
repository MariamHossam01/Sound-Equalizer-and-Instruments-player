import random 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import streamlit_vertical_slider  as svs
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from scipy.fft import fft, fftfreq, fftshift

from scipy.misc import electrocardiogram

class variabls:
    
    current_slider_values=np.zeros(50)
    labels_values=np.zeros(50)
    ranges_values=np.zeros(50)
    uploaded_time=[]
    uploaded_Yamp=[]
    points_num=1000




#  ----------------------------------- FOURIER TRANSFORM FUNCTION ---------------------------------------------------
#  ----------------------------------- FOURIER TRANSFORM FUNCTION ---------------------------------------------------

def fourier_transform(df,factor):
    # Getting df x_axis and y_axis
    points_num=int(factor/6 *250 )
    list_of_columns = df.columns
    df_x_axis = np.linspace(0,10,points_num)
    # df_x_axis = list(df[list_of_columns[0]])  
    df_y_axis = (df[list_of_columns[0]])
     # Slicing big data

    if (len(df_x_axis)>points_num):
        df_x_axis=df_x_axis[:points_num]
    if (len(df_y_axis)>points_num):
        df_y_axis=df_y_axis[:points_num]

    # Frequency domain representation
    fourier_transform = np.fft.fft(df_y_axis)

    # Do an inverse Fourier transform on the signal
    inverse_fourier = np.fft.ifft(fourier_transform)

    # Create subplot
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)

    # Frequency domain representation
    axis.set_title('Fourier transform depicting the frequency components')
    axis.plot(df_x_axis, abs(fourier_transform))
    axis.set_xlabel('Frequency')
    axis.set_ylabel('Amplitude')

    st.plotly_chart(figure,use_container_width=True)

    return inverse_fourier, fourier_transform

#  ----------------------------------- INVERSE FOURIER TRANSFORM FUNCTION ---------------------------------------------------
def fourier_inverse_transform(inverse_fourier,df,factor):
    points_num=int(factor/6 *250 )
    # Getting df x_axis and y_axis
    list_of_columns = df.columns
    df_x_axis = np.linspace(0,10,points_num)
    # df_x_axis = list(df[list_of_columns[0]])   
    df_y_axis = list(df[list_of_columns[1]])
    # Slicing big data
    if (len(df_x_axis)>points_num):
        df_x_axis=df_x_axis[:points_num]
    if (len(df_y_axis)>points_num):
        df_y_axis=df_y_axis[:points_num]


    # Create subplot
    figure, axis = plt.subplots()
    plt.subplots_adjust(hspace=1)

    # Frequency domain representation
    axis.set_title('Inverse Fourier transform depicting the frequency components')
    axis.plot(df_x_axis, inverse_fourier)
    axis.set_xlabel('Frequency')
    axis.set_ylabel('Amplitude')

    fig,ax = plt.subplots()
    ax.set_title('The Actual Data')
    ax.plot(df_x_axis,df_y_axis)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    st.plotly_chart(figure,use_container_width=True)
    st.plotly_chart(fig,use_container_width=True)

def save_signal():
    # define electrocardiogram as ecg model
    ecg = electrocardiogram()
    # frequency is 0
    frequency = 80
    # calculating time data with ecg size along with frequency
    time_data = np.arange(ecg.size) / frequency
    
    # plotting time and ecg model
    # plt.plot(time_data[:1000], ecg[:1000])
    # plt.xlabel("time in seconds")
    # plt.ylabel("ECG in milli Volts")
    
    # display
    # plt.show()
    graph = pd.DataFrame({'time':time_data,'amp':ecg})
    df = pd.DataFrame(graph) 
    # saving the dataframe 
    csv_file=df.to_csv()
    return csv_file


# def factor (sampFreq):
#     fft_out = np.fft.fft(variabls.uploaded_audio_Yamp)          #el fourier bt3t el amplitude eli hnshtghl beha fl equalizer
#     abs_fft_out=np.abs(fft_out)[:len(variabls.uploaded_audio_time)//2] #np.abs 3shan el rsm
#     x_axis_fourier = fftfreq(len(variabls.uploaded_audio_Yamp),(1/sampFreq))[:len(variabls.uploaded_audio_Yamp)//2] #3shan mbd2sh mn -ve
#     filtered=[]
#     filtered_out=[]
#     for i in range(lltered_out.append(0)
#             filtered.append(fft_out[i])en(x_axis_fourier)):
#         if 600<x_axis_fourier[i]<700 or 1620<x_axis_fourier[i]<1820 or 2310<x_axis_fourier[i]<2510:
#             filtered.append(0)
#             filtered_out.append(fft_out[i])
#         else:
#             fi
#             plotting(x_axis_fourier,filtered)
#     inverse=np.real(np.fft.ifft(filtered,))
#     inverse_out=np.real(np.fft.ifft(filtered_out,))