import random 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.fft import irfft
from scipy.io.wavfile import write
import wave
import pylab
import librosa
import librosa.display
import time
import altair as alt
import pandas as pd
import wavio
import os 
import streamlit.components.v1 as components 
class variabls:
    
    points_num=1000
    start=0
    graph_size=0

    vowel_freq_ae=[860,2850]
    vowel_freq_a=[850,2800]
    slider_tuble=(vowel_freq_ae,vowel_freq_a)

parent_dir  = os.path.dirname(os.path.abspath(__file__))
build_dir   = os.path.join(parent_dir, "build")
_vertical_slider = components.declare_component("vertical_slider", path=build_dir)


def vertical_slider(value, step, min=min, max=max, key=None):
    slider_value = _vertical_slider(value=value,step=step, min=min, max=max, key=key, default=value)
    return slider_value
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
def uniform_range_mode(column2, column3, audio_file, show_spectrogram,file_name):
    signal_x_axis, signal_y_axis, sample_rate ,sound_info = read_audio(audio_file)    # Read Audio File

    yf = rfft(signal_y_axis)                                               # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform
    
    points_per_freq = len(xf) / (xf[-1])  # duration
    columns=st.columns(10)
    index=0
    list_of_sliders_values = []
    while index < 10:
        with columns[index]:
            sliders = (vertical_slider(1,1,1,10,index))
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
        # pylab.specgram(sound_info, Fs=sample_rate)
        # column2.pyplot(pylab)
        plot_spectrogram(column2,file_name)
        plot_spectrogram(column3,".Equalized_Music.wav")


    else:
        # plotting_graphs(column2,signal_x_axis,signal_y_axis,False)
        Dynamic_graph(signal_x_axis,signal_y_axis,signal_x_axis,modified_signal)

    column2.audio  (audio_file, format='audio/wav') # displaying the audio before editing
    column3.audio(".Equalized_Music.wav", format='audio/wav')             # displaying the audio after editing
#-------------------------------------- ARRYTHMIA FUNCTION ----------------------------------------------------
def ECG_mode(uploaded_file, show_spectrogram):

    # ------------ECG Sliders  
    Arrhythmia  =st.slider('Arrhythmia mode'                    , step=1, max_value=100 , min_value=-100  ,value=0 )
    Arrhythmia/=100
    # Reading uploaded_file
    df = pd.read_csv(uploaded_file)
    uploaded_xaxis=df['time']
    uploaded_yaxis=df['amp']
    smap_time=uploaded_xaxis[1]-uploaded_xaxis[0]
    samp_rate=1/smap_time
    input_col, output_col=st.columns([1,1])

    # Slicing big data
    if (len(uploaded_xaxis)>variabls.points_num):
        uploaded_xaxis=uploaded_xaxis[:variabls.points_num]
    if (len(uploaded_yaxis)>variabls.points_num):
        uploaded_yaxis=uploaded_yaxis[:variabls.points_num]
    # Plotting input signal
    if (show_spectrogram):
            pylab.specgram(uploaded_yaxis[50:950], Fs=samp_rate)
            input_col.pyplot(pylab)
    else:
        with input_col:
            uploaded_fig,uploaded_ax = plt.subplots()
            uploaded_ax.set_title('ECG input ')
            uploaded_ax.plot(uploaded_xaxis[50:950],uploaded_yaxis[50:950])
            uploaded_ax.set_xlabel('Time ')
            uploaded_ax.set_ylabel('Amplitude (mv)')
            st.plotly_chart(uploaded_fig)

    # fourier transorm
    y_fourier = np.fft.fft(uploaded_yaxis)
    
    y_fourier=arrhythmia (Arrhythmia,y_fourier)
    # abs_y_fourier=np.abs(y_fourier)
    # x_fourier = fftfreq(len(uploaded_xaxis),(uploaded_xaxis[5]-uploaded_xaxis[0])/5) 
    # length=(len( y_fourier ))//2


    y_inverse_fourier = np.fft.ifft(y_fourier)

    if (show_spectrogram):
            pylab.specgram(np.abs(y_inverse_fourier[50:950]), Fs=samp_rate)
            output_col.pyplot(pylab)
    else:
        with output_col:
            inverse_fig,inverse_ax = plt.subplots()
            inverse_ax.set_title('ECG output ')
            inverse_ax.plot(uploaded_xaxis[50:950],y_inverse_fourier[50:950])  
            inverse_ax.set_xlabel('Time ')
            inverse_ax.set_ylabel('Amplitude (mv)')
            st.plotly_chart(inverse_fig)
        
def arrhythmia (arrhythmia,y_fourier):
    new_y=y_fourier
    # reading arrhhytmia component
    df = pd.read_csv('arrhythmia_components.csv')
    sub=df['sub'][0:variabls.points_num]
    abs_sub=df['abs_sub']
    
    # converting string to complex
    for index in np.arange(0,len(sub)):
            sub[index]=complex(sub[index])
    # multiplying arrhythmia components by weighting factor
    weighted_arrhythmia=sub
    for index in np.arange(0,len(weighted_arrhythmia)):
        weighted_arrhythmia[index]=sub[index]* complex(arrhythmia)*(-1)
    # addinh weighted arrhythmia and uploaded complex amp

    # result = [item * arrhythmia for item in sub]
    new_y=np.add(y_fourier,weighted_arrhythmia)

    return new_y
#------------------
#------------------------------------------ PLOTTING FUNCTION -------------------------------------------------------
def plotting_graphs(column,x_axis,y_axis,flag):
    fig,axs = plt.subplots()
    fig.set_size_inches(6,3)
    plt.plot(x_axis,y_axis)
    if flag == True:
        pass
        # plt.xlim(45, 55)
        # plt.xlabel("Time in s")
        # plt.ylabel("ECG in mV")     
    column.plotly_chart(fig)
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

def plot_animation(df):
    brush = alt.selection_interval()
    chart1 = alt.Chart(df).mark_line().encode(
            x=alt.X('time', axis=alt.Axis(title='Time')),
            # y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude')),
        ).properties(
            width=500,
            height=300
        ).add_selection(
            brush
        ).interactive()
    
    figure = chart1.encode(y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude'))) | chart1.encode(y ='amplitude after processing').add_selection(
            brush)
    return figure
def Dynamic_graph( signal_x_axis, signal_y_axis,signal_x_axis1, signal_y_axis1,):
        df = pd.DataFrame({'time': signal_x_axis[::30], 'amplitude': signal_y_axis[:: 30], 'amplitude after processing': signal_y_axis1[::30]}, columns=['time', 'amplitude','amplitude after processing'])

        lines = plot_animation(df)
        line_plot = st.altair_chart(lines)

        col1,col2,col3,col4 = st.columns(4)
        start_btn  = col1.button(label='Start')
        pause_btn  = col2.button(label='Pause')
        resume_btn = col3.button(label='resume')
        # stop_btn   = col4.button(label='Stop')

        N = df.shape[0]  # number of elements in the dataframe
        burst = 10       # number of elements  to add to the plot
        size = burst     # size of the current dataset

        if start_btn:
          
            for i in range(1, N):
                variabls.start=i
                step_df = df.iloc[0:size]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                variabls.graph_size=size
                size = i * burst 
                print('start')                    

        if resume_btn: 
            
            for i in range( variabls.start,N):
                variabls.start=i
                step_df = df.iloc[0:size]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                variabls.graph_size=size
                size = i * burst
                print('resume')                

        if pause_btn:
            step_df = df.iloc[0:variabls.graph_size]
            lines = plot_animation(step_df)
            line_plot = line_plot.altair_chart(lines)
            print('pause')  
          
def plot_spectrogram(column,audio_file):
    y, sr = librosa.load(audio_file)
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    column.pyplot(fig)

def vowels_mode(uploaded_file,column_in,column_out):
    ae_freq_value=st.slider('ae',min_value=0,max_value=1,value=1,key=1)
    a_freq_value=st.slider('ae',min_value=0,max_value=1,value=1,key=2)
    signal_x_axis, signal_y_axis, sample_rate ,sound_info = read_audio(uploaded_file)    # Read Audio File


    y_axis_fourier = np.fft.fft(signal_y_axis)                       
    x_axis_fourier = np.fft.fftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0]))
    st.write(len(y_axis_fourier))
    
    filtered_fft=apply_slider_value(x_axis_fourier,y_axis_fourier,ae_freq_value,a_freq_value)
    st.write(len(filtered_fft))
    inverse=(np.fft.ifft(filtered_fft,))
    plotting_graphs(column_in,signal_x_axis,signal_y_axis)
    plotting_graphs(column_out,x_axis_fourier*2,np.real(inverse))
    wavio.write("myfile.wav", inverse, sample_rate//2, sampwidth=1)
    st.audio("myfile.wav", format='audio/wav')

def apply_slider_value(x_axis_fourier,y_axis_fourier,ae_freq_value,a_freq_value):
    ae_freq_value = 0.01 if ae_freq_value == 0 else ae_freq_value 
    a_freq_value = 0.01 if a_freq_value == 0 else a_freq_value 
    slider_values=[ae_freq_value,a_freq_value]
    filtered_fft=y_axis_fourier
    for slider_counter in range(len(slider_values)):
        min_value=variabls.slider_tuble[slider_counter][0]
        max_value=variabls.slider_tuble[slider_counter][1]
        for i in range(len(x_axis_fourier)):
            if min_value<=x_axis_fourier[i]<=max_value:#ae
                filtered_fft[i]+=y_axis_fourier[i]*slider_values[slider_counter]
            else:
                filtered_fft[i]+=y_axis_fourier[i]*1000
    return(filtered_fft/3)


