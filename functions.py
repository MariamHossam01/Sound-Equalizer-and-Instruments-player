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
import altair as alt
import pandas as pd
import wavio
import os 
import streamlit.components.v1 as components 
import IPython.display as ipd

class variables:
    #-------- dynamic
    points_num=1000
    start=0
    graph_size=0
    #--------ecg
    ecg_file=''
    ecg_xaxis=[]
    ecg_yaxis=[]
    ecg_fft=[]
    ecg_samp_rate=0
    #--- vowels
    vowel_freq_range_SH=[1850,5600,'SH']
    vowel_freq_range_S=[4951,8000,'S']
    vowel_freq_range_CH=[4018,8000,'CH']
    vowel_freq_range_T=[1530,8000,'T']
    vowel_tuble=(vowel_freq_range_SH,vowel_freq_range_S,vowel_freq_range_CH,vowel_freq_range_T)
    vowel_old_file=None
    vowel_used_file_fft=[]
    vowel_used_file_fftfreq=[]
    vowel_used_file_sample_rate=0

parent_dir  = os.path.dirname(os.path.abspath(__file__))
build_dir   = os.path.join(parent_dir, "build")
_vertical_slider = components.declare_component("vertical_slider", path=build_dir)


def vertical_slider(value, step, min=min, max=max, key=None):
    slider_value = _vertical_slider(value=value,step=step, min=min, max=max, key=key, default=value,)
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
    return signal_x_axis, signal_y_axis, sample_rate
#--------------------------------------------Fourir Transform----------------------------------------------------------
def fourir(signal_x_axis, signal_y_axis):
    yf = rfft(signal_y_axis)                                               # returns complex numbers of the y axis in the data frame
    xf = rfftfreq(len(signal_y_axis), (signal_x_axis[1]-signal_x_axis[0])) # returns the frequency x axis after fourier transform
    return yf,xf
#------------------------------------------Inverse Fourir Transform----------------------------------------------------------
def inverse_fourir(yf):
    inverse_signal = irfft(yf)     
    modified_signal_channel = np.int16(inverse_signal) # returns two channels            # returns the inverse transform a
    return inverse_signal,modified_signal_channel 
#-------------------------------------- UNIFORM RANGE MODE FUNCTION ----------------------------------------------------
def uniform_range_mode(column1,column2, column3,audio_file, show_spectrogram,file_name):
    column4,column5,column6=st.columns([4,4,4])
    
    signal_x_axis, signal_y_axis, sample_rate= read_audio(audio_file)    # Read Audio File

    yf,xf = fourir(signal_x_axis, signal_y_axis)
    
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
    
    modified_signal,modified_signal_channel =inverse_fourir(yf)   # returns two channels
    
    write(".Equalized_Music.wav", sample_rate*2, modified_signal_channel)   # creates the modified song
    with column1:
        column1.audio  (audio_file, format='audio/wav') # displaying the audio before editing
    with column3:
        column3.audio(".Equalized_Music.wav", format='audio/wav') 
    if (show_spectrogram):
        plot_spectrogram(column1,file_name)
        plot_spectrogram(column3,".Equalized_Music.wav")
        # column1.audio  (audio_file, format='audio/wav') # displaying the audio before editing
        # column3.audio(".Equalized_Music.wav", format='audio/wav') 
    else:
        
            start_btn  = column4.button(label='Start')
            pause_btn  = column5.button(label='Pause')
                                                                            # displaying the audio after editing
            resume_btn = column6.button(label='resume')
    # column1.audio(".Equalized_Music.wav", format='audio/wav')             # displaying the audio after editing
            with column1:
  
                Dynamic_graph(signal_x_axis,signal_y_axis,modified_signal,start_btn,pause_btn,resume_btn)
            
            
    
    
#--------------------------------------Musical Instrument Function -------------------------------------------
def music_control(column1,column2, column3, audio_file, show_spectrogram,file_name):
    column4,column5,column6=st.columns([4,4,4])
    signal_x_axis, signal_y_axis, sample_rate  = read_audio(audio_file)    # Read Audio File
    columns=st.columns(2)
    index=0
    list_of_sliders_values_musical = []
    yf,xf = fourir(signal_x_axis, signal_y_axis)
    duration = len(xf) / (xf[-1])  # duration
    while index <=1:
        with columns[index]:
                   sliders =  (vertical_slider(1,1,0,2,index))
        index +=1
        list_of_sliders_values_musical .append(sliders)
   
  
    yf[int(1*duration):int(1000*duration)]*=list_of_sliders_values_musical [1]     # drums off 1
    yf*=5
 
    yf[int(500*duration ):int(14000*duration)]*=list_of_sliders_values_musical [0]     # violion off 2 and piano off
    modified_signal,modified_signal_channel =inverse_fourir(yf)   # returns two channels
    
    write(".Equalized_Music.wav", sample_rate*2, modified_signal_channel)   # creates the modified song
    with column1:
        column1.audio  (audio_file, format='audio/wav') # displaying the audio before editing
    with column3:
        column3.audio(".Equalized_Music.wav", format='audio/wav') 
    if (show_spectrogram):
        plot_spectrogram(column1,file_name)
        plot_spectrogram(column3,".Equalized_Music.wav")
    else:
            start_btn  = column4.button(label='Start')
            pause_btn  = column5.button(label='Pause')
                                                        # displaying the audio before editing
                                                        # displaying the audio after editing
            resume_btn = column6.button(label='resume')
               # displaying the audio after editing

            with column1:
                Dynamic_graph(signal_x_axis,signal_y_axis,modified_signal,start_btn,pause_btn,resume_btn)
       
           
                
            #     column1.audio  (audio_file, format='audio/wav') # displaying the audio before editing
            # column4.audio(".Equalized_Music.wav", format='audio/wav') 
#-------------------------------------- ARRYTHMIA FUNCTION ----------------------------------------------------
def ECG_mode(column1,column3,uploaded_file, show_spectrogram):
    # ------------ECG Sliders 
    # Arrhythmia=vertical_slider(0,1,-100,100)
    Arrhythmia=st.slider('Arrhythmia',min_value =-100,max_value=100,step=1,value=0)
    Arrhythmia/=100 
    # 3 columns input , output ,slider
    # input_col, output_col=st.columns([10,10])

    if(variables.ecg_file==''):
        ECG_init(uploaded_file)
    elif (variables.ecg_file==uploaded_file): 
            pass
    else:
        ECG_init(uploaded_file)


    uploaded_xaxis=variables.ecg_xaxis
    uploaded_yaxis=variables.ecg_yaxis
    y_fourier=variables.ecg_fft
    # Plotting input signal
    if (show_spectrogram):
            pylab.specgram(uploaded_yaxis, Fs=360)
            column1.pyplot(pylab)
    else:
        with column1:
            uploaded_fig,uploaded_ax = plt.subplots()
            uploaded_ax.set_title('ECG input ')
            uploaded_ax.plot(uploaded_xaxis,uploaded_yaxis)
            uploaded_ax.set_xlabel('Time ')
            uploaded_ax.set_ylabel('Amplitude (mv)')
            st.plotly_chart(uploaded_fig)
    # signal after modification (adding,removing arrhythmia)
    y_fourier=arrhythmia (Arrhythmia,y_fourier)
    # Plotting output graph after modification
    y_inverse_fourier = np.fft.ifft(y_fourier)
    if (show_spectrogram):
            pylab.specgram(np.abs(y_inverse_fourier), Fs=360)
            column3.pyplot(pylab)
    else:
        with column3:
            inverse_fig,inverse_ax = plt.subplots()
            inverse_ax.set_title('ECG output ')
            inverse_ax.plot(uploaded_xaxis,y_inverse_fourier)  
            inverse_ax.set_xlabel('Time ')
            inverse_ax.set_ylabel('Amplitude (mv)')
            st.plotly_chart(inverse_fig)
def ECG_init(uploaded_file):
    # Reading uploaded_file
    variables.ecg_file=uploaded_file
    df = pd.read_csv(uploaded_file)
    uploaded_xaxis=df['time']
    uploaded_yaxis=df['amp']

    smap_time=uploaded_xaxis[1]-uploaded_xaxis[0]
    samp_rate=1/smap_time

    # Slicing big data
    if (len(uploaded_xaxis)>variables.points_num):
        uploaded_xaxis=uploaded_xaxis[:variables.points_num]
    if (len(uploaded_yaxis)>variables.points_num):
        uploaded_yaxis=uploaded_yaxis[:variables.points_num]
    # fourier transorm
    y_fourier = np.fft.fft(uploaded_yaxis)
    variables.ecg_fft=y_fourier
    # fourier transorm
    y_fourier = np.fft.fft(uploaded_yaxis)
    # updatting
    variables.ecg_fft=y_fourier
    variables.ecg_xaxis=uploaded_xaxis
    variables.ecg_yaxis=uploaded_yaxis
    variables.ecg_samp_rate=samp_rate
def arrhythmia (arrhythmia,y_fourier):
    new_y=y_fourier
    # reading arrhhytmia component
    df = pd.read_csv('arrhythmia_components.csv')
    sub=df['amp_fft'][0:variables.points_num]

    
    # converting string to complex
    for index in np.arange(0,len(sub)):
            sub[index]=complex(sub[index])
    # multiplying arrhythmia components by weighting factor
    weighted_arrhythmia=sub
    for index in np.arange(0,len(weighted_arrhythmia)):
        weighted_arrhythmia[index]=sub[index]* complex(arrhythmia)*(-1)
    # addinh weighted arrhythmia and uploaded complex amp
    new_y=np.add(y_fourier,weighted_arrhythmia)

    return new_y
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
    #---------------------------------------- OPTIONAL FUNCTION ------------------------------------------------
def voice_changer(uploaded_file, column1, column2, column3, show_spectrogram,start_btn,pause_btn,resume_btn):

    signal_x_axis, signal_y_axis, sample_rate  = read_audio(uploaded_file)    # Read Audio File

    voice = column1.radio('Voice', options=["Thicker Voice", "Smoother Voice"])

    column2.audio(uploaded_file, format="audio/wav")
    if voice == "Thicker Voice":
        empty = column3.empty()
        empty.empty()
        speed_rate = 1.4
        sampling_rate_factor = 1.4

    elif voice == "Smoother Voice":
        empty = column3.empty()
        empty.empty()
        speed_rate = 0.5
        sampling_rate_factor = 0.5

    loaded_sound_file, sampling_rate = librosa.load(uploaded_file, sr=None)
    loaded_sound_file                = librosa.effects.time_stretch(loaded_sound_file, rate=speed_rate)

    song = ipd.Audio(loaded_sound_file, rate = sampling_rate / sampling_rate_factor)
    empty.write(song)
    if(show_spectrogram):
            plot_spectrogram(column2, uploaded_file.name)
    else:
            # plotting_graphs(column2,signal_x_axis,signal_y_axis,False)
        
        Dynamic_graph( signal_x_axis, signal_y_axis, signal_y_axis,start_btn,pause_btn,resume_btn)
#------------------------------------------------- DYNAMIC PLOTTING -----------------------------------------------
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
def Dynamic_graph( signal_x_axis, signal_y_axis, signal_y_axis1,start_btn,pause_btn,resume_btn):
        df = pd.DataFrame({'time': signal_x_axis[::1400], 'amplitude': signal_y_axis[:: 1400], 'amplitude after processing': signal_y_axis1[::1400]}, columns=['time', 'amplitude','amplitude after processing'])

        lines = plot_animation(df)
        line_plot = st.altair_chart(lines)

        N = df.shape[0]  # number of elements in the dataframe
        burst = 10       # number of elements  to add to the plot
        size = burst     # size of the current dataset

        if start_btn:
          
            for i in range(1, N):
                variables.start=i
                step_df = df.iloc[0:size]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                variables.graph_size=size
                size = i + burst 

        if resume_btn: 
            
            for i in range( variables.start,N):
                variables.start=i
                step_df = df.iloc[0:size]
                lines = plot_animation(step_df)
                line_plot = line_plot.altair_chart(lines)
                variables.graph_size=size
                size = i + burst

        if pause_btn:
            step_df = df.iloc[0:variables.graph_size]
            lines = plot_animation(step_df)
            line_plot = line_plot.altair_chart(lines)
          
def plot_spectrogram(column,file_name):
    y, sr = librosa.load(file_name)
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    column.pyplot(fig)


#-------------------------------------- Vowels FUNCTION ----------------------------------------------------

def vowels_mode(column1,column2,column3,uploaded_file,show_spectrogram):
    column4,column5,column6=st.columns([4,4,4])
    signal_x_axis, signal_y_axis, sample_rate = read_audio(uploaded_file)    # Read Audio File
    if uploaded_file==variables.vowel_old_file:  #check if file hasn't changed use the computed fft
        y_axis_fourier=variables.vowel_used_file_fft
        sample_rate=variables.vowel_used_file_sample_rate
        x_axis_fourier=variables.vowel_used_file_fftfreq
        print('olddddddddddddddddddddddddddddddddddddddddddddddddddddd')
    else:
        y_axis_fourier,x_axis_fourier = fourir(signal_x_axis, signal_y_axis)
        variables.vowel_old_file=uploaded_file
        variables.vowel_used_file_fft=y_axis_fourier
        variables.vowel_used_file_sample_rate=sample_rate
        variables.vowel_used_file_fftfreq=x_axis_fourier
        print('newwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww')
    
    
    sliders_value=[]        #write four variables
    columns=st.columns(4)
    index=0
    while index <=3:
        with columns[index]:
                   sliders =  (vertical_slider(1,0.2,0,2,index))
        index +=1
        sliders_value.append(sliders)

    
    filtered_fft=apply_slider_value(x_axis_fourier,y_axis_fourier,sliders_value)
    


    inverse,modified_signal_channel =inverse_fourir(filtered_fft,)   # returns two channels
    
    column1.audio(uploaded_file, format='audio/wav')
    wavio.write("filtered_word.wav", inverse, sample_rate, sampwidth=1)
    column3.audio("filtered_word.wav", format='audio/wav')
    column3.write(' ')
    column3.write(' ')
    if show_spectrogram:
        plot_spectrogram(column1,uploaded_file.name)
        plot_spectrogram(column3,"filtered_word.wav")
    else:
        start_btn  = column4.button(label='Start')
        pause_btn  = column5.button(label='Pause')
        resume_btn = column6.button(label='resume')
        

        with column1:
                Dynamic_graph(signal_x_axis,signal_y_axis,inverse,start_btn,pause_btn,resume_btn)

def apply_slider_value(x_axis_fourier,y_axis_fourier,sliders_value):
    fft_han=y_axis_fourier.copy()
    for counter in range(len(sliders_value)):
        if  not sliders_value[counter]==1:
            range_list=[]
            for index in range(len(x_axis_fourier)):
                if variables.vowel_tuble[counter][0]<x_axis_fourier[index]<variables.vowel_tuble[counter][1]:
                    range_list.append(x_axis_fourier[index])
            range_of_freq=np.where(x_axis_fourier==max(range_list))[0][0]-np.where(x_axis_fourier==min(range_list))[0][0]
            han_value=(1-np.hanning(range_of_freq+1))* sliders_value[counter]   
            for counter in range(range_of_freq):
                fft_han[int(counter+np.where(x_axis_fourier==min(range_list))[0][0])]*=han_value[counter]

    return(fft_han)




