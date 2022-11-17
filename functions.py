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
    vowel_freq_range_S=[1695,8000,'S']
    vowel_freq_range_J=[2200,8000,'J'] 
    vowel_freq_range_T=[1800,8000,'T']    
    vowel_tuble=(vowel_freq_range_SH,vowel_freq_range_S,vowel_freq_range_J,vowel_freq_range_T)
    vowel_old_file=None
    vowel_used_file_fft=[]
    vowel_used_file_fftfreq=[]
    vowel_used_file_sample_rate=0

    slider_values=[]

parent_dir  = os.path.dirname(os.path.abspath(__file__))
build_dir   = os.path.join(parent_dir, "build")
_vertical_slider = components.declare_component("vertical_slider", path=build_dir)


def slider_init_values(slider_labels,value, step, min=min, max=max):

    slider_num=len(slider_labels)
    labels_columns=st.columns(slider_num)

    sliders_columns=st.columns(slider_num)

    index1=0
    index2=0
    list_of_sliders_values = []

    while index1 < slider_num:
        with labels_columns[index1]:
            st.write(slider_labels[index1])
        index1 +=1

    while index2 < slider_num:
        with sliders_columns[index2]:
            sliders = (vertical_slider(value, step, min, max,index2))
            list_of_sliders_values.append(sliders)
        index2 +=1
        

    variables.slider_values=list_of_sliders_values


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
def uniform_range_mode(column1,column2, column3, column5,column6, column7,audio_file, show_spectrogram,file_name):
    signal_x_axis, signal_y_axis, sample_rate= read_audio(audio_file)    # Read Audio File
    
    yf,xf = fourir(signal_x_axis, signal_y_axis)
    step=len(xf)/9.8
    with column3:
        slider_labels=['   1 ','  2' ,'3','4','5','6','7','8','9','10']
   

    points_per_freq = len(xf) / (xf[-1])  # duration
   
    slider_init_values(slider_labels,0,1,0,10)

    for index2,value in enumerate(variables.slider_values):
  
        # st.write(int(step * index2)  , int( step * index2 +int( step)))
        yf[int(step * index2)  : int( step * index2 +int( step))] *= value
    else:
        pass
    
    modified_signal,modified_signal_channel =inverse_fourir(yf)   # returns two channels
    
    write(".Equalized_Music.wav", sample_rate*2, modified_signal_channel)   # creates the modified song
    
    if (show_spectrogram):
        plot_spectrogram(column1,file_name)
        plot_spectrogram(column2,".Equalized_Music.wav")
    else:
        start_btn  = column5.button(label='Start')
        pause_btn  = column6.button(label='Pause')
        column1.audio  (audio_file, format='audio/wav')                       # displaying the audio before editing
        resume_btn = column7.button(label='resume')
        column3.audio(".Equalized_Music.wav", format='audio/wav')             # displaying the audio after editing

        with column1:
            Dynamic_graph(signal_x_axis,signal_y_axis,modified_signal,start_btn,pause_btn,resume_btn)
            
    
    
#--------------------------------------Musical Instrument Function -------------------------------------------
def music_control(column1,column2, column3, column5,column6, column7,audio_file, show_spectrogram,file_name):
    signal_x_axis, signal_y_axis, sample_rate  = read_audio(audio_file)    # Read Audio File
    yf,xf = fourir(signal_x_axis, signal_y_axis)
    duration = len(xf) / (xf[-1])  # duration

    slider_labels=['Violin / Piano','Drums']
    slider_init_values(slider_labels,1,1,0,2)
   
  
    yf[int(1*duration):int(1000*duration)]*=variables.slider_values [1]     # drums off 1
 
    yf[int(500*duration ):int(14000*duration)]*=variables.slider_values [0]     # violion off 2 and piano off
    yf*2
    modified_signal,modified_signal_channel =inverse_fourir(yf)   # returns two channels
    
    write(".Equalized_Music.wav", sample_rate*2, modified_signal_channel)   # creates the modified song
    
    if (show_spectrogram):
        plot_spectrogram(column1,file_name)
        plot_spectrogram(column2,".Equalized_Music.wav")
    else:
        start_btn  = column5.button(label='Start')
        pause_btn  = column6.button(label='Pause')
        column1.audio  (audio_file, format='audio/wav')                       # displaying the audio before editing
        resume_btn = column7.button(label='resume')
        column3.audio(".Equalized_Music.wav", format='audio/wav')             # displaying the audio after editing

        with column1:
            Dynamic_graph(signal_x_axis,signal_y_axis,modified_signal,start_btn,pause_btn,resume_btn)
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
    df = pd.read_csv('arrhythmia_components.csv') # reading arrhhytmia component
    sub=df['amp_fft'][0:variables.points_num]
    for index in np.arange(0,len(sub)):    # converting string to complex
            sub[index]=complex(sub[index])
    weighted_arrhythmia=sub     # multiplying arrhythmia components by weighting factor
    for index in np.arange(0,len(weighted_arrhythmia)):
        weighted_arrhythmia[index]=sub[index]* complex(arrhythmia)*(-1)
    new_y=np.add(y_fourier,weighted_arrhythmia)     # adding weighted arrhythmia and uploaded complex amp

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
def voice_changer(column1,column5,file_name, show_spectrogram):
    voice = st.sidebar.radio('Voice', options=["Thicker Voice", "Smoother Voice"])
    
    if voice == "Thicker Voice":
        empty = column5.empty()
        empty.empty()
        speed_rate = 1.4
        speed_rate           = 1.4
        sampling_rate_factor = 1.4
    
    elif voice == "Smoother Voice":
        empty = column5.empty()
        empty.empty()
        speed_rate           = 0.5
        sampling_rate_factor = 0.5


    loaded_sound_file, sampling_rate = librosa.load(file_name, sr=None)
    loaded_sound_file                = librosa.effects.time_stretch(loaded_sound_file, rate=speed_rate)

    song = ipd.Audio(loaded_sound_file, rate = sampling_rate / sampling_rate_factor)
    empty.write(song)

    # column1.audio(audio_file, format="audio/wav")

    if(show_spectrogram):
            plot_spectrogram(column1, file_name)
    else:
            plot_spectrogram(column1,file_name)

#------------------------------------------------- DYNAMIC PLOTTING -----------------------------------------------
def plot_animation(df):
    brush  = alt.selection_interval()
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
def vowels_mode(column1,column3,column5,column6, column7,uploaded_file,show_spectrogram):
    signal_x_axis, signal_y_axis, sample_rate = read_audio(uploaded_file)    # Read Audio File
    if uploaded_file==variables.vowel_old_file:  #check if file hasn't changed use the computed fft
        y_axis_fourier=variables.vowel_used_file_fft
        sample_rate=variables.vowel_used_file_sample_rate
        x_axis_fourier=variables.vowel_used_file_fftfreq
    else:
        y_axis_fourier,x_axis_fourier = fourir(signal_x_axis, signal_y_axis)
        variables.vowel_old_file=uploaded_file
        variables.vowel_used_file_fft=y_axis_fourier
        variables.vowel_used_file_sample_rate=sample_rate
        variables.vowel_used_file_fftfreq=x_axis_fourier

    slider_labels=['SH','S','J','T']
    slider_init_values(slider_labels,1,1,0,2)

    
    filtered_fft=apply_slider_value(x_axis_fourier,y_axis_fourier,variables.slider_values)
    


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
        start_btn  = column5.button(label='Start')
        pause_btn  = column6.button(label='Pause')
        resume_btn = column7.button(label='resume')
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