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
class variabls:
    
    points_num=1000
    vowel_freq_ae=[860,2850]
    vowel_freq_a=[850,2800]
    slider_tuble=(vowel_freq_ae,vowel_freq_a)
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

    if (show_spectrogram):
        # pylab.specgram(sound_info, Fs=sample_rate)
        # column2.pyplot(pylab)
        plot_spectrogram(column2,file_name)

    else:
        # plotting_graphs(column2,signal_x_axis,signal_y_axis,False)
        with column2 :
             Dynamic_graph(signal_x_axis,signal_y_axis,'start1')

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
        # pylab.specgram(modified_signal_channel, Fs=sample_rate)
        # column3.pyplot(pylab)
        plot_spectrogram(column3,".Equalized_Music.wav")
    else:
        # plotting_graphs(column3,signal_x_axis,modified_signal,False)
        with column3 :
            Dynamic_graph(signal_x_axis,modified_signal,'start2')

    column2.audio  (audio_file, format='audio/wav') # displaying the audio before editing
    column3.audio(".Equalized_Music.wav", format='audio/wav')             # displaying the audio after editing
    # Dynamic_graph(signal_x_axis, signal_y_axis )
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
            pylab.specgram(y_inverse_fourier[50:950], Fs=samp_rate)
            output_col.pyplot(pylab)
    else:
        with output_col:
            inverse_fig,inverse_ax = plt.subplots()
            inverse_ax.set_title('ECG output ')
            inverse_ax.plot(uploaded_xaxis[50:950],y_inverse_fourier[50:950])  
            inverse_ax.set_xlabel('Time ')
            inverse_ax.set_ylabel('Amplitude (mv)')
            st.plotly_chart(inverse_fig)
        
 
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
    fig,axs = plt.subplots()
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
def plot_animation(df):
    lines = alt.Chart(df).mark_line().encode(
        x=alt.X('time', axis=alt.Axis(title='date')),
        y=alt.Y('amplitude', axis=alt.Axis(title='value')),
    ).properties(
        width=600,
        height=300
    )
    return lines
def Dynamic_graph(signal_x_axis, signal_y_axis,button_name ):

    # start of dynamic plotting
    resize = alt.selection_interval(bind='scales')

    df = pd.DataFrame({'time': signal_x_axis[::1500], 'amplitude': signal_y_axis[:: 1500]}, columns=['time', 'amplitude'])
    
    lines = alt.Chart(df).mark_line().encode( x=alt.X('0:T', axis=alt.Axis(title='time')),
                                              y=alt.Y('1:Q', axis=alt.Axis(title='value'))).properties(width=600,height=300).add_selection(
        resize
    )

    N = df.shape[0]  # number of elements in the dataframe
    burst = 6        # number of elements (months) to add to the plot
    size = burst     # size of the current dataset

    # Plot Animation
    line_plot = st.altair_chart(lines)
    start_btn = st.checkbox(label=button_name)



    if start_btn:
        for i in range(1, N):
            step_df = df.iloc[0:size]
            lines = plot_animation(step_df)
            line_plot = line_plot.altair_chart(lines)
            size = i + burst
            if size >= N:
                size = N - 1
            time.sleep(.00000000001)
    # end of dynamic plotting
# def initial_time_graph(df1,df2):
#     df1 = pd.DataFrame({'time': signal_x_axis[::1500], 'amplitude': signal_y_axis[:: 1500]}, columns=['time', 'amplitude'])
#     df2= pd.DataFrame({'time': signal_x_axis[::1500], 'amplitude': signal_y_axis[:: 1500]}, columns=['time', 'amplitude'])

#     resize = alt.selection_interval(bind='scales')
#     chart1 = alt.Chart(df1).mark_line().encode(
#     x=alt.X('time:T', axis=alt.Axis(title='date',labels=False)),
#     y=alt.Y('signal:Q',axis=alt.Axis(title='value'))
#     ).properties(
#         width=600,
#         height=300
#     ).add_selection(
#         resize
#     )

#     chart2 = alt.Chart(df2).mark_line().encode(
#         x=alt.X('time:T', axis=alt.Axis(title='date',labels=False)),
#         y=alt.Y('signal:Q',axis=alt.Axis(title='value'))
#     ).properties(
#         width=600,
#         height=300
#     ).add_selection(
#         resize
#     )


#     chart=alt.concat(chart1, chart2)
#     return chart

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




# def dynamic():
#     test_music1, sr1 = librosa.load("StarWars3.wav")
#     d=(pd.DataFrame(test_music1,columns=['amp']))
#     data=st.empty()
#     def make_chart(df,ymin, ymax):
#         fig = go.Figure(layout_yaxis_range=[ymin, ymax])
#         fig.add_trace(go.Scatter(x=(df.index/300000),y=df['amp']))
#         fig.update_layout(width=800, height=570, xaxis_title='time',yaxis_title='amplitude')
#     ymax = max(d['amp'])
#     ymin = min(d['amp'])
#     n = len(d)
#     for i in range(0,n+10000, 200):
#         df_tmp = (d.iloc[i:i+10000, :])
#         with data:
#             make_chart(df_tmp, ymin, ymax)
#         time.sleep(0.00001)
