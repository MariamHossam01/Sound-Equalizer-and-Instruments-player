import io
import wave
import streamlit as st
from email.mime import audio
import numpy as np  
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import librosa as lr
import scipy
from scipy import signal
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import vowels2
from scipy.fft import fft, fftfreq,rfftfreq,ifft
from scipy.io.wavfile import write
import wavio
import plotly.graph_objects as go



freq_fig= go.Figure()
audio_fig= go.Figure()

uploaded_file =st.file_uploader('upload the signal file',['csv','ogg','wav'] , help='upload your signal file',key=12 )
if (uploaded_file):

    st.audio(uploaded_file, format='audio/wav',start_time=10)
    # analyising uploaded audio --> getting x ,y
    audio_amp,sampFreq= lr.load(uploaded_file)
    st.write('og')
    st.write(type(uploaded_file))
    time_axis =np.arange(0,len(audio_amp))/sampFreq
    # updating data in functions
    vowels2.variabls.uploaded_audio_time=time_axis
    vowels2.variabls.uploaded_audio_Yamp=audio_amp 

    # fft_parameters= np.fft.fft(trail2.variabls.uploaded_audio_Yamp)
    # frequency_phase = np.angle(fft_parameters)

    fft_letter = np.fft.fft(vowels2.variabls.uploaded_audio_Yamp)[:len(vowels2.variabls.uploaded_audio_time)//2]          #el fourier bt3t el amplitude eli hnshtghl beha fl equalizer
    

uploaded_file =st.file_uploader('upload the signal file',['csv','ogg','wav'] , help='upload your signal file' )
if (uploaded_file):

    st.audio(uploaded_file, format='audio/wav',start_time=10)
    # analyising uploaded audio --> getting x ,y
    audio_amp,sampFreq= lr.load(uploaded_file)
    st.write('og')
    st.write(type(uploaded_file))
    time_axis =np.arange(0,len(audio_amp))/sampFreq
    # updating data in functions
    vowels2.variabls.uploaded_audio_time=time_axis
    vowels2.variabls.uploaded_audio_Yamp=audio_amp 

    # fft_parameters= np.fft.fft(trail2.variabls.uploaded_audio_Yamp)
    # frequency_phase = np.angle(fft_parameters)

    fft_out = np.fft.fft(vowels2.variabls.uploaded_audio_Yamp)[:len(vowels2.variabls.uploaded_audio_time)//2]          #el fourier bt3t el amplitude eli hnshtghl beha fl equalizer
    frequency_phase = np.angle(fft_out)
    abs_fft_out=np.abs(fft_out)[:len(vowels2.variabls.uploaded_audio_time)//2] #np.abs 3shan el rsm
    temporary_frequency_magnitude = np.abs(fft_out)
    x_axis_fourier = np.fft.fftfreq(len(vowels2.variabls.uploaded_audio_Yamp),(1/sampFreq))[:len(vowels2.variabls.uploaded_audio_Yamp)//2] #3shan mbd2sh mn -ve

    
    ## sound_amplitude, sampling_rate = librosa.load('Input/FileName') 
    ## fig, ax = plt.subplots()
    ##librosa.display.waveshow(sound_amplitude, sr=sampling_rate, x_axis="time",ax=ax)
    ## st.pyplot(fig)
    # signal_temporary_amplitude=sound_amplitude
    # frequency=np.fft.rfftfreq(len( sound_amplitude),1/ sampling_rate)[:len(sound_amplitude)//2]
    # fft_parameters= np.fft.fft(sound_amplitude)[:len(sound_amplitude)//2]
    # frequency_phase = np.angle(fft_parameters)
    # frequency_magnitude = np.abs(fft_parameters)[:len(sound_amplitude)//2]
    # temporary_frequency_magnitude = np.abs(fft_parameters)

    complex_parameters = np.multiply(temporary_frequency_magnitude, np.exp(np.multiply(1j, frequency_phase)))
    signal_temporary_amplitude = np.int16(np.fft.ifft(complex_parameters))
#    filtered_val=[[660,840],[1077,1297],[2425,2765],[3531,4031],[3900,4500]]
#     filtered=[]
#     filtered_out=[]
#     flag=False
#     for i in range(len(x_axis_fourier)):
#         for j in filtered_val:
#             if j[0]<x_axis_fourier[i]<j[1]:
#                 flag=True
#             else:
#                 flag=False
        
#         if  flag:
#             filtered.append(0)
#             filtered_out.append(fft_out[i])
#         else:
            
#             filtered_out.append(0)
#             filtered.append(fft_out[i])
    filtered=[]
    filtered_out=[]
    # fft_letter.shape(fft_out)
    filtered_with_letter=fft_out - fft_letter
    for i in range(len(x_axis_fourier)):
        if 660<x_axis_fourier[i]<840 or 1077<x_axis_fourier[i]<1297 or 1600<x_axis_fourier[i]<1700 or 2425<x_axis_fourier[i]<2765:
            filtered.append(0)
            filtered_out.append(fft_out[i])
        else:
            filtered_out.append(0)
            filtered.append(fft_out[i])

            
    inverse=np.real(np.fft.ifft(filtered,))
    inverse_out=np.real(np.fft.ifft(filtered_out,))
    inverse_minus_letter=np.real(np.fft.ifft(filtered_with_letter,))
    # st.write(len(trail2.variabls.uploaded_audio_Yamp))
    # st.write(len(frequency_phase))
    # st.write(len(filtered))


    # x= np.exp(np.multiply(1j, frequency_phase))
    # # x.resize(abs_fft_out.shape)
    # complex_parameters = np.multiply(inverse,x)
    # complex_parameters
    # signal_temporary_amplitude = np.int16(np.fft.ifft(complex_parameters))

    # signal_temporary_amplitude 
    audio_fig.add_trace(go.Scatter(x=vowels2.variabls.uploaded_audio_time, y= vowels2.variabls.uploaded_audio_Yamp,name='real'))
    freq_fig.add_trace(go.Scatter(x=x_axis_fourier, y=abs_fft_out,name='fft'))
    freq_fig.add_trace(go.Scatter(x=x_axis_fourier, y=np.abs(filtered_with_letter),name='fft-letter'))
    audio_fig.add_trace(go.Scatter(x=vowels2.variabls.uploaded_audio_time*2, y= inverse,name='inverse'))
    audio_fig.add_trace(go.Scatter(x=vowels2.variabls.uploaded_audio_time*2, y= inverse_out,name='inverse out'))
    freq_fig.add_trace(go.Scatter(x=x_axis_fourier, y= np.abs(filtered_out),name='filtered_out'))
    freq_fig.add_trace(go.Scatter(x=x_axis_fourier, y=np.abs(filtered),name='filtered fft'))
    audio_fig.add_trace(go.Scatter(x=vowels2.variabls.uploaded_audio_time, y= signal_temporary_amplitude,name='inverse w phase'))
    audio_fig.add_trace(go.Scatter(x=vowels2.variabls.uploaded_audio_time*2, y= inverse_minus_letter,name='inverse wo letter'))

    # time_axis =np.arange(0,len(inverse))

    wavio.write("myfile.wav", inverse, sampFreq//2, sampwidth=1)
    st.audio("myfile.wav", format='audio/wav')

    wavio.write("myfileoutnoletter.wav", inverse_minus_letter, sampFreq//2, sampwidth=1)
    st.audio("myfileoutnoletter.wav", format='audio/wav')

    wavio.write("myfileout.wav", inverse_out, sampFreq//2, sampwidth=1)
    st.audio("myfileout.wav", format='audio/wav')
    st.write(freq_fig)
    st.write(audio_fig)









    # write("example.wav", sampFreq//2, inverse.astype(np.int16))

    # st.audio("example.wav", format='audio/wav')
    
    # rate =sampFreq
    # T =int( max(trail2.variabls.uploaded_audio_time))         # sample duration (seconds)
    # t = np.linspace(0, T, T*rate, endpoint=False)
    # x = inverse

    # file_in_memory = io.BytesIO()

    # wavio.write(file_in_memory, x, rate, sampwidth=3)
    
    
    # st.audio(file_in_memory, format='audio/wav')

    # lowcut = 1
    # highcut = 2000
    # order = 4
    # nyq = 0.5 * sampFreq
    # low = lowcut / nyq
    # high = highcut / nyq
    # b, a = signal.butter(order, [low, high], btype='bandstop')
    # y_filt = signal.filtfilt(b, a, trail2.variabls.uploaded_audio_Yamp)


    # st.audio(y_filt, format='audio/wav',start_time=10)

    # fft_out = fft(y_filt)          #el fourier bt3t el amplitude eli hnshtghl beha fl equalizer
    # fft_out=np.abs(fft_out)[:len(y_filt)//2] #np.abs 3shan el rsm
    # x_axis_fourier = fftfreq(len(y_filt),(1/sampFreq))[:len(y_filt)//2] #3shan mbd2sh mn -ve



    # Z=Y-Yf
    # f = np.arange(0, len(Z)) * sampFreq / len(Z)
    # df = int(200 * len(y_filt) / sampFreq)
    # Uploaded_fig.add_trace(go.Scatter(x=f[:df], y= Z[:df],))

    # resummed=0.5*Z+Yf
    # f = np.arange(0, len(resummed)) * sampFreq / len(resummed)
    # df = int(200 * len(y_filt) / sampFreq)

    # Uploaded_fig.add_trace(go.Scatter(x=f[:df], y= resummed[:df],))



# # Put the channels together with shape (2, 44100).
# audio = np.array([f, y_filt]).T

# # Convert to (little-endian) 16 bit integers.
# audio = (audio * (2 ** 15 - 1)).astype("<h")

# st.write(Uploaded_fig)

