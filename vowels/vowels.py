import io
import random
from time import sleep
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
import scipy.stats as stats

s= [[306.9553857724592,631.5177500146294], [ 1647.020165966535,1880.2260188067203], [2193.599524690138,2957.966758555921]]
st.session_state.data = [] if 'data' not in st.session_state else st.session_state.data

# def main():
#     # initial conditions
#     st.session_state.iter = 0 if 'iter' not in st.session_state else st.session_state.iter
#     st.session_state.datat = [] if 'data' not in st.session_state else st.session_state.data

#     # repopulate chart on next run with last data
#     chart = st.line_chart()
#     for data in st.session_state.data:
#         chart.add_rows(data)

#     stop = st.checkbox("Stop Update")

#     # iteration for simulation
#     while st.session_state.iter < 1000:
#         if stop:
#             break
#         # update the simulated chart
#         chart.add_rows(st.session_state.data[st.session_state.iter])
#         st.session_state.datat.append(st.session_state.data[st.session_state.iter])
#         sleep(0.1)
#         # keep last iteration of simulation in state
#         st.session_state.iter += 1



freq_fig= go.Figure()
audio_fig= go.Figure()


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

    fft_out = np.fft.fft(vowels2.variabls.uploaded_audio_Yamp)[:len(vowels2.variabls.uploaded_audio_time)//2]          #el fourier bt3t el amplitude eli hnshtghl beha fl equalizer
    frequency_phase = np.angle(fft_out)
    abs_fft_out=np.abs(fft_out)[:len(vowels2.variabls.uploaded_audio_time)//2] #np.abs 3shan el rsm
    temporary_frequency_magnitude = np.abs(fft_out)
    x_axis_fourier = np.fft.fftfreq(len(vowels2.variabls.uploaded_audio_Yamp),(1/sampFreq))[:len(vowels2.variabls.uploaded_audio_Yamp)//2] #3shan mbd2sh mn -ve

    complex_parameters = np.multiply(temporary_frequency_magnitude, np.exp(np.multiply(1j, frequency_phase)))
    signal_temporary_amplitude = np.int16(np.fft.ifft(complex_parameters))

    filtered=[]
    filtered_out=[]
    # fft_letter.shape(fft_out)
    # filtered_with_letter=np.subtract(np.real(fft_out[:2000]) ,np.real(fft_letter[:2000]))
    # st.write(type(fft_out))
    # st.write(np.abs(fft_out[:100]))
    # st.write(np.abs(fft_letter[:100]))
    
    # l spelling -WORKS SOUNDS LIKE M
    sigma=300
    y1=1-stats.norm.pdf(x_axis_fourier, 337.86637714069735, sigma)*(sigma*((2*np.pi)**0.5))
    y2=1-stats.norm.pdf(x_axis_fourier, 1816.980787283173, sigma)*((sigma)*((2*np.pi)**0.5))
    y3=1-stats.norm.pdf(x_axis_fourier, 3042.3901120979044, sigma)*(sigma*((2*np.pi)**0.5))

    # l tts 
    # sigma=200
    # y1=1-stats.norm.pdf(x_axis_fourier, 498.2988749296144, sigma)*(sigma*((2*np.pi)**0.5))
    # y2=1-stats.norm.pdf(x_axis_fourier, 1155.2030750767356, sigma)*((sigma)*((2*np.pi)**0.5))
    # y3=1-stats.norm.pdf(x_axis_fourier, 2173.874855345698, sigma)*(sigma*((2*np.pi)**0.5))
    st.write(len(y1)) 
    st.write(len(fft_out)) 
    
    for i in range(len(x_axis_fourier)):
        # if 745.9817451968214<x_axis_fourier[i]<880.0495737603036 or 1604.1363604717196<x_axis_fourier[i]<1866.880313925011 or 2347.9373824424865<x_axis_fourier[i]<2551.1516579934455: #ae (1)
        # if 391.90452637888<x_axis_fourier[i]<696.0078254532224 or 1756.870821951654<x_axis_fourier[i]<2147.014728686645 or 2647.997425964564<x_axis_fourier[i]<2876.613443905255: #a
        # if 177.979291060147<x_axis_fourier[i]<3473.68828737193:#bbbb
        # if 306.9553857724592<x_axis_fourier[i]<631.5177500146294 or 1647.020165966535<x_axis_fourier[i]<1880.2260188067203 or 2193.599524690138<x_axis_fourier[i]<2957.966758555921:#s 
        # if 283.12034785303433<x_axis_fourier[i]<519.0975521471505 or 1875.7450300388546<x_axis_fourier[i]<2082.413158818551 or 2339.051267675256<x_axis_fourier[i]<2791.5652675408687:#k
        # if 360.17867542370516<x_axis_fourier[i]<1005.7052871429231 or 1046.0119725859554<x_axis_fourier[i]<1816.9940795816058 or 1960.9273843932408<x_axis_fourier[i]<2940.183408830905:#k

        # if 222.04253766979843<x_axis_fourier[i]<749.3874583112313 or 1014.7679833589643<x_axis_fourier[i]<1818.6636587276323 or 2784.0546275102342<x_axis_fourier[i]<3479.3615241537846:#l
        # if 547.121889210727645<x_axis_fourier[i]<569.2131181562775 or 1070.61997668086907<x_axis_fourier[i]<1567.54186240030953 or 2431.7734520513633<x_axis_fourier[i]<2594.4337627330127:#l
        # if 534.7895784524468<x_axis_fourier[i]<620.4218811881244 or 931.8167160124324<x_axis_fourier[i]<1664.617624560184 or 2431.0230130175396<x_axis_fourier[i]<2723.3337594189907:#l
        # if 676.9690150372103<x_axis_fourier[i]<1276.9690150372103 or 1386.9831109673835<x_axis_fourier[i]<2186.9831109673835 or 2700.135922108039<x_axis_fourier[i]<3150.135922108039:#l
       
        
        if 250<x_axis_fourier[i]<850 or 1500<x_axis_fourier[i]<2350 or 2100<x_axis_fourier[i]<3100:#f
        # if 660<x_axis_fourier[i]<3050:#ae
            filtered.append(fft_out[i]*0.01)
            # filtered_out.append(fft_out[i])
        else:
            # filtered_out.append(0)
            filtered.append(fft_out[i])
        y=(y1[i]*y2[i]*y3[i])
        filtered_out.append(fft_out[i]*y)
    
    inverse=(np.fft.ifft(filtered,))
    inverse_out=(np.fft.ifft(filtered_out,))
    # inverse_minus_letter=np.real(np.fft.ifft(filtered_with_letter,))

    audio_fig.add_trace(go.Scatter(x=vowels2.variabls.uploaded_audio_time, y= vowels2.variabls.uploaded_audio_Yamp,name='real'))
    freq_fig.add_trace(go.Scatter(x=x_axis_fourier, y=abs_fft_out,name='fft'))
    freq_fig.add_trace(go.Scatter(x=x_axis_fourier, y=y1*y2*y3,name='normal distrbution'))
    # freq_fig.add_trace(go.Scatter(x=x_axis_fourier, y=y1,name='normal1 distrbution'))
    # freq_fig.add_trace(go.Scatter(x=x_axis_fourier, y=np.abs(filtered_with_letter),name='fft-letter'))
    audio_fig.add_trace(go.Scatter(x=vowels2.variabls.uploaded_audio_time*2, y= np.real(inverse),name='inverse'))
    audio_fig.add_trace(go.Scatter(x=vowels2.variabls.uploaded_audio_time*2, y= np.real(inverse_out),name='inverse out'))
    freq_fig.add_trace(go.Scatter(x=x_axis_fourier, y= np.abs(filtered_out),name='filtered normal'))
    freq_fig.add_trace(go.Scatter(x=x_axis_fourier, y=np.abs(filtered),name='filtered fft'))
    # audio_fig.add_trace(go.Scatter(x=vowels2.variabls.uploaded_audio_time*2, y= signal_temporary_amplitude,name='inverse w phase'))
    # audio_fig.add_trace(go.Scatter(x=vowels2.variabls.uploaded_audio_time*2, y= inverse_minus_letter,name='inverse wo letter'))

    # time_axis =np.arange(0,len(inverse))
    data={'time':vowels2.variabls.uploaded_audio_time,
            'value':vowels2.variabls.uploaded_audio_Yamp}
    for i in range(len(vowels2.variabls.uploaded_audio_Yamp)):
        data=[vowels2.variabls.uploaded_audio_time[i],vowels2.variabls.uploaded_audio_Yamp[i]]
        st.session_state.data.append(data)
    st.write('loop')
    wavio.write("myfile.wav", inverse, sampFreq//2, sampwidth=1)
    st.audio("myfile.wav", format='audio/wav')

    # wavio.write("myfileoutnoletter.wav", inverse_minus_letter, sampFreq//2, sampwidth=1)
    # st.audio("myfileoutnoletter.wav", format='audio/wav')
    st.write('normal')
    wavio.write("myfileout.wav", inverse_out, sampFreq//2, sampwidth=1)
    st.audio("myfileout.wav", format='audio/wav')

    st.write(freq_fig)
    st.write(audio_fig)
    # main()

