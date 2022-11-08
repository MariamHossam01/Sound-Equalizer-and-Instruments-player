import numpy as np

class variabls:
    
    current_slider_values=np.zeros(50)
    labels_values=np.zeros(50)
    ranges_values=np.zeros(50)
    uploaded_audio_time=[]
    uploaded_audio_Yamp=[]

def set_labels(mode_selector):
    print('set_labels')