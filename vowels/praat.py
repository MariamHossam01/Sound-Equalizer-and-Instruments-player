import parselmouth 
from parselmouth import praat
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math

testfile = "C:\\Users\\hp\\Downloads\\ttsMP3 l.wav"

sound = parselmouth.Sound(testfile)

f0min=75
f0max=300
pointProcess = praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

formants = praat.call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
print(formants)

numPoints = praat.call(pointProcess, "Get number of points")
f1_list = []
f2_list = []
f3_list = []
for point in range(0, numPoints):
    point += 1
    t = praat.call(pointProcess, "Get time from index", point)
    f1 = praat.call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
    f2 = praat.call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
    f3 = praat.call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
    f1_list.append(f1)
    f2_list.append(f2)
    f3_list.append(f3)
f1_list= np.array(f1_list)[~np.isnan(f1_list)]
f2_list= np.array(f2_list)[~np.isnan(f2_list)]
f3_list= np.array(f3_list)[~np.isnan(f3_list)]
print('********************************')
# print(f1_list)
print(min(f1_list))
print(max(f1_list))
mean1 = sum(f1_list) / len(f1_list)
variance = sum([((x - mean1) ** 2) for x in f1_list]) / len(f1_list)
sigma1 = variance ** 0.5
print("Mean1 : " + str(mean1)+" Standard deviation : " + str(sigma1))


print('********************************')
print(min(f2_list))
print(max(f2_list))
mean2 = sum(f2_list) / len(f2_list)
variance = sum([((x - mean2) ** 2) for x in f2_list]) / len(f2_list)
sigma2 = variance ** 0.5
print("Mean2 : " + str(mean2)+" Standard deviation : " + str(sigma2))
print('********************************')
print(min(f3_list))
print(max(f3_list))
mean3= sum(f3_list) / len(f3_list)
variance = sum([((x - mean3) ** 2) for x in f3_list]) / len(f3_list)
sigma3 = variance ** 0.5
print("Mean3 : " + str(mean3)+" Standard deviation : " + str(sigma3))


x = np.linspace(0, 10000, 10)
y1=stats.norm.pdf(x, mean1, sigma1)
y2=stats.norm.pdf(x, mean2, sigma2)*50
y3=stats.norm.pdf(x, mean3, sigma3)*1000

# plt.plot(x, y1)
# plt.show()
# plt.plot(x, y2)
# plt.show()
# plt.plot(x,y3)
# plt.show()
# plt.plot(x, y1+y2+y3)
# plt.title('sum')
# plt.show()
# print(f1_list[-1])
# print(type(f2_list))
# print(f2_list[-1])
# print(f3_list)
# print(f3_list[-1])
