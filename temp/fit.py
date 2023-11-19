import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

time0, PNS_x0 = np.loadtxt("PNS_position_vs_time_original.txt", unpack=True) 
time, PNS_x = np.loadtxt("PNS_position_vs_time.txt", unpack=True) 

PNS_x0 = PNS_x0*1.477
PNS_x = PNS_x*1.477

xdata = np.asarray(time)
ydata = np.asarray(PNS_x)


start_time_for_fit = 150
end_time_for_fit = 160

start_idx = np.where(xdata > start_time_for_fit)[0][0]
end_idx = np.where(xdata < end_time_for_fit)[0][-1]
print("start_idx = {}, end_idx = {}\n", start_idx, end_idx)

#print("Before shape: ", xdata.shape, ydata.shape)
xdata = xdata[start_idx:end_idx]
ydata = ydata[start_idx:end_idx]
#print("After shape:", xdata.shape, ydata.shape)
#exit(0)

# Define the function
def Linear(x, A, B):
    y = A*x + B
    return y
    
parameters, covariance = curve_fit(Linear, xdata, ydata)
print("parameters", parameters)
print("covariance", covariance)
fit_A = parameters[0]
fit_B = parameters[1]

print("PNS_x = A*time + B: A = {}, B = {}\n".format(fit_A, fit_B))

fit_PNS_x = Linear(xdata, fit_A, fit_B)
fit_time =  xdata

plt.figure()
plt.xlabel("time (ms)")
plt.ylabel("x-position of PNS (km)")
plt.title("speed in fit range = {} km/s".format(round(fit_A*1000)))

plt.scatter(time0, PNS_x0, s=10, marker="o", facecolors='none', edgecolors='b')
plt.scatter(time, PNS_x, s=10, marker=".", facecolors='r', edgecolors='r')
plt.plot(fit_time, fit_PNS_x, linewidth=3, color="black")

plt.show()
