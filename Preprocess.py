import numpy as np
import scipy.stats as stats
from scipy import signal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import seaborn as sns
from scipy import optimize
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


"""
         1) SIGNAL DETREND AND NORMALIZATION 
         Note : Two options of normalization (linear - exponential)

 """
datos = np.swapaxes(data, 0, 1)
    
def NormF(datos):
    baseline = np.amin(datos[:, :4000], -1)[:, None]  # Until where we take the baseline activity
    return datos / baseline  # Treatment, cells, time

def linear_detrend(datos):
    X = np.arange(0, len(datos[0, :]))  # Length of the window (fragment)
    XX = X[None, :] * np.ones((datos.shape[-2], 1))  
    y = np.ravel(datos)
    slope, inter, _, _, _ = stats.linregress(x, y)
    t = np.arange(0, datos.shape[-1])
    trends = np.array(inter + slope * t)
    return datos - trends[None, :]

def exponential_detrend(datos):
    X = np.arange(0, len(datos[0, :]))  # Length of the window (fragment)
    XX = X[None, :] * np.ones((datos.shape[-2], 1))  # Cells, length of the window
    x = np.ravel(XX)
    y = np.ravel(datos)

    # Fit an exponential curve to the data
    log_y = np.log(y)
    slope, inter, _, _, _ = stats.linregress(x, log_y)
    t = np.arange(0, datos.shape[-1])
    exponential_trends = np.exp(inter + slope * t)

    return datos - exponential_trends[None, :]

b, a = signal.bessel(2, 0.3, btype='lowpass')  # Filter order 3
datosfilt = signal.filtfilt(b, a, datos, axis=-1)

### Choose you detrend format ###

# 1) Apply linear detrending
# datosNorm_linear = linear_detrend(NormF(datos))

# 2) Apply exponential detrending
datosNorm_exponential = exponential_detrend(NormF(datos))
datosNormFilt = NormF(datosfilt)  # Without the detrend function, it fits the signal better

dt = 0.3
time = np.arange(0, dt * datosNorm_exponential.shape[-1], dt)  # Seconds

cells = len(datosNorm_exponential)

"""
         2) SIGNAL DECOMPOSITION TO SEPARATE THE SPECTRAL CLUSTER OF OSCILLATIONS
                         Signal Filter band pass
"""
Frames = 600
    
datosNorm = datosNorm_exponential[:,:Frames]
datosNormFilt = datosNormFilt[:,:Frames]

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Choose the Frequency band to evaluate - Calcium transitories goes from mHz order
lowcut = 0.0001   ### 0.0001
highcut = 0.9
frames = int(len(datosNorm[0,:]))

fs = 2
cells = len(datosNorm_exponential)

filtered = []
for j in range(cells):
   filtered.append(butter_bandpass_filter(datosNorm_exponential[j,:], lowcut, highcut, fs, order = 3)) #entre 3 y 5 order
DataFiltBPT = np.array(filtered).reshape(cells,frames) # solo un electrodo



"""
        3)  SPECTROGRAM COMPUTATION
        ATTENTION : change the values of spectrogram size, it will depend on 
                    the length your signal.
                
                
"""
# calculate all the spectrogram values

freq = np.zeros((129,cells)) # 1D        
spec = np.zeros ((129,58,cells)) # 2D
time = np.zeros((58,cells)) 

framesample = int(len(datosNorm_exponential[0,:Frames])*2)

plt.figure()
for i in range(cells):
  spec[:,:,i],freq[:,i],time[:,i],_= plt.specgram(datosNorm_exponential[i,:Frames],              
                                                  Fs = 2, noverlap = 250, 
                                                  cmap = "seismic")             
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.5)
fig.suptitle("First 10 time series", fontsize=12, y=0.95)


signals = 10
axs = axs.ravel()
for d in range(10):
    axs[d].pcolormesh(time[:,d], freq[:,d], np.log10(spec[:,:,d]), cmap ='jet')
    axs[d].set_ylim(0,2)
    axs[d].set_title(str(d))

