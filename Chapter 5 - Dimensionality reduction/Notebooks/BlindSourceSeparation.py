"""
Created on Tue May 29 17:11:25 2018

@author: Emma C.  Robinson

Blind source separation of sound waves:
    
    based on the Scikit-Learn Tutorial 
    http://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html
    
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

from sklearn.decomposition import FastICA, PCA

# #############################################################################

# load mixed audio files
samplingRate, signal1 = wavfile.read('Data/mix1.wav')
print("Sampling rate = ", samplingRate)
print("Data type is ", signal1.dtype)

samplingRate, signal2 = wavfile.read('Data/mix2.wav')
print("Sampling rate = ", samplingRate)
print("Data type is ", signal2.dtype)

# load source audio files for comparison
samplingRate, source1 = wavfile.read('Data/source1.wav')
samplingRate, source2 = wavfile.read('Data/source2.wav')

# combine files into one matrix X
X= np.stack((signal1,signal2),axis=0).T # mixed signals
S= np.stack((source1,source2),axis=0).T # mixed signals

###################### PLOT AUDIO SIGNALS ##############################

# plot MIXED signals
f, (ax1,ax2)=plt.subplots(2,1)

ax1.plot(X[:,0],'k')
ax2.plot(X[:,1],'k')
f.suptitle('Mixed signals')
# plot source signals
f, (ax1,ax2)=plt.subplots(2,1)

ax1.plot(S[:,0],'k')
ax2.plot(S[:,1],'k')
f.suptitle('Souce Signals')

######################## PERFORM DECOMPOSITION ##########################

# Compute ICA, to extract two components
ica = FastICA(n_components=2, whiten=True,fun='exp')
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# For comparison, compute PCA
pca = PCA(n_components=2)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

# save output
wavfile.write('Data/unmix_FastICA1.wav',samplingRate,S_[:,0])
wavfile.write('Data/unmix_FastICA2.wav',samplingRate,S_[:,1])

wavfile.write('Data/unmix_PCA1.wav',samplingRate,S_[:,0])
wavfile.write('Data/unmix_PCA2.wav',samplingRate,S_[:,1])

# #################### PLOT RESULTS #####################################

# plot ICA components
f, (ax1,ax2)=plt.subplots(2,1)

ax1.plot(S_[:,0],'k')
ax2.plot(S_[:,1],'k')
f.suptitle('ICA components')

# plot source signals
f, (ax1,ax2)=plt.subplots(2,1)

ax1.plot(H[:,0],'k')
ax2.plot(H[:,1],'k')
f.suptitle('PCA components')
