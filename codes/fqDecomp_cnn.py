"""
Created on Sat Nov 16 14:36:24 2024

denoise by
putting synthetic seismogram into NN
- test for different fq decomposition techniques

@author: asifashraf
"""

# --- User-configurable inputs ---
N_total = 100         # Total samples to feed into network
N_train = 80          # Total samples to train the network, the rest will go to validating the network

# --- Import necessary libraries ---
from PyEMD import EMD               # Empirical Mode Decomposition (EMD)
import numpy as np                  # Numerical operations on arrays
from scipy import signal            # Signal processing functions (e.g., spectrogram)
from scipy.signal import chirp      # Generate chirp signals

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.signal import hilbert                # Hilbert transform (for envelope)
from scipy.stats import binned_statistic_2d     # 2D binning (for custom spectrograms)

import tensorflow as tf
from tensorflow.keras import layers, models     # Keras API: layers and model creation

import pywt                                     # Wavelet transforms (for CWT)
from toolbox import spectrogram, cua_envelope   # Custom spectrogram and envelope functions

# PATH MANAGEMENT — canonicalise important directories
import os
script_dir  = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(script_dir)
inp_dir     = os.path.join(parent_dir, 'input')
out_dir     = os.path.join(parent_dir, 'output')

#####
# Don't edit below this line --------------------------------------------------

# Time Axis
dt         = 0.005      # Sampling interval in second
start_time = 0          # Start time in seconds
end_time   = 50         # End time in seconds
t          = np.arange(start_time, end_time, dt)

# MAKE SEISMOGRAM ENVELOP (CUA ENVELOP)
mag        = np.random.uniform(3, 8, size = N_total+3)          # magnitude
dist_km    = np.random.uniform(20, 1000, size = N_total+3)      # epicentral distance
ptime      = np.random.uniform(5, 20, size = N_total+3)         # P arrival time
stime      = np.random.uniform(20, 40, size = N_total+3)        # S arrival time

# CHIRPs parameter
method     = np.random.choice(['linear', 'logarithmic'], size = N_total+3)
f1         = np.random.uniform(0, 10, size = N_total+3)
f2         = np.random.uniform(0, 30, size = N_total+3)
amplitude  = np.random.uniform(1, 10, size = N_total+3)

# Function
S   = []      # S - 'Seismograms'; to store base signal without any noise
S_N = []      # S_N - 'Seismogram with noise'
# loop to make synthetic seismograms with and without noise
print('Building Synthetic Seismograms ...')
for i in range(N_total):
    
    print( '    Seismogram: (' +str(i+1) + '/' + str(N_total) + ')')
    
    # individual chirp signals
    ch1 = amplitude[i]*chirp(t, f1[i], t.max(), f2[i], method = method[i])
    ch2 = amplitude[i+1]*chirp(t, f1[i+1], t.max(), f2[i+1], method = method[i+1])
    ch3 = amplitude[i+2]*chirp(t, f1[i+2], t.max(), f2[i+2], method = method[i+2])
    ch4 = amplitude[i+3]*chirp(t, f1[i+3], t.max(), f2[i+3], method = method[i+3])
    
    # sum the chirp signals
    ch_total = ch1 + ch2 + ch3 + ch4
    
    # apply seismogram envelop
    env = cua_envelope(mag[i], dist_km[i], t, ptime[i], stime[i])
    
    # Seismogram
    ssgm = ch_total*env     # seismogram without noise
    
    # NOISE
    Namp = np.random.uniform(0, np.max(abs(ssgm))/7, 3)
                    # parting the maximum amplitude in 3 ways, so
                    # total noise amplitude is always less than the signal
    N1   = np.random.normal(0, Namp[0], size = ssgm.shape)
    N2   = np.random.normal(0, Namp[1], size = ssgm.shape)
    N3   = np.random.normal(0, Namp[2], size = ssgm.shape)

    N_ttl  = N1 + N2 + N3
    
    ssgm_N = ssgm + N_ttl   # seismogram with noise

    # plt.close('all')
    # plt.figure(1)
    # plt.subplot(2,1,1)
    # plt.plot(t, ssgm)
    # plt.title('seismogram without noise')
    # plt.subplot(2,1,2)
    # plt.plot(t, ssgm_N)
    # plt.title('seismogram with noise')
    # plt.tight_layout()
    # plt.savefig((out_dir + '/' + str(i) + '.png'), format = 'png')
    
    S.append(ssgm)
    S_N.append(ssgm_N)
    
print('Finished: (Synthetic seismograms)')

train_clean, val_clean = S[:N_train],   S[N_train:]
train_noisy, val_noisy = S_N[:N_train], S_N[N_train:]

## Frequency Decomposition

# FFT
    # fast fourier transformation
n_fq = (len(t)/(end_time - start_time)) / 2   # Nyquist frequency
fs = 2*n_fq + n_fq/10                         # Sampling frequency

# WT
    # wavelet transformation
wavelet = 'morl'
scales = np.arange(1, 500)

# EMD-HHT
    # emperical mode decomposition with hilbert huang transform
max_period = 50
min_period = 2*dt
n_bins     = 800
period_bins = np.logspace(np.log10(min_period), np.log10(max_period), n_bins) #logarithmically spaced period bins

# empty arrays to store amplitude of the frequency decomposition
ampSpec_s = []; ampSpec_sN = []
ampFFT_s  = []; ampFFT_sN  = []
coeff_s   = [];  coeff_sN  = []
print('Frequency Decomposition ...')
# loop to do frequency decomposition 
for i in range(len(S)):
    print('     FQ decomp (' + str(i+1) + '/' + str(len(S)) + ') ...')
    s  = S[i]
    sN = S_N[i]

    # EMD
    timeSpec, periodSpec, ampSpec = spectrogram(t, s, dt, [min_period, max_period], period_bins = period_bins, return_imfs=False)
    ampSpec = ampSpec[..., np.newaxis] # make compatible with CNN input dimensiton
    ampSpec_s.append(ampSpec)
    timeSpec, periodSpec, ampSpec = spectrogram(t, sN, dt, [min_period, max_period], period_bins = period_bins, return_imfs=False)
    ampSpec = ampSpec[..., np.newaxis] # make compatible with CNN input dimensiton
    ampSpec_sN.append(ampSpec)

    # FFT
    timeFFT  , freqFFt,   ampFFT  = signal.spectrogram(s, fs = fs, nperseg = 400, noverlap = 400 - 1)
    ampFFT = ampFFT[..., np.newaxis] # make compatible with CNN input dimensiton    
    ampFFT_s.append(ampFFT)
    timeFFT  , freqFFt,   ampFFT  = signal.spectrogram(sN, fs = fs, nperseg = 400, noverlap = 400 - 1)
    ampFFT = ampFFT[..., np.newaxis] # make compatible with CNN input dimensiton    
    ampFFT_sN.append(ampFFT)

    # WT
    coeff, wvfq = pywt.cwt(data = s, scales = scales, wavelet = wavelet, sampling_period=1/(fs))
    coeff = coeff[..., np.newaxis] # make compatible with CNN input dimensiton        
    coeff_s.append(coeff)
    coeff, wvfq = pywt.cwt(data = sN, scales = scales, wavelet = wavelet, sampling_period=1/(fs))
    coeff = coeff[..., np.newaxis] # make compatible with CNN input dimensiton    
    coeff_sN.append(coeff)

print('Finished: (Frequency Decomposition)')

## plot-test to check frequency decomposition
'''
ij = 1 # sample number to check
plt.figure(1)

plt.subplot(1, 3, 1)
plt.pcolormesh(timeSpec_s[ij], 1/periodSpec_s[ij], np.abs(ampSpec_s[ij]), cmap = 'hot')
plt.title("EMD")
plt.ylim(0, 25)

plt.subplot(1, 3, 2)
t_wt = np.linspace(t[0], t[len(t)-1], coeff_s[ij].shape[1])
plt.pcolormesh(t_wt, wvfq_s[ij], coeff_s[ij], cmap = 'hot')
plt.title('WT')
plt.ylim(0, 25)

plt.subplot(1, 3, 3)
t_ft = np.linspace(t[0], t[len(t)-1], ampFFT_s[ij].shape[0])
plt.pcolormesh(freqFFt_s[ij], t_ft, ampFFT_s[ij], cmap = 'hot')
plt.title('FFT')
plt.ylim(0, 25)

plt.figure(2)

plt.subplot(1, 3, 1)
plt.pcolormesh(timeSpec_sN[ij], 1/periodSpec_sN[ij], np.abs(ampSpec_sN[ij]), cmap = 'hot')
plt.title("EMD")
plt.ylim(0, 25)

plt.subplot(1, 3, 2)
t_wt = np.linspace(t[0], t[len(t)-1], coeff_sN[ij].shape[1])
plt.pcolormesh(t_wt, wvfq_sN[ij], coeff_sN[ij], cmap = 'hot')
plt.title('WT')
plt.ylim(0, 25)

plt.subplot(1, 3, 3)
t_ft = np.linspace(t[0], t[len(t)-1], ampFFT_sN[ij].shape[0])
plt.pcolormesh(freqFFt_sN[ij], t_ft, ampFFT_sN[ij], cmap = 'hot')
plt.title('FFT')
plt.ylim(0, 25)

plt.tight_layout()
'''
# Define CNN model function
def build_cnn_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder (Downsampling)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding = 'same')(x)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding = 'same')(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding = 'same')(x)

    # Decoder (Upsampling)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2DTranspose(8, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Output layer
    x = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    # Cropping layer to ensure the output matches the input dimensions exactly
    x = layers.Cropping2D(cropping=((0, x.shape[1] - input_shape[0]), (0, x.shape[2] - input_shape[1])))(x)

    # Create the model
    model = models.Model(inputs, x)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    return model

# DIVIDE DATASETS (training and validation)
# emd
emd_train_noisy, emd_val_noisy = np.array(ampSpec_sN[:N_train]), np.array(ampSpec_sN[N_train:])  # Noisy
emd_train_clean, emd_val_clean = np.array(ampSpec_s[:N_train]),  np.array(ampSpec_s[N_train:])   # Clean
# fft
fft_train_noisy, fft_val_noisy = np.array(ampFFT_sN[:N_train]),  np.array(ampFFT_sN[N_train:])   # Noisy
fft_train_clean, fft_val_clean = np.array(ampFFT_s[:N_train]),   np.array(ampFFT_s[N_train:])    # Clean
# wavelet
wv_train_noisy,  wv_val_noisy  = np.array(coeff_sN[:N_train]),   np.array(coeff_sN[N_train:])    # Noisy
wv_train_clean,  wv_val_clean  = np.array(coeff_s[:N_train]),    np.array(coeff_s[N_train:])     # Clean

# MODEL TRAINING
# emd
print("Training EMD Model...")
emd_model = build_cnn_model(input_shape=(emd_train_noisy[0].shape))
emd_history = emd_model.fit(emd_train_noisy, emd_train_clean, epochs=20, batch_size=N_train,
                            validation_data=(emd_val_noisy, emd_val_clean))
# fft
print("Training FFT Model...")
fft_model = build_cnn_model(input_shape=(fft_train_noisy[0].shape))
fft_history = fft_model.fit(fft_train_noisy, fft_train_clean, epochs=20, batch_size=N_train,
                            validation_data=(fft_val_noisy, fft_val_clean))

# wavelet
print("Training Wavelet Model...")
wavelet_model = build_cnn_model(input_shape=(wv_train_noisy[0].shape))
wavelet_history = wavelet_model.fit(wv_train_noisy, wv_train_clean, epochs=20, batch_size=N_train,
                                    validation_data=(wv_val_noisy, wv_val_clean))

# RESULT PLOT
#'''
plt.figure(figsize=(12, 8))

# FFT Validation Loss
plt.plot(fft_history.history['val_loss'], label='FFT Validation Loss')

# Wavelet Validation Loss
#plt.plot(wavelet_history.history['val_loss'], label='Wavelet Validation Loss')

# EMD Validation Loss
plt.plot(emd_history.history['val_loss'], label='EMD Validation Loss')

plt.title('Comparison of Validation Loss for Different Decomposition Methods')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()

# Optional: Compare MAE as Well
plt.figure(figsize=(12, 8))

# FFT Validation MAE
plt.plot(fft_history.history['val_mae'], label='FFT Validation MAE')

# Wavelet Validation MAE
plt.plot(wavelet_history.history['val_mae'], label='Wavelet Validation MAE')

# EMD Validation MAE
plt.plot(emd_history.history['val_mae'], label='EMD Validation MAE')

plt.title('Comparison of Validation MAE for Different Decomposition Methods')
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.legend()
plt.show()
#'''