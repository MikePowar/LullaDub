
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
from tqdm import tqdm
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import wave

def plot_signals(signals):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, 
        figsize=(20,5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(2):
        for y in range(2):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fft(fft):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True,
        figsize=(20,5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for x in range(2):
        for y in range(2):
            data = list(fft.values())[i]
            Y, freq = data[0], data[1]
            axes[x,y].set_title(list(fft.keys())[i])
            axes[x,y].plot(freq, Y)
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True,
        figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(2):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i], cmap='hot',
                interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False,
                             sharey=True, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(2):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i], cmap='hot',
                interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1

def calc_fft(signal, rate):
    signal_length = len(signal)
    inverse_rate = 1/rate
    frequency = np.fft.rfftfreq(signal_length, d=inverse_rate)
    magnitude = abs(np.fft.rfft(signal)/signal_length)
    return (magnitude, frequency)

data_frame = pd.read_csv("cries.csv")
data_frame.set_index("fname", inplace=True)

for file in data_frame.index:
    print(file)
    rate, signal = wavfile.read("wavfiles/"+file)
    data_frame.at[file, 'length'] = signal.shape[0]/rate
    #d print(data_frame)

classes = list(np.unique(data_frame.label))
class_dist = data_frame.groupby(["label"])['length'].mean()
#d print("\n", class_dist)

figure, axis = plt.subplots()
axis.set_title("class distribution", y=1.08)
axis.pie(class_dist, labels=class_dist.index, autopct="%1.1f%%", shadow=False, 
    startangle=90)
axis.axis("equal")
plt.show()
data_frame.reset_index(inplace=True)

signals = dict()
fast_fourier_transform = dict()
filter_bank_energies = dict()
mfccs = dict()

for label in classes:
    file = data_frame[data_frame.label == label].iloc[0, 0]
    filters = 26
    ceptrals = int(filters/2)
    rate = None
    with wave.open("wavfiles/"+file, "rb") as wav:
        rate = wav.getframerate()
    fourier_transforms = int(rate/40)
    signal, rate = librosa.load("wavfiles/"+file, sr=rate)
    signals[label] = signal
    fast_fourier_transform[label] = calc_fft(signal, rate)
    filter_bank = logfbank(signal[:rate], rate, nfilt=filters,
        nfft=fourier_transforms).T
    filter_bank_energies[label] = filter_bank
    mel = mfcc(signal[:rate], rate, numcep=ceptrals, nfilt=filters,
        nfft=fourier_transforms).T
    mfccs[label] = mel

plot_signals(signals)
plt.show()

plot_fft(fast_fourier_transform)
plt.show()

plot_fbank(filter_bank_energies)
plt.show()

plot_mfccs(mfccs)
plt.show()

if len(os.listdir("clean")) == 0:
    threshold = 0.0005
    for file in tqdm(data_frame.fname):
        signal, rate = librosa.load("wavfiles/"+file, sr=16000)
        # verify non cleaned data is not usable
        #mask = envelope(signal, rate, threshold)
        #wavfile.write(filename="clean/"+file, rate=rate, data=signal[mask])
        wavfile.write(filename="clean/"+file, rate=rate, data=signal)
