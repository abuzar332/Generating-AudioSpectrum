#Importing the fft and inverse fft functions from fftpackage
from scipy.fftpack import fft
import scipy
from scipy.fftpack import ifft
import numpy as np
#create an array with random n numbers
x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])

#Applying the fft function
y = fft(x)
print(y)
yinv = ifft(y)
print(yinv)
from scipy.io import wavfile # scipy library to read wav files

AudioName = "mainWaveFile1.wav" # Audio File
fs, Audiodata = wavfile.read(AudioName)

# Plot the audio signal in time
import matplotlib.pyplot as plt

plt.plot(Audiodata)
plt.title('Audio signal in time',size=16)

# spectrum
from scipy.fftpack import fft # fourier transform
n = len(Audiodata) 
AudioFreq = fft(Audiodata)
AudioFreq = AudioFreq[0:int(np.ceil((n+1)/2.0))] #Half of the spectrum
MagFreq = np.abs(AudioFreq) # Magnitude
MagFreq = MagFreq / float(n)
# power spectrum
MagFreq = MagFreq**2
if n % 2 > 0: # ffte odd 
    MagFreq[1:len(MagFreq)] = MagFreq[1:len(MagFreq)] * 2
else:# fft even
    MagFreq[1:len(MagFreq) -1] = MagFreq[1:len(MagFreq) - 1] * 2 

plt.figure()
freqAxis = np.arange(0,int(np.ceil((n+1)/2.0)), 1.0) * (fs / n);
plt.plot(freqAxis/1000.0, 10*np.log10(MagFreq)) #Power spectrum
plt.xlabel('Frequency (kHz)'); plt.ylabel('Power spectrum (dB)');


#Spectrogram
from scipy import signal
N = 512 #Number of point in the fft
f, t, Sxx = signal.spectrogram(Audiodata, fs,window = signal.blackman(N),nfft=N)

f, ax = plt.subplots(figsize=(4.8, 2.4))
ax.pcolormesh(t, f,10*np.log10(Sxx))
ax.set_ylabel('Frequency [kHz]')
ax.set_xlabel('Time [s]');


plt.figure()
plt.pcolormesh(t/2, f/1000,10*np.log10(Sxx)) # dB spectrogram
#plt.pcolormesh(t, f,Sxx) # Lineal spectrogram
plt.ylabel('Frequency [kHz]')
plt.xlabel('Time [seg]')
plt.title('Spectrogram with scipy.signal',size=16);
input("Press Enter to continue...")