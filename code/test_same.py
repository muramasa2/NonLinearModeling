import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from glob import glob
from scipy.fftpack import fft, ifft
from scipy import hamming, hanning
import soundfile as sf

####################
# make signal data #
####################
fs = 44100
L = 60
f = 1200

t = np.arange(0,L,1/44100)
amp = 1
base_signal = amp * np.sin(2*np.pi*f*t)

num = 10 #何次高調波まで見るか

mode = 'all' # all:全ての高調波含む, even:偶数次高調波のみ, odd:奇数時高調波のみ
in_len = 512
if mode == 'all':
    start = 2
    step = 1
    # save_path = './figure/make_all_non_lineardist_signal.jpg'
else:
    step = 2
    if mode == 'even':
        start = 2
        # save_path = './figure/make_even_non_lineardist_signal.jpg'
    elif mode == 'odd':
        start = 3
        # save_path = './figure/make_odd_non_lineardist_signal.jpg'

# non_lin_dist = [( 4.472 * 10**(-6)/ 2**(i-1)) * np.sin(2*np.pi*(i)*f*t) for i in range(start,num+1,step)]
non_lin_dist = [( 100000 * 10**(-6)/ 2**(i-1)) * np.sin(2*np.pi*(i)*f*t)
# non_lin_dist = [( 1/ (2*(i-1))) * np.sin(2*np.pi*(i)*f*t)
    for i in range(start,num+1,step)] #このくらいから歪みを知覚できる
dist_signal = base_signal+sum(non_lin_dist)

sf.write(f'D:/masas/Documents/Script/NonLinDistDenoise/create_dist_wave/dist_all_{f}_signal.wav',
        dist_signal, 44100,subtype='PCM_16') # 16bit 44.1kHz


plt.figure(1,figsize=(10,5))
plt.plot(t,dist_signal,'r',label='distoted')
plt.plot(t,base_signal,'b',label='clean')
plt.xlim([0,0.002])
plt.legend()
plt.show()

dist_signal, fs = sf.read(f'D:/masas/Documents/Script/NonLinDistDenoise/create_dist_wave/dist_all_{f}_signal.wav')

dist_signal.shape
base_signal.shape
dist_signal = dist_signal/max(dist_signal)
base_signal = base_signal/max(base_signal)


out_len = in_len

model_save_path = f'D:/masas/Documents/Script/NonLinDistDenoise/weight/same_{mode}_weight{in_len}_{out_len}.h5'

model = Sequential()
model.add(Conv1D(64, 8, padding='same', input_shape=(in_len, 1), activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Conv1D(64, 8, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same'))
model.add(Conv1D(32, 8, padding='same', activation='relu'))
model.add(MaxPooling1D(2, padding='same'))

model.add(Conv1D(32, 8, padding='same', activation='relu'))
model.add(UpSampling1D(2))
model.add(Conv1D(64, 8, padding='same', activation='relu'))
model.add(UpSampling1D(2))
model.add(Conv1D(64, 8, padding='same', activation='relu'))
model.add(UpSampling1D(2))
model.add(Conv1D(1, 8, padding='same', activation='tanh'))

model.compile(optimizer='adam', loss='mse')
model.summary()

model.load_weights(model_save_path)

start = 800
sheed = np.reshape(dist_signal[start:start+in_len], (1, in_len, 1))
# full_prediction = np.empty(1)

for i in range(600):
    res = model.predict(sheed)
    prediction = np.concatenate((sheed, res), axis=1)
    if i == 0:
        full_prediction = res
    else:
        full_prediction = np.concatenate((full_prediction, res), axis=1)

    # plt.figure()
    # plt.plot(range(len(prediction[0])), prediction[0], 'b', label='pred')
    # plt.plot(range(len(sheed[0])), sheed[0], 'r', label='input')
    # plt.legend()
    # print(i)

    sheed = np.reshape(dist_signal[(start+(i+1)*out_len):(start+(i+1)*out_len+in_len)], (1, in_len, 1))

predictor = np.reshape(full_prediction, (-1))

sf.write(f'D:/masas/Documents/Script/NonLinDistDenoise/result_wave/Conv1D/{f}_same_{mode}_weight{in_len}_{out_len}_signal.wav',predictor,44100,subtype='PCM_16') # 16bit 44.1kHz

plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid
plt.rcParams['figure.dpi'] = 300

# plt.figure(figsize=(15, 10)) # figure size in inch, 横×縦
plt.figure(1)
plt.plot(range(len(predictor)), dist_signal[start:start + len(predictor)],'r', linewidth=3, label='distoted')
plt.plot(range(len(predictor)), base_signal[start:start + len(predictor)],'b', linewidth=3, label='clean')
plt.plot(range(len(predictor)), predictor, 'y--', label='predict')
plt.xlim([100,195])
plt.xlabel('N[sample]')
plt.ylabel('Amplitude[V]')
plt.legend(loc='upper right')
plt.savefig(f'D:/masas/Documents/Script/NonLinDistDenoise/figure/Conv1D/{f}_same_{mode}_weight{in_len}_{out_len}_wave.jpg',
            bbox_inches="tight", pad_inches=0.05)



# plt.figure()
# plt.plot(range(len(predictor)), dist_signal[start:start + len(predictor)],'r', linewidth=3, label='distoted')
# plt.plot(range(len(predictor)), base_signal[start:start + len(predictor)],'b', linewidth=3, label='clean')
# plt.plot(range(len(predictor)), predictor, 'y--', label='predict')
# plt.xlabel('n[sample]')
# plt.ylabel('amp[V]')
# plt.legend(loc='upper right')
# plt.show()

def signal_fft(signal, N): #FFTするsignal長と窓長Nは同じサンプル数に固定する
    win = hanning(N) # 窓関数
    spectrum = fft(signal*win) # フーリエ変換
    spectrum_abs = np.abs(spectrum) # 振幅を元に信号に揃える
    half_spectrum = spectrum_abs[:int(N/2)]
    half_spectrum[0] = half_spectrum[0] / 2 # 直流成分（今回は扱わないけど）は2倍不要
    half_spectrum_dBV = 20*np.log10(half_spectrum)

    return spectrum, half_spectrum_dBV

# in_data = dist_signal
path = f'D:/masas/Documents/Script/NonLinDistDenoise/create_dist_wave/dist_{mode}_{f}_signal.wav'
# path = './dist_all_signal.wav'
in_data,fs = sf.read(path)
N = fs

# in_data = in_data*np.sqrt(2)
in_data = in_data[:N]
_, in_half_spectrum_dBV = signal_fft(in_data, len(in_data))
f1 = np.arange(0, fs/2, (fs/2)/in_half_spectrum_dBV.shape[0]) # 横軸周波数軸[Hz]
plt.semilogx(f1, in_half_spectrum_dBV)

out_data = predictor
# out_data = out_data*np.sqrt(2)
out_data = out_data[:N]
_, out_half_spectrum_dBV = signal_fft(out_data, len(out_data))
f2 = np.arange(0, fs/2, (fs/2)/out_half_spectrum_dBV.shape[0]) # 横軸周波数軸[Hz]
plt.semilogx(f2, out_half_spectrum_dBV)

plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid
plt.rcParams['figure.dpi'] = 300

plt.figure(figsize=(15, 10)) # figure size in inch, 横×縦
# plot
plt.semilogx(f1, in_half_spectrum_dBV, 'b')
plt.semilogx(f1, out_half_spectrum_dBV, 'r')
plt.xlim([1,22050])
plt.xlabel('Frequency[Hz]', fontsize=15)
plt.ylabel('Amplitude[dB]', fontsize=15)
plt.legend(['input(distorted)', 'output(corrected)'],loc='upper right',fontsize=15)
# save
plt.savefig(f'D:/masas/Documents/Script/NonLinDistDenoise/figure/Conv1D/{f}_same_{mode}_weight{in_len}_{out_len}_fft.jpg',
            bbox_inches="tight", pad_inches=0.05)
np.where(out_half_spectrum_dBV==max(out_half_spectrum_dBV))
np.where(in_half_spectrum_dBV==max(in_half_spectrum_dBV))


# def THD(spectrum, n, f):
#     V = 10**(spectrum[f]/20)
#     lin_V = []
#
#     for i in range(2,n+1):
#         lin_V.append(10**(spectrum[f*i]/20))
#
#     lin_V = np.array(lin_V)**2
#     thd = sum(lin_V)/V
#
#     return thd
#
# f = 1019
# in_thd = THD(in_half_spectrum_dBV, 10, f)
# out_thd = THD(out_half_spectrum_dBV, 10, f)
# print(20*np.log10(in_thd), 20*np.log10(out_thd))
