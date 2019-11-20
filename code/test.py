"""numpy module."""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from glob import glob
from natsort import natsorted
import soundfile as sf
from scipy.fftpack import fft, ifft
from scipy import hamming, hanning


####################
# make signal data #
####################
type = 0
mode = 'denoise'
structure = 'Conv1D'  # Conv1D or LSTM
reg = 'off'


input_paths = natsorted(glob('data/mono/NoFX/*'))

all_output_paths = natsorted(glob('data/mono/Distortion/*'))
output_paths = all_output_paths[type::3]

in_len = 1024
out_len = in_len
step = 64

input_data = []
output_data = []

wav_num = int(len(input_paths)*0.8)+100
# wav_num = 15
print('input:', input_paths[wav_num], 'output:', output_paths[wav_num])
in_signal, fs = sf.read(input_paths[wav_num])
out_signal, _ = sf.read(output_paths[wav_num])

if reg == 'on':
    in_max = max(in_signal)
    out_max = max(out_signal)
    in_signal = in_signal/max(in_signal)
    out_signal = out_signal/max(out_signal)

t = np.arange(0, (len(in_signal))/fs, 1 / fs)

plt.figure(1)
plt.plot(t, out_signal, 'r', label='distorted')
plt.plot(t, in_signal, 'b', label='clean')
plt.xlabel('time[s]')
plt.ylabel('amplitude[V]')
plt.xlim([0, 2])
plt.legend()

for n in range(int((len(in_signal)-(in_len))/step)):
    input_data.append(in_signal[int(n * step):int(n * step + in_len)])
    output_data.append(out_signal[int(n * step):int(n * step + out_len)])

input_data = np.array(input_data)
output_data = np.array(output_data)

input_data = input_data.reshape(-1, in_len, 1)
output_data = output_data.reshape(-1, out_len, 1)

if mode == 'modeling':
    testX = input_data
    testy = output_data
    label = ['pred_dist', 'true_dist', 'clean']


elif mode == 'denoise':
    testX = output_data
    testy = input_data
    label = ['pred_denoise', 'true_clean', 'distorted']

model_save_path = f'./weight/model_{structure}_dist_type{type}_weight{in_len}_{out_len}_{step}_reg{reg}_{mode}.h5'
model = Sequential()

if structure == 'Conv1D':
    model.add(Conv1D(64, 8, padding='same',
                     input_shape=(in_len, 1), activation='relu'))
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

elif structure == 'LSTM':
    model.add(LSTM(64, input_shape=(in_len, 1), return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(1, return_sequences=True))

model.compile(optimizer='adam', loss='mse')
model.summary()

model.load_weights(model_save_path)

for i in range(len(testX)):
    predy = model.predict(testX[i].reshape(1, in_len ,1))

    if i == 0:
        predict = predy
    else:
        predict = np.concatenate((predict, predy[:, in_len-step:]), axis=1)

input = in_signal[:len(predict[0])]
output = out_signal[:len(predict[0])]

if reg == 'on':
    predict = predict*out_max
    input = input*in_max
    output = output*out_max

plt.rcParams["font.size"] = 15  # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.direction'] = 'in'  # x axis in
plt.rcParams['ytick.direction'] = 'in'  # y axis in
plt.rcParams['axes.linewidth'] = 1.0  # axis line width
plt.rcParams['axes.grid'] = True  # make grid
plt.rcParams['figure.dpi'] = 300

plt.figure()
plt.plot(range(len(input)), output, 'g', linewidth=3, label=label[2], alpha=0.5)
plt.plot(range(len(input)), input, 'r', linewidth=3, label=label[1])
plt.plot(range(len(predict[0])), predict[0], 'b', linewidth=3, label=label[0], alpha=0.5)
plt.xlabel('N[sample]')
plt.ylabel('Amplitude[V]')
plt.legend(loc='upper left',bbox_to_anchor=(1.05, 1))
plt.savefig(f'./figure/{wav_num}_model_{structure}_dist_type{type}_{in_len}_{out_len}_{step}_reg{reg}_{mode}.jpg',
            bbox_inches="tight", pad_inches=0.05)


sf.write(f'./result_wave/{wav_num}_model_{structure}_dist_type{type}_{in_len}_{out_len}_{step}_reg{reg}_{mode}.wav',
        predict[0],44100,subtype='PCM_16') # 16bit 44.1kHz


def signal_fft(signal, N): #FFTするsignal長と窓長Nは同じサンプル数に固定する
    win = hanning(N) # 窓関数
    spectrum = fft(signal*win) # フーリエ変換
    spectrum_abs = np.abs(spectrum) # 振幅を元に信号に揃える
    half_spectrum = spectrum_abs[:int(N/2)]
    half_spectrum[0] = half_spectrum[0] / 2 # 直流成分（今回は扱わないけど）は2倍不要
    half_spectrum_dBV = 20*np.log10(half_spectrum)

    return spectrum, half_spectrum_dBV


path = f'./result_wave/{wav_num}_model_{structure}_dist_type{type}_{in_len}_{out_len}_{step}_reg{reg}_{mode}.wav'
out_data,fs = sf.read(path)
_, out_half_spectrum_dBV = signal_fft(out_data, len(out_data))
f2 = np.arange(0, fs/2, (fs/2)/out_half_spectrum_dBV.shape[0]) # 横軸周波数軸[Hz]
plt.semilogx(f2, out_half_spectrum_dBV)


in_data,fs = sf.read(output_paths[wav_num])
in_data = in_data[:len(out_data)]
_, in_half_spectrum_dBV = signal_fft(in_data, len(in_data))
f1 = np.arange(0, fs/2, (fs/2)/in_half_spectrum_dBV.shape[0])  # 横軸周波数軸[Hz]
plt.semilogx(f1, in_half_spectrum_dBV)



plt.rcParams["font.size"] = 15  # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.direction'] = 'in'  # x axis in
plt.rcParams['ytick.direction'] = 'in'  # y axis in
plt.rcParams['axes.linewidth'] = 1.0  # axis line width
plt.rcParams['axes.grid'] = True  # make grid
plt.rcParams['figure.dpi'] = 300

plt.figure(figsize=(15, 10))  # figure size in inch, 横×縦
# plot
plt.semilogx(f1, out_half_spectrum_dBV, 'r', label=label[0])
plt.semilogx(f1, in_half_spectrum_dBV, 'b', label=label[1], alpha=0.5)
plt.xlim([1,22050])
plt.xlabel('Frequency[Hz]', fontsize=15)
plt.ylabel('Amplitude[dB]', fontsize=15)
plt.ylim([-75,55])
plt.legend(loc='upper right',fontsize=15)
# save
plt.savefig(f'figure/fft_{wav_num}_model_{structure}_dist_type{type}_{in_len}_{out_len}_{step}_reg{reg}_{mode}.jpg',
            bbox_inches="tight", pad_inches=0.05)



sub = max(in_half_spectrum_dBV)-max(out_half_spectrum_dBV)
in_half_spectrum_dBV = in_half_spectrum_dBV - sub

plt.figure(figsize=(15, 10))  # figure size in inch, 横×縦
# plot
plt.semilogx(f1, out_half_spectrum_dBV, 'r', label=label[0])
plt.semilogx(f1, in_half_spectrum_dBV, 'b', label=label[1], alpha=0.5)
plt.xlim([1,22050])
plt.xlabel('Frequency[Hz]', fontsize=15)
plt.ylabel('Amplitude[dB]', fontsize=15)
plt.ylim([-75,55])
plt.legend(loc='upper right',fontsize=15)
# save
plt.savefig(f'figure/fix_fft_{wav_num}_model_{structure}_dist_type{type}_{in_len}_{out_len}_{step}_reg{reg}_{mode}.jpg',
            bbox_inches="tight", pad_inches=0.05)


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
