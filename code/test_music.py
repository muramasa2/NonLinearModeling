"""test denoise or modeling nonlinear noise."""
##################
# import library #
##################
import os
import argparse
import numpy as np
import soundfile as sf
from datetime import date
from scipy import hanning
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from keras.layers import CuDNNLSTM
from keras.models import Sequential
from keras.losses import mean_squared_error
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D, UpSampling1D
import scipy.stats
from sklearn import preprocessing

####################
# load signal data #
####################
parser = argparse.ArgumentParser()
parser.add_argument('--structure', '-s', type=str, default='LSTM')
parser.add_argument('--reg', '-r', type=str, default='mm')

parser.add_argument('--input_length', '-I', type=int, default=5000)
parser.add_argument('--output_length', '-O', type=int, default=5000)
parser.add_argument('--step', '-S', type=int, default=500)

args = parser.parse_args()

structure = args.structure  # Conv1D or LSTM
reg = args.reg  # on or off

in_len = args.input_length
out_len = args.output_length
step = args.step

print('structure:', structure)
print('reg:', reg)

print('------------------------')
print('input_length:', in_len)
print('output_length:', out_len)
print('step:', step)

music = 'Beat_it'
# music = 'Take_five'
# devices = 'nuforce_curve'
devices = 'teac_curve'

input_path = f'../data/wav/{music}/fix_{music}_{devices}.wav'
output_path = f'../data/wav/{music}/fix_{music}.wav'
label = ['pred_denoise', 'true_clean', 'distorted']

########################
# make train, val data #
########################
input_data = []
output_data = []

in_signal, fs = sf.read(input_path)

out_signal, _ = sf.read(output_path)
length = np.min([len(in_signal), len(out_signal)])

in_signal = in_signal[:length]
out_signal = out_signal[:length]

if reg == 'mm':
    in_signal = in_signal.reshape(-1, 1)
    in_mmscaler = preprocessing.MinMaxScaler() # インスタンスの作成
    in_mmscaler.fit(in_signal)           # xの最大・最小を計算
    in_signal = in_mmscaler.transform(in_signal) # xを変換

    out_signal = out_signal.reshape(-1, 1)
    out_mmscaler = preprocessing.MinMaxScaler() # インスタンスの作成
    out_mmscaler.fit(out_signal)           # xの最大・最小を計算
    out_signal = out_mmscaler.transform(out_signal) # xを変換

elif reg == 'std':
    in_signal = scipy.stats.zscore(in_signal)
    out_signal = scipy.stats.zscore(out_signal)

for n in range(int((len(in_signal)-(in_len))/step)):
    input_data.append(in_signal[int(n * step):int(n * step + in_len)])
    output_data.append(out_signal[int(n * step):int(n * step + out_len)])

input_data = np.array(input_data)
output_data = np.array(output_data)

test_input_data = input_data[int(len(input_data)*0.8):]
test_output_data = output_data[int(len(output_data)*0.8):]

testX = test_input_data.reshape(-1, in_len, 1)

print('testX shape:', testX.shape)

###############
# build model #
###############


class LossFunc:
    """Loss mse."""

    def __init__(self, timesteps):
        """Init."""
        self.__name__ = "LossFunc"
        self.timesteps = timesteps

    def __call__(self, y_true, y_pred):
        """Call."""
        return mean_squared_error(
            y_true[:, -self.timesteps:, :],
            y_pred[:, -self.timesteps:, :])


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
    model.compile(optimizer='adam', loss='mse')

elif structure == 'LSTM':
    model.add(CuDNNLSTM(64, input_shape=(in_len, 1), return_sequences=True))
    model.add(CuDNNLSTM(64, return_sequences=True))
    model.add(CuDNNLSTM(1, return_sequences=True))

    if in_len == out_len:
        model.compile(optimizer='adam', loss='mse')
    else:
        model.compile(optimizer='adam', loss=LossFunc(out_len))

model.summary()

year = date.today().year
month = date.today().month
day = date.today().day

# model_save_path = f'../weight/{year}{month}{day}/{music}_{devices}_{structure}_{reg}_{in_len}_{out_len}_{step}.h5'
model_save_path = f'/home/murata/project/NonLinearModeling/weight/20191230/Beat_it_teac_curve_LSTM_5280_5280_64.h5'
model.load_weights(model_save_path)


########
# pred #
########
for i in range(len(testX)):
    predy = model.predict(testX[i].reshape(1, in_len, 1))

    if i == 0:
        predict = predy
    else:
        predict = np.concatenate((predict, predy[:, in_len-step:]), axis=1)

input = in_signal[:int(len(in_signal)*0.8)][:len(predict[0])]
output = out_signal[:int(len(out_signal)*0.8)][:len(predict[0])]

plt.rcParams["font.size"] = 15  # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.direction'] = 'in'  # x axis in
plt.rcParams['ytick.direction'] = 'in'  # y axis in
plt.rcParams['axes.linewidth'] = 1.0  # axis line width
plt.rcParams['axes.grid'] = True  # make grid
plt.rcParams['figure.dpi'] = 300


os.makedirs(f'../figure/{year}{month}{day}', exist_ok=True)

t = np.arange(0, (len(input))/fs, 1 / fs)
plt.figure()
plt.plot(t, input, 'r', linewidth=3, label=label[2])  # destorted
plt.plot(t, output, 'g', linewidth=3, label=label[1])  # true
plt.plot(t, predict[0], 'b', linewidth=3, label=label[0])  # denoise
plt.xlabel('time[s]')
plt.ylabel('Amplitude[V]')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.savefig(f'../figure/{year}{month}{day}/signal_{music}_{devices}_{structure}_{reg}_{in_len}_{out_len}_{step}.jpg',
            bbox_inches="tight", pad_inches=0.05)

os.makedirs(f'../result_wave/{year}{month}{day}', exist_ok=True)

if reg == 'mm':
    predict = predict[0].reshape(-1, 1)
    predict = out_mmscaler.inverse_transform(predict) # xを変換

elif reg == 'std':
    predict = predict[0] * predict[0].std() + predict[0].mean()
else:
    predict = predict[0]

sf.write(f'../result_wave/{year}{month}{day}/{music}_{devices}_{structure}_{reg}_{in_len}_{out_len}_{step}.wav',
         predict, 44100, subtype='PCM_16')  # 16bit 44.1kHz


def signal_fft(signal, N):  # FFTするsignal長と窓長Nは同じサンプル数に固定する
    """Fft siganl."""
    win = hanning(N)  # 窓関数
    spectrum = fft(signal*win)  # フーリエ変換
    spectrum_abs = np.abs(spectrum)  # 振幅を元に信号に揃える
    half_spectrum = spectrum_abs[:int(N/2)]
    half_spectrum[0] = half_spectrum[0] / 2  # 直流成分（今回は扱わないけど）は2倍不要
    half_spectrum_dBV = 20*np.log10(half_spectrum)

    return spectrum, half_spectrum_dBV


path = f'../result_wave/{year}{month}{day}/{music}_{devices}_{structure}_{reg}_{in_len}_{out_len}_{step}.wav'
out_data, fs = sf.read(path)
_, out_half_spectrum_dBV = signal_fft(out_data, len(out_data))
f2 = np.arange(0, fs/2, (fs/2)/out_half_spectrum_dBV.shape[0])  # 横軸周波数軸[Hz]
plt.semilogx(f2, out_half_spectrum_dBV)


in_data, fs = sf.read(output_path)
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


# plot
plt.figure(figsize=(15, 10))  # figure size in inch, 横×縦
plt.semilogx(f1, in_half_spectrum_dBV, 'b', label=label[1])
plt.semilogx(f1, out_half_spectrum_dBV, 'r', label=label[0])
plt.xlim([20, 22000])
plt.xlabel('Frequency[Hz]', fontsize=15)
plt.ylabel('Amplitude[dB]', fontsize=15)
plt.legend(loc='upper right', fontsize=15)

# save
plt.savefig(f'../figure/{year}{month}{day}/fft_{music}_{devices}_{structure}_{reg}_{in_len}_{out_len}_{step}.jpg',
            bbox_inches="tight", pad_inches=0.05)

sub = max(in_half_spectrum_dBV)-max(out_half_spectrum_dBV)
print('subtract:', sub)
in_half_spectrum_dBV = in_half_spectrum_dBV - sub

# plot
plt.figure(figsize=(15, 10))  # figure size in inch, 横×縦
plt.semilogx(f1, in_half_spectrum_dBV, 'b', label=label[1])
plt.semilogx(f1, out_half_spectrum_dBV, 'r', label=label[0])
plt.xlim([20, 22000])
plt.xlabel('Frequency[Hz]', fontsize=15)
plt.ylabel('Amplitude[dB]', fontsize=15)
plt.legend(loc='upper right', fontsize=15)

# save
plt.savefig(f'../figure/{year}{month}{day}/fix_fft_{music}_{devices}_{structure}_{reg}_{in_len}_{out_len}_{step}.jpg',
            bbox_inches="tight", pad_inches=0.05)
