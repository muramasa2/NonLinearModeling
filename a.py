import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from glob import glob
from natsort import natsorted
import scipy.stats
from sklearn import preprocessing

all_input_paths = natsorted(glob('./data/mono/Distortion/*'))
input_paths = all_input_paths[0::3]


output_paths = natsorted(glob('./data/mono/NoFX/*'))

np.random.seed(0)
np.random.shuffle(input_paths)
np.random.seed(0)
np.random.shuffle(output_paths)

input_data = []
output_data = []

# 374~498
wav_num = 499  #499 と 502
print('input:', input_paths[wav_num], 'output:', output_paths[wav_num])


##################
# make test data #
##################
test_input_data = []
test_output_data = []

print('make test data')
in_signal, fs = sf.read(input_paths[wav_num])
out_signal, _ = sf.read(output_paths[wav_num])

# music = 'Beat_it'
# devices = devices = 'nuforce_curve'
#
# input_path = f'./data/wav/{music}/fix_{music}_{devices}.wav'
# output_path = f'./data/wav/{music}/fix_{music}.wav'
#
# in_signal, fs = sf.read(input_path)
# out_signal, _ = sf.read(output_path)
# length = np.min([len(in_signal), len(out_signal)])
#
# in_signal = in_signal[:length]
# out_signal = out_signal[:length]

plt.plot(in_signal)
plt.plot(out_signal)

std_in = scipy.stats.zscore(in_signal)
std_out = scipy.stats.zscore(out_signal)

plt.plot(std_out)
plt.plot(std_in)
in_signal[0]
in_signal[0]
fix_std_in[0]
fix_std_in = std_in - std_in[0]
fix_std_out = std_out - std_out[0]
np.mean(std_in)
plt.plot(fix_std_out)
plt.plot(fix_std_in)

in_signal = in_signal.reshape(-1, 1)
in_mmscaler = preprocessing.MinMaxScaler() # インスタンスの作成
in_mmscaler.fit(in_signal)           # xの最大・最小を計算
minmax_in = in_mmscaler.transform(in_signal) # xを変換

out_signal = out_signal.reshape(-1, 1)
out_mmscaler = preprocessing.MinMaxScaler() # インスタンスの作成
out_mmscaler.fit(out_signal)           # xの最大・最小を計算
minmax_out = out_mmscaler.transform(out_signal) # xを変換

plt.plot(minmax_out)
plt.plot(minmax_in)


fix_minmax_in = minmax_in - minmax_in[0]
fix_minmax_out = minmax_out - minmax_out[0]
