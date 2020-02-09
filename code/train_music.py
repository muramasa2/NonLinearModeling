"""train denoise or modeling nonlinear noise by using generator."""
##################
# import library #
##################
import os
import argparse
import numpy as np
import soundfile as sf
from datetime import date
import matplotlib.pyplot as plt
from keras.layers import CuDNNLSTM
from keras.models import Sequential
from keras.losses import mean_squared_error
from keras.callbacks import ModelCheckpoint
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

parser.add_argument('--batch_size', '-B', type=int, default=32)
parser.add_argument('--epochs', '-E', type=int, default=100)

parser.add_argument('--input_length', '-I', type=int, default=5000)
parser.add_argument('--output_length', '-O', type=int, default=5000)
parser.add_argument('--step', '-S', type=int, default=500)


args = parser.parse_args()

structure = args.structure  # Conv1D or LSTM
reg = args.reg  # on or off

batch_size = args.batch_size
epochs = args.epochs
in_len = args.input_length
out_len = args.output_length
step = args.step


print('structure:', structure)
print('reg:', reg)

print('------------------------')
print('batch_size:', batch_size)
print('epochs:', epochs)
print('input_length:', in_len)
print('output_length:', out_len)
print('step:', step)

music = 'Beat_it'
devices = 'nuforce_curve'

input_path = f'../data/wav/{music}/fix_{music}_{devices}.wav'
output_path = f'../data/wav/{music}/fix_{music}.wav'

########################
# make train, val data #
########################
input_data = []
output_data = []

in_signal, fs = sf.read(input_path)
out_signal, _ = sf.read(output_path)

in_signal = in_signal[:min(len(in_signal), len(out_signal))]
out_signal = out_signal[:min(len(in_signal), len(out_signal))]

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

np.random.seed(0)
np.random.shuffle(input_data)
np.random.seed(0)
np.random.shuffle(output_data)

train_input_data = input_data[:int(len(input_data)*0.6)]
train_output_data = output_data[:int(len(output_data)*0.6)]
val_input_data = input_data[int(len(input_data)*0.6):int(len(input_data)*0.8)]
val_output_data = output_data[int(len(output_data)*0.6):int(len(output_data)*0.8)]

trainX = train_input_data.reshape(-1, in_len, 1)
trainy = train_output_data.reshape(-1, out_len, 1)
valX = val_input_data.reshape(-1, in_len, 1)
valy = val_output_data.reshape(-1, out_len, 1)

np.random.seed(0)
np.random.shuffle(trainX)
np.random.seed(0)
np.random.shuffle(trainy)

np.random.seed(0)
np.random.shuffle(valX)
np.random.seed(0)
np.random.shuffle(valy)

print('trainX shape:', trainX.shape)
print('trainy shape:', trainy.shape)
print('valX shape:', valX.shape)
print('valy shape:', valy.shape)

###############
# build model #
###############
year = date.today().year
month = date.today().month
day = date.today().day
os.makedirs(f'../weight/{year}{month}{day}', exist_ok=True)
os.makedirs(f'../figure/{year}{month}{day}', exist_ok=True)

model_save_path = f'../weight/{year}{month}{day}/{music}_{devices}_{structure}_{reg}_{in_len}_{out_len}_{step}.h5'

cp_cb = ModelCheckpoint(filepath=model_save_path, monitor='val_loss',
                        verbose=1, save_weights_only=True,
                        save_best_only=True, mode='auto')
# es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

model = Sequential()


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
    if in_len == out_len:
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss=LossFunc(out_len, mode=mode), metrics=['accuracy'])

elif structure == 'LSTM':
    model.add(CuDNNLSTM(64, input_shape=(in_len, 1), return_sequences=True))
    model.add(CuDNNLSTM(64, return_sequences=True))
    model.add(CuDNNLSTM(1, return_sequences=True))

    if in_len == out_len:
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss=LossFunc(out_len, mode=mode), metrics=['accuracy'])

model.summary()


def generator(input, output, batch):
    """Generate data."""
    print('step:', (len(input) // batch_size))

    while True:
        for num_batch in range(1, len(input) // batch+1):  # mini-batch loop
            X = input[num_batch*batch_size:(num_batch*batch+batch), :, :]
            y = output[num_batch*batch_size:(num_batch*batch+batch), :, :]
            yield (X, y)
            print('\nnum_batch:', num_batch, '\n')


history = model.fit_generator(generator(trainX, trainy, batch_size),
                              validation_data=(valX, valy), epochs=epochs,
                              steps_per_epoch=len(trainX) // batch_size,
                              max_queue_size=int(len(trainX) // batch_size),
                              verbose=1, callbacks=[cp_cb])

print('Finish learning!')

plt.figure(1)
epoch = np.arange(len(history.history['loss']))
plt.plot(epoch, history.history['loss'], label='loss')
plt.plot(epoch, history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'../figure/{year}{month}{day}/{music}_{devices}_{structure}_{in_len}_{out_len}_{step}.jpg')

print('best_loss:', history.history['val_loss'])

plt.figure(2)
plt.plot(epoch, history.history['acc'], label='acc')
plt.plot(epoch, history.history['val_acc'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig(f'../figure/{year}{month}{day}/{music}_{devices}_{structure}_{in_len}_{out_len}_{step}.jpg')
