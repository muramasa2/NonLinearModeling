"""numpy module."""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Conv2DTranspose, Lambda, CuDNNLSTM
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from glob import glob
from natsort import natsorted
import soundfile as sf
import keras.backend as K
from keras.losses import mean_squared_error

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

####################
# make signal data #
####################
type = 0
mode = 'denoise'  # modeling, denoise
structure = 'LSTM'  # Conv1D or LSTM
reg = 'off'

if mode == 'modeling':
    input_paths = natsorted(glob('../data/mono/NoFX/*'))

    all_output_paths = natsorted(glob('../data/mono/Distortion/*'))
    output_paths = all_output_paths[type::3]

elif mode == 'denoise':
    all_input_paths = natsorted(glob('../data/mono/Distortion/*'))
    input_paths = all_input_paths[type::3]

    output_paths = natsorted(glob('../data/mono/NoFX/*'))


in_len = 5280
out_len = 480
step = 480

np.random.seed(0)
np.random.shuffle(input_paths)
np.random.seed(0)
np.random.shuffle(output_paths)

train_input_data = []
train_output_data = []

print('make train data')
for wav_num in range(int(len(output_paths)*0.6)):  # 0~373

    in_signal, fs = sf.read(input_paths[wav_num])
    out_signal, _ = sf.read(output_paths[wav_num])

    if reg == 'on':
        in_max = max(abs(in_signal))
        out_max = max(abs(out_signal))
        in_signal = in_signal/in_max
        out_signal = out_signal/out_max

    t = np.arange(0, (len(in_signal))/fs, 1 / fs)

    # plt.figure(1)
    # plt.plot(t, out_signal, 'r', label='distorted')
    # plt.plot(t, in_signal, 'b', label='clean')
    # plt.xlabel('time[s]')
    # plt.ylabel('amplitude[V]')
    # plt.xlim([0, 2])
    # plt.legend()

    for n in range(int((len(in_signal)-(in_len))/step)):
        train_input_data.append(in_signal[int(n * step):int(n * step + in_len)])
        train_output_data.append(out_signal[int(n * step)+(in_len-out_len):int(n * step + in_len)])

val_input_data = []
val_output_data = []

print('make val data')
for wav_num in range(int(len(output_paths)*0.6), int(len(output_paths)*0.8)):  # 374~498

    in_signal, fs = sf.read(input_paths[wav_num])
    out_signal, _ = sf.read(output_paths[wav_num])

    if reg == 'on':
        in_max = max(abs(in_signal))
        out_max = max(abs(out_signal))
        in_signal = in_signal/in_max
        out_signal = out_signal/out_max

    t = np.arange(0, (len(in_signal))/fs, 1 / fs)

    # plt.figure(1)
    # plt.plot(t, out_signal, 'r', label='distorted')
    # plt.plot(t, in_signal, 'b', label='clean')
    # plt.xlabel('time[s]')
    # plt.ylabel('amplitude[V]')
    # plt.xlim([0, 2])
    # plt.legend()

    for n in range(int((len(in_signal)-(in_len))/step)):
        val_input_data.append(in_signal[int(n * step):int(n * step + in_len)])
        val_output_data.append(out_signal[int(n * step)+(in_len-out_len):int(n * step + in_len)])

train_input_data = np.array(train_input_data)
train_output_data = np.array(train_output_data)
val_input_data = np.array(val_input_data)
val_output_data = np.array(val_output_data)


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

print('trainX shape:',trainX.shape)
print('trainy shape:',trainy.shape)
print('valX shape:',valX.shape)
print('valy shape:',valy.shape)

model_save_path = f'../weight/model_{structure}_dist_type{type}_weight{in_len}_{out_len}_{step}_reg{reg}_{mode}.h5'
epochs = 100
cp_cb = ModelCheckpoint(filepath=model_save_path, monitor='val_loss',
                        verbose=1, save_weights_only=True,
                        save_best_only=True, mode='auto')
es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

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
    model.compile(optimizer='adam', loss=LossFunc(out_len))
class LossFunc:

    def __init__(self, timesteps):
        self.__name__ = "LossFunc"
        self.timesteps = timesteps

    def __call__(self, y_true, y_pred):
        return mean_squared_error(
            y_true[:, -self.timesteps:, :],
            y_pred[:, -self.timesteps:, :])


model.summary()

history = model.fit(trainX, trainy, validation_data=(valX, valy),
                    epochs=epochs, batch_size=32, verbose=1,
                    callbacks=[cp_cb, es_cb])

print('finish learning!')

plt.figure(2)
x = np.arange(len(history.history['loss']))
plt.plot(x, history.history['loss'], label='loss')
plt.plot(x, history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.savefig(f'../model_{structure}_dist_type{type}_weight{in_len}_{out_len}_{step}_reg{reg}_{mode}.jpg')
