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

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x


####################
# make signal data #
####################
type = 0
structure = 'LSTM'  # Conv1D or LSTM
reg = 'off'
input_paths = natsorted(glob('../data/mono/NoFX/*'))

all_output_paths = natsorted(glob('../data/mono/Distortion/*'))
output_paths = all_output_paths[type::3]

in_len = 1024
out_len = in_len
step = 64

input_data = []
output_data = []

for wav_num in range(int(len(input_paths)*0.8)):

    in_signal, fs = sf.read(input_paths[wav_num])
    out_signal, _ = sf.read(output_paths[wav_num])

    # in_signal = in_signal/max(in_signal)
    # out_signal = out_signal/max(out_signal)

    t = np.arange(0, (len(in_signal))/fs, 1 / fs)

    # plt.figure(1)
    # plt.plot(t, out_signal, 'r', label='distorted')
    # plt.plot(t, in_signal, 'b', label='clean')
    # plt.xlabel('time[s]')
    # plt.ylabel('amplitude[V]')
    # plt.xlim([0, 2])
    # plt.legend()

    for n in range(int((len(in_signal)-(in_len))/step)):
        input_data.append(in_signal[int(n * step):int(n * step + in_len)])
        output_data.append(out_signal[int(n * step):int(n * step + out_len)])

input_data = np.array(input_data)
output_data = np.array(output_data)

input_data = input_data.reshape(-1, in_len, 1)
output_data = output_data.reshape(-1, out_len, 1)

np.random.seed(0)
np.random.shuffle(input_data)
np.random.seed(0)
np.random.shuffle(output_data)

trainX = input_data[:int(len(input_data) * 0.8)]
trainy = output_data[:int(len(input_data) * 0.8)]

valX = input_data[int(len(input_data) * 0.8):]
valy = output_data[int(len(input_data) * 0.8):]

model_save_path = f'../weight/model_{structure}_dist_type{type}_weight{in_len}_{out_len}_{step}_reg{reg}.h5'
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

elif structure == 'LSTM':
    model.add(LSTM(64, input_shape=(in_len, 1), return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(1, return_sequences=True))

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(trainX, trainy, validation_data=(valX, valy),
                    epochs=epochs, batch_size=32, verbose=1,
                    callbacks=[cp_cb, es_cb])

plt.figure(2)
x = np.arange(len(history.history['loss']))
plt.plot(x, history.history['loss'], label='loss')
plt.plot(x, history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.savefig(f'{structure}_type{type}_reg{reg}.jpg')
