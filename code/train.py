"""train denoise or modeling nonlinear noise by using generator."""
##################
# import library #
##################
import os
import argparse
import numpy as np
from glob import glob
import soundfile as sf
from datetime import date
from natsort import natsorted
import matplotlib.pyplot as plt
from keras.layers import CuDNNLSTM, BatchNormalization, Activation, Input, Concatenate
from keras.models import Sequential, Model
from keras.losses import mean_squared_error
from keras.callbacks import ModelCheckpoint
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D, UpSampling1D


####################
# load signal data #
####################
parser = argparse.ArgumentParser()
parser.add_argument('--type', '-t', type=int, default=0)
parser.add_argument('--mode', '-m', type=str, default='denoise')
parser.add_argument('--structure', '-s', type=str, default='LSTM')
parser.add_argument('--reg', '-r', type=str, default='off')

parser.add_argument('--batch_size', '-B', type=int, default=32)
parser.add_argument('--epochs', '-E', type=int, default=100)

parser.add_argument('--input_length', '-I', type=int, default=1024)
parser.add_argument('--output_length', '-O', type=int, default=1024)
parser.add_argument('--step', '-S', type=int, default=64)


args = parser.parse_args()

type = args.type  # dist type: 0, 1, 2
mode = args.mode  # modeling, denoise
structure = args.structure  # Conv1D or LSTM
reg = args.reg  # on or off

batch_size = args.batch_size
epochs = args.epochs
in_len = args.input_length
out_len = args.output_length
step = args.step


print('type:', type)
print('mode:', mode)
print('structure:', structure)
print('reg:', reg)

print('------------------------')
print('batch_size:', batch_size)
print('epochs:', epochs)
print('input_length:', in_len)
print('output_length:', out_len)
print('step:', step)

if mode == 'modeling':
    input_paths = natsorted(glob('../data/mono/NoFX/*'))

    all_output_paths = natsorted(glob('../data/mono/Distortion/*'))
    output_paths = all_output_paths[type::3]

elif mode == 'denoise':
    all_input_paths = natsorted(glob('../data/mono/Distortion/*'))
    input_paths = all_input_paths[type::3]

    output_paths = natsorted(glob('../data/mono/NoFX/*'))

np.random.seed(0)
np.random.shuffle(input_paths)
np.random.seed(0)
np.random.shuffle(output_paths)


########################
# make train, val data #
########################
train_input_data = []
train_output_data = []

print('make train data')
for wav_num in range(int(len(output_paths)*0.6)):  # wav_num = 0~373

    in_signal, fs = sf.read(input_paths[wav_num])
    out_signal, _ = sf.read(output_paths[wav_num])

    if reg == 'on':
        in_max = max(abs(in_signal))
        out_max = max(abs(out_signal))
        in_signal = in_signal/in_max
        out_signal = out_signal/out_max

    for n in range(int((len(in_signal)-(in_len))/step)):
        train_input_data.append(in_signal[int(n * step):int(n * step + in_len)])
        train_output_data.append(out_signal[int(n * step):int(n * step + out_len)])

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

    for n in range(int((len(in_signal)-(in_len))/step)):
        val_input_data.append(in_signal[int(n * step):int(n * step + in_len)])
        val_output_data.append(out_signal[int(n * step):int(n * step + out_len)])

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
model_save_path = f'../weight/{year}{month}{day}/{structure}_{in_len}_{out_len}_{step}.h5'

cp_cb = ModelCheckpoint(filepath=model_save_path, monitor='val_loss',
                        verbose=1, save_weights_only=True,
                        save_best_only=True, mode='auto')

model = Sequential()


class LossFunc:
    """Loss mse."""

    def __init__(self, timesteps, mode):
        """Init."""
        self.__name__ = "LossFunc"
        self.timesteps = timesteps
        self.mode = mode

    def __call__(self, y_true, y_pred):
        """Call."""
        if self.mode == 'modeling':
            return mean_squared_error(
                y_true[:, -self.timesteps:, :],
                y_pred[:, -self.timesteps:, :])

        elif self.mode == 'denoise':
            return mean_squared_error(
                y_true[:, :self.timesteps, :],
                y_pred[:, :self.timesteps, :])


if structure == 'Conv1D':
    model.add(Conv1D(filters=64, kernel_size=8, padding='same',
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

elif structure == 'Unet':
    input = Input((in_len, 1))
    x = Conv1D(filters=64, kernel_size=8, padding='same')(input)
    x = BatchNormalization()(x)
    block1 = Activation("relu")(x)
    x = MaxPooling1D(2, padding='same')(block1)

    x = Conv1D(filters=64, kernel_size=8, padding='same')(x)
    x = BatchNormalization()(x)
    block2 = Activation("relu")(x)
    x = MaxPooling1D(2, padding='same')(block2)

    x = Conv1D(filters=32, kernel_size=8, padding='same')(x)
    x = BatchNormalization()(x)
    block3 = Activation("relu")(x)
    x = MaxPooling1D(2, padding='same')(block3)

    # Middle
    x = Conv1D(filters=32, kernel_size=8, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Decoder
    x = UpSampling1D(2)(x)
    x = Concatenate()([block3, x])
    x = Conv1D(filters=32, kernel_size=8, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling1D(2)(x)
    x = Concatenate()([block2, x])
    x = Conv1D(filters=64, kernel_size=8, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling1D(2)(x)
    x = Concatenate()([block1, x])
    x = Conv1D(filters=64, kernel_size=8, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # output
    x = Conv1D(1, 8, padding='same')(x)
    output = Activation("sigmoid")(x)

    model = Model(input, output)
    model.compile(optimizer='adam', loss=LossFunc(out_len, mode=mode))

elif structure == 'LSTM':
    model.add(CuDNNLSTM(64, input_shape=(in_len, 1), return_sequences=True))
    model.add(CuDNNLSTM(64, return_sequences=True))
    model.add(CuDNNLSTM(1, return_sequences=True))

    if in_len == out_len:
        model.compile(optimizer='adam', loss='mse')
    else:
        model.compile(optimizer='adam', loss=LossFunc(out_len, mode=mode))

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
plt.savefig(f'../figure/{year}{month}{day}/{structure}_{in_len}_{out_len}_{step}.jpg')

plt.figure(2)
plt.plot(epoch, history.history['acc'], label='acc')
plt.plot(epoch, history.history['val_acc'], label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.savefig(f'../figure/{year}{month}{day}/{structure}_{in_len}_{out_len}_{step}.jpg')
