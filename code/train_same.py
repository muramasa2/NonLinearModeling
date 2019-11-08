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

if __name__ == '__main__' and __package__ == None:
    sys.path.append(os.pardir)

####################
# make signal data #
####################
fs = 44100
L = 60
f = 1019

t = np.arange(0,L,1/44100)
# amp = np.sqrt(2)*1 # 1Vrms = √2Vpp
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
# non_lin_dist = [( 1/ (2*(i-1))) * np.sin(2*np.pi*(i)*f*t)
non_lin_dist = [( 100000 * 10**(-6)/ 2**(i-1)) * np.sin(2*np.pi*(i)*f*t)
    for i in range(start,num+1,step)] #このくらいから歪みを知覚できる
dist_signal = base_signal+sum(non_lin_dist)

dist_signal = dist_signal/max(dist_signal)
base_signal = base_signal/max(base_signal)

plt.figure(1,figsize=(10,5))
plt.plot(t,dist_signal,'r',label='distoted')
plt.plot(t,base_signal,'b',label='clean')
plt.xlim([0,0.002])
plt.legend()
plt.show()

dist_signal.shape
base_signal.shape
input_data = []
output_data = []


out_len = in_len

data_num = 1000
step = in_len/2
for n in range(data_num):
    input_data.append(dist_signal[int(n*step):int(n*step+in_len)])
    output_data.append(base_signal[int(n*step):int(n*step+in_len)])

input_data = np.array(input_data)
output_data = np.array(output_data)

input_data.shape
output_data.shape

plt.figure()
plt.plot(np.arange(len(input_data[100])),input_data[100])
plt.plot(np.arange(len(output_data[100])),output_data[100])

input_data = input_data.reshape(-1,in_len,1)
output_data = output_data.reshape(-1,out_len,1)

trainX = input_data[:int(data_num*0.6)]
trainy = output_data[:int(data_num*0.6)]

valX = input_data[int(data_num*0.6):int(data_num*0.8)]
valy = output_data[int(data_num*0.6):int(data_num*0.8)]

model_save_path = f'./weight/same_{mode}_weight{in_len}_{out_len}.h5'
epochs = 100
cp_cb = ModelCheckpoint(filepath = model_save_path, monitor='val_loss', verbose=1,
                    save_weights_only=True, save_best_only=True, mode='auto')
es_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

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

history = model.fit(trainX, trainy, validation_data=(valX, valy), epochs=epochs, verbose=1, callbacks=[cp_cb, es_cb])

plt.figure(2)
x = np.arange(len(history.history['loss']))
plt.plot(x, history.history['loss'], label='loss')
plt.plot(x, history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
