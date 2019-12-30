import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

music = 'Take_five'
device = 'maudio'

input_path = f'./data/fix_wav/{music}_nuforce_{device}.wav'

in_signal, fs = sf.read(input_path)
in_signal = -in_signal

sf.write(f'./data/inv_fix/inv_{music}_nuforce_{device}.wav', in_signal, 44100, subtype='PCM_16')  # 16bit 44.1kHz
