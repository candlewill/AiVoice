from prepro import get_spectrograms
from hyperparams import Hyperparams as hp
from utils import spectrogram2wav
import numpy as np
from scipy.io.wavfile import write
from scipy import signal

if __name__ == '__main__':
    wave_file = "000001.wav"
    mel, dones, mag = get_spectrograms(wave_file)
    print(mel.shape)
    print(dones.shape)
    print(mag.shape)
    mag = mag * hp.mag_std + hp.mag_mean  # denormalize
    audio = spectrogram2wav(np.power(10, mag) ** hp.sharpening_factor)
    audio = signal.lfilter([1], [1, -hp.preemphasis], audio)
    write("./{}_{}.wav".format("test", "1"), hp.sr, audio)
