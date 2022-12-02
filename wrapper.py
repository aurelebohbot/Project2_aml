import pickle
from data_processing import Data, Features
from model import SVC_ECG, CNN1
from scoring import f1_score
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pdb
import keras
import numpy as np
import biosppy.signals.ecg as ecg
from typing import List

def transform_predictions(a):
    idx = np.argmax(a, axis=-1)
    a = np.zeros( a.shape )
    a[ np.arange(a.shape[0]), idx] = 1
    return a

class SignalPredictor:
    def __init__(self, signal) -> None:
        self.signal = signal

    def segment_signal(self, signal:np.array) -> List[np.array]:
        """Retrieves the heartbeats of constant length for one signal (one sample)

        Args:
            signal (np.array): one sample containing several heartbeats

        Returns:
            List[np.array]: List of all the heartbeats of constant length
        """
        peaks =  ecg.engzee_segmenter(signal, 300)['rpeaks']
        if len(peaks) >= 2:
            beats = ecg.extract_heartbeats(signal, peaks, 300)['templates']
            return beats
        else:
            None