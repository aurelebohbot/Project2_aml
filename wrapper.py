import pickle
from data_processing import Data, Features
from model import SVC_ECG, CNN1
from scoring import f1_score
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import ipdb
import numpy as np
import biosppy.signals.ecg as ecg
from typing import List
import pandas as pd
import tensorflow as tf

def transform_predictions(a):
    idx = np.argmax(a, axis=-1)
    a = np.zeros( a.shape )
    a[ np.arange(a.shape[0]), idx] = 1
    return a

def vote(l:np.array) -> np.array:
    new_l = np.zeros((1, l.shape[1]))
    new_l[0,np.argmax(np.sum(l, axis=0))] = 1
    return new_l[0]

class SignalPredictor:
    def __init__(self, row, category=None) -> None:
        self.signal = row.dropna().values
        self.beats = None
        self.category = category
        self.all_predictions = None
        self.category_predicted = np.array(np.array([1, 0, 0, 0]))
        self.unable2predict = 0

    def segment_signal(self):
        """Retrieves the heartbeats of constant length for one signal (one sample)

        Args:
            signal (np.array): one sample containing several heartbeats
        """
        peaks =  ecg.engzee_segmenter(self.signal, 300)['rpeaks']
        if len(peaks) >= 2:
            self.beats = ecg.extract_heartbeats(self.signal, peaks, 300)['templates']

    def predict_signal(self, model):
        if self.beats is not None:
            x_signal = self.beats.reshape(len(self.beats), self.beats.shape[1], 1)
            self.all_predictions = transform_predictions(model.predict(x_signal))
            self.category_predicted = vote(self.all_predictions)
        else :
            self.unable2predict = 1

    def predict_ensemble(self, model):
        self.segment_signal()
        self.predict_signal(model)

class GlobalPredictor:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.percentage_unpredicted = 0
    
    def predict_all(self, model):
        all_predictions = []
        for (_, row), (_, category) in zip(self.x.iterrows(), self.y.iterrows()):
            signal_predictor = SignalPredictor(row, category)
            signal_predictor.predict_ensemble(model)
            self.percentage_unpredicted += signal_predictor.unable2predict/len(self.x)
            all_predictions.append(signal_predictor.category_predicted)
            # ipdb.set_trace()
        self.all_predictions = np.array(all_predictions)

    def score(self):
        self.score = f1_score(to_categorical(self.y), self.all_predictions)

    def process_all(self, model):
        self.predict_all(model)
        self.score()

class Submit:
    def __init__(self, x) -> None:
        self.x = x
        self.percentage_unpredicted = 0
    
    def predict_all(self, model):
        all_predictions = []
        for (_, row) in self.x.iterrows():
            signal_predictor = SignalPredictor(row)
            signal_predictor.predict_ensemble(model)
            self.percentage_unpredicted += signal_predictor.unable2predict/len(self.x)
            all_predictions.append(signal_predictor.category_predicted)
            # ipdb.set_trace()
        self.all_predictions = np.array(all_predictions)
        ipdb.set_trace()
        self.all_predictions = np.argmax(self.all_predictions, axis=1)
        self.submission = pd.DataFrame(self.all_predictions, index=self.x.index, columns=["y"])
        ipdb.set_trace()
        self.submission.to_csv("public/submission.csv")
        

    def submission(self, model):
        self.predict_all(model)


def pipeline():
    data = Data("public/global_evaluation_x.csv", "public/global_evaluation_y.csv")
    x_test = data.x
    y_test = data.y
    model = tf.keras.models.load_model("best_model")
    global_predictor = GlobalPredictor(x_test, y_test)
    global_predictor.process_all(model)
    ipdb.set_trace()

def submission():
    x_test = pd.read_csv("public/X_test.csv", index_col='id')
    model = tf.keras.models.load_model("best_model")
    global_predictor = Submit(x_test)
    global_predictor.submission(model)
    ipdb.set_trace()

submission()




    
