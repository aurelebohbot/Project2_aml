import csv
import os
import logging

from typing import List
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import pickle
import json
import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg
import pdb

def read_data(path):
    return pd.read_csv(path, index_col='id')

class Data:
    """Class to handle data processing before feature selection and model
    """
    def __init__(self, path_X:str, path_y:str, sampling:bool=False) -> None:
        """
        Args:
            path_X (str)
            path_y (str)
        """
        self.x = read_data(path_X)
        self.y = read_data(path_y) 
        if sampling:
            self.x = self.x.sample(n=10)
            self.y = self.y[self.y.index.isin(self.x.index)]
        print("Data successfully read")

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
            return np.array([])

    def all_signals(self) -> dict:
        """Returns a dict containing the processed data

        Returns:
            dict: key = class, value = array of all the signals
        """
        self.all_signals = {}
        for index, row in self.x.iterrows():
            category = self.y.loc[index]["y"]
            row = row.dropna().values
            try:
                self.all_signals[category] = np.concatenate((self.all_signals[category],self.segment_signal(row)), axis=0)
            except KeyError:
                self.all_signals[category] = self.segment_signal(row)
        with open("all_signals.pickle", "wb") as fp:
            pickle.dump(self.all_signals,fp) 

    def preprocessing(self):
        self.all_signals()

    


    