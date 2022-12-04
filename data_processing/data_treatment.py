import csv
import os
import logging

from typing import List, Tuple
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import pickle
import json
import numpy as np
import pandas as pd
import biosppy.signals.ecg as ecg
from sklearn.utils import resample
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
            selected_data = self.x.sample(n=int(0.75*len(self.x)))
            # saving data for evaluation on global samples afterwards
            self.to_global_evaluation_x = self.x[~self.x.index.isin(selected_data.index)]
            self.to_global_evaluation_y = self.y[self.y.index.isin(self.to_global_evaluation_x.index)]
            self.to_global_evaluation_x.to_csv("public/global_evaluation_x.csv")
            self.to_global_evaluation_y.to_csv("public/global_evaluation_y.csv")
            self.x = selected_data
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
            None

    def all_signals(self) -> np.array:
        """Returns a dict containing the processed data

        Returns:
            dict: key = class, value = array of all the signals
        """
        self.all_signals = []
        for index, row in self.x.iterrows():
            category = self.y.loc[index]["y"]
            row = row.dropna().values
            signals = self.segment_signal(row)
            if signals is not None :
                for signal in signals:
                    # concatenation of the category
                    signal = np.concatenate((signal, np.array([category])), axis=0)
                    # concatenation of the sample to the big matrix
                    self.all_signals.append(signal)
        self.all_signals = pd.DataFrame(np.array(self.all_signals))
        self.all_signals.rename(columns={180:"label"}, inplace=True)
        return self.all_signals
        
    def size_samples(self) -> int:
        # return int(self.all_signals["label"].value_counts().mean())
        return 20000

    def resample(self):
        """Resampling for all classes
        """
        n_samples = self.size_samples()
        df_0 = self.all_signals[self.all_signals["label"]==0]
        df_1 = self.all_signals[self.all_signals["label"]==1]
        df_2 = self.all_signals[self.all_signals["label"]==2]
        df_3 = self.all_signals[self.all_signals["label"]==3]
        df_0_upsample = resample(df_0, replace=True, n_samples=n_samples)
        df_1_upsample = resample(df_1, replace=True, n_samples=n_samples)
        df_2_upsample = resample(df_2, replace=True, n_samples=n_samples)
        df_3_upsample = resample(df_3, replace=True, n_samples=n_samples)

        self.all_signals_resampled = pd.concat([df_0_upsample, df_1_upsample, df_2_upsample, df_3_upsample])
        with open("all_signals_resampled.pickle", "wb") as fp:
            pickle.dump(self.all_signals_resampled,fp) 
        

    def preprocessing(self):
        self.all_signals()
        self.resample()

    


    