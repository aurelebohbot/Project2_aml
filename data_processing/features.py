import csv
import os
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Tuple
from data_processing.data_treatment import Data

class Features:
    def __init__(self, data:Data) -> None:
        self.X = data.x
        self.y = data.y


    def retrieve_features_per_sample(self, signal:pd.DataFrame) -> dict:
        """Retrieves the features per sample

        Args:
            signal (pd.DataFrame): Time Series for one sample

        Returns:
            dict: Corresponding features
        """
        r_peaks = ecg.engzee_segmenter(signal, 300)['rpeaks']
        if len(r_peaks) >= 2:
            #ToDo : here we drop several samples
            beats = ecg.extract_heartbeats(signal, r_peaks, 300)['templates']
            if len(beats) != 0:
                mu = np.mean(beats, axis=0) 
                var = np.std(beats, axis=0)
                md = np.median(beats, axis=0)
                
                return {"mu":mu,"var":var,"md":md}

    def retrieves_features(self):
        """All the features for all the samples possible
        """
        self.pre_features = {sample_id:self.retrieve_features_per_sample(sample.dropna().to_numpy(dtype='float32')) for sample_id, sample in self.X.iterrows()}
        print("Features retrieved -> reduction")


    def flatten_features(self):
        """For each sample only one instance of each feature

        Args:
            features_samples (dict)
        """
        ft = self.pre_features.copy()
        keys_to_remove = []
        for key in ft.keys():
            if ft[key]:
                ft[key]["mu"] = np.nanmean(ft[key]["mu"])
                ft[key]["var"] = np.nanmean(ft[key]["var"])
                ft[key]["md"] = np.nanmean(ft[key]["md"])
            else:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del ft[key]
        self.features = ft
        print("Features reduction done")

    def data_cleaned(self) -> None:
        """Function to provide a dataframe with one unique feature per sample
        """
        self.X_cleaned = pd.DataFrame.from_dict(self.features, orient="index")
        self.y_cleaned = self.y[self.y.index.isin(self.X_cleaned.index)]

    def plot_points(self) -> None:
        self.all_data = pd.concat([self.X_cleaned, self.y_cleaned], axis=1)
        fig = px.scatter_3d(self.all_data, x="mu", y="var", z="md", color="y")
        fig.show()

    def featuring(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Pipeline to extract the features
        """
        self.retrieves_features()
        self.flatten_features()
        self.data_cleaned()
        self.plot_points()
        return self.X_cleaned, self.y_cleaned


