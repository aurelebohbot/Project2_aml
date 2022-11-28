import csv
import os
import logging

import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
            self.x = self.x.sample(n=1000)
            self.y = self.y[self.y.index.isin(self.x.index)]
        print("Data successfully read")

    


    