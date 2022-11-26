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
    def __init__(self, path_X:str, path_y:str) -> None:
        """
        Args:
            path_X (str)
            path_y (str)
        """
        self.x = read_data(path_X)
        self.y = read_data(path_y) 
        print("Data successfully read")

    


    