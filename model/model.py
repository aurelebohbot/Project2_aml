import sklearn
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from typing import Tuple
from tensorflow.keras.layers import Input, Conv2D, ELU, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Convolution1D, MaxPool1D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class SVC_ECG(SVC):
    def __init__(self) -> None:
        self.model = self.svm_multiclass()

    def svm_multiclass(self):
        return SVC(C=1, kernel="rbf")

    def save(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))

class AdaBoost(AdaBoostClassifier):
    def __init__(self) -> None:
        self.model = self.create_model()

    def create_model(self):
        return AdaBoostClassifier(n_estimators=100, random_state=0)

    def save(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))

class CNN1():
    def __init__(self) -> None:
        self.model = Sequential()

    def describe(self) -> None:
        print(self.model.summary())
    
    def get_model(self):
        return self.model

    def add_layers(self, batch_normalization:bool = True, pool:bool = False, conv:int = 0, input_shape:int = 0, flatten:bool = False,
                         dropout:int = 0, dense:int = 0, dense_activation:str = '', elu:bool = False) -> None:
        if batch_normalization:
            self.model.add(BatchNormalization())
        if pool:
            self.model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        if conv and input_shape:
            self.model.add(Conv2D(conv, (3,3), strides=(1,1), input_shape=input_shape, kernel_initializer='glorot_uniform'))
        elif conv:
            self.model.add(Conv2D(conv, (3,3), strides=(1,1), kernel_initializer='glorot_uniform'))
        if flatten: 
            self.model.add(Flatten())
        if dropout:
            self.model.add(Dropout(dropout))
        if dense and dense_activation:
            self.model.add(Dense(dense, dense_activation))
        elif dense:
            self.model.add(Dense(dense))
        if elu:
            self.model.add(ELU())

    def build_default(self, input_shape:int) -> None:
        self.add_layers(batch_normalization=False, conv=64, input_shape=input_shape, elu=True)
        self.add_layers(conv=64, elu=True)
        self.add_layers(pool=True, conv=128, elu=True)
        self.add_layers(conv=128, elu=True)
        self.add_layers(pool=True, conv=256, elu=True)
        self.add_layers(conv=256, elu=True)
        self.add_layers(pool=True, flatten=True, dense=2048, elu=True)
        self.add_layers(dropout=0.5, dense=7, dense_activation='softmax')

    def compile(self, loss:str='categorical_crossentropy', metric:str='accuracy') -> None:
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metric])

    def save(self, filename) -> None:
        pickle.dump(self.model, open(filename, 'wb'))

def CNN2():
    pass
