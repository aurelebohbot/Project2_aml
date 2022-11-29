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

class CNN():
    def __init__(self) -> None:
        self.model = Sequential()

    def describe(self) -> None:
        print(self.model.summary())
    
    def get_model(self):
        return self.model

    def compile(self, loss:str='categorical_crossentropy', metric:str='accuracy') -> None:
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metric])

    def save(self, filename) -> None:
        pickle.dump(self.model, open(filename, 'wb'))

class CNN1(CNN):
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

    def build_default(self, input_shape:Tuple[int,int]) -> None:
        self.add_layers(batch_normalization=False, conv=64, input_shape=input_shape, elu=True)
        self.add_layers(conv=64, elu=True)
        self.add_layers(pool=True, conv=128, elu=True)
        self.add_layers(conv=128, elu=True)
        self.add_layers(pool=True, conv=256, elu=True)
        self.add_layers(conv=256, elu=True)
        self.add_layers(pool=True, flatten=True, dense=2048, elu=True)
        self.add_layers(dropout=0.5, dense=7, dense_activation='softmax')

class CNN2(CNN):
    def add_layers(self, batch_normalization:bool = True, pool:int = 0, conv:int = 0, input_shape:int = 0,
                    flatten:bool = False, dense:int = 0, dense_activation:str=''):
        if batch_normalization:
            self.model.add(BatchNormalization())
        if pool:
            self.model.add(MaxPool1D(pool_size=(pool), strides=(2), padding="same"))
        if conv and input_shape:
            self.model.add(Convolution1D(conv, (6), activation='relu', input_shape=input_shape))
        elif conv:
            self.model.add(Convolution1D(conv, (6), activation='relu'))
        if flatten:
            self.model.add(Flatten())
        if dense:
            self.model.add(Dense(dense, dense_activation))

    def build_default(self, input_shape:int):
       self.add_layers(batch_normalization=False, conv=64, input_shape=input_shape)
       self.add_layers(pool=3, conv=64)
       self.add_layers(pool=2, conv=64)
       self.add_layers(batch_normalization=False, pool=2, flatten=True, dense=64, dense_activation='relu')
       self.add_layers(batch_normalization=False, dense=32, dense_activation='relu')
       self.add_layers(batch_normalization=False, dense=5, dense_activation='softmax')


'''

im_shape=(X_train.shape[1],1)
inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
conv1_1=Convolution1D(64, (6), activation='relu', input_shape=im_shape)(inputs_cnn)

conv1_1=BatchNormalization()(conv1_1)
pool1=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
conv2_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool1)

conv2_1=BatchNormalization()(conv2_1)
pool2=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv2_1)
conv3_1=Convolution1D(64, (3), activation='relu', input_shape=im_shape)(pool2)

conv3_1=BatchNormalization()(conv3_1)
pool3=MaxPool1D(pool_size=(2), strides=(2), padding="same")(conv3_1)
flatten=Flatten()(pool3)
dense_end1 = Dense(64, activation='relu')(flatten)
dense_end2 = Dense(32, activation='relu')(dense_end1)
main_output = Dense(5, activation='softmax', name='main_output')(dense_end2)


model = Model(inputs= inputs_cnn, outputs=main_output)
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])

'''