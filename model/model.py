import sklearn
import pickle
import pdb
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, ELU, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Convolution1D, MaxPool1D
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras_tuner as kt



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

def CNN1(X_train,y_train,X_test,y_test, epochs, batch_size):
    im_shape=(X_train.shape[1],1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    conv1_1=Convolution1D(128, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
    conv1_1=BatchNormalization()(conv1_1)
    pool1=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    drop1 = Dropout(0.2, input_shape=im_shape)(pool1)
    conv2_1=Convolution1D(64, (6), activation='relu', input_shape=im_shape)(drop1)
    conv2_1=BatchNormalization()(conv2_1)
    pool2=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv2_1)
    drop2 = Dropout(0.2, input_shape=im_shape)(pool2)
    flatten=Flatten()(drop2)
    dense_end2 = Dense(64, activation='relu')(flatten)
    main_output = Dense(4, activation='softmax', name='main_output')(dense_end2)
    
    
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
            ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True), TensorBoard(log_dir="./logs")]

    history=model.fit(X_train, y_train,epochs=epochs,callbacks=callbacks, batch_size=batch_size,validation_data=(X_test,y_test))
    model.save('best_model')
    return(model,history)

def HyperCNN(hp):
    im_shape=(180,1)
    inputs_cnn=Input(shape=(im_shape), name='inputs_cnn')
    size_filters = hp.Choice("size_filters", [64,128,256])
    size_dense = hp.Choice("size_dense", [128,256])

    conv1_1=Convolution1D(size_filters, (6), activation='relu', input_shape=im_shape)(inputs_cnn)
    conv1_1=BatchNormalization()(conv1_1)
    pool1=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1_1)
    drop1 = Dropout(0.3, input_shape=im_shape)(pool1)

    conv2_1=Convolution1D(size_filters, (6), activation='relu', input_shape=im_shape)(drop1)
    conv2_1=BatchNormalization()(conv2_1)
    pool2=MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv2_1)
    drop2 = Dropout(0.3, input_shape=im_shape)(pool2)

    flatten=Flatten()(drop2)
    dense_end2 = Dense(size_dense, activation='relu')(flatten)
    main_output = Dense(4, activation='softmax', name='main_output')(dense_end2)
    
    
    model = Model(inputs= inputs_cnn, outputs=main_output)
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', min_value=0.0001, max_value=0.01, sampling="log")), loss='categorical_crossentropy',metrics = ["accuracy"])
    return model

    # def fit(self, hp, model, *args, **kwargs):
    #     return model.fit(
    #         *args,
    #         batch_size=hp.Choice("batch_size", [64,128,256]),
    #         **kwargs,
    #     )
