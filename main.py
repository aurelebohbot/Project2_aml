import pickle
from data_processing import Data, Features
from model import SVC_ECG, CNN1, HyperCNN
from scoring import f1_score
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pdb
import keras
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

def transform_predictions(a):
    idx = np.argmax(a, axis=-1)
    a = np.zeros( a.shape )
    a[ np.arange(a.shape[0]), idx] = 1
    return a



def pipeline_tuning():
    #loading of the data
    # data = Data("public/X_train.csv", "public/y_train.csv", sampling=True)
    # data.preprocessing()
    # all_signals = data.all_signals_resampled
    with open("all_signals_resampled.pickle", "rb") as f:
        all_signals = pickle.load(f)
    y = all_signals["label"]
    X = all_signals.drop("label", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109) 
    x_train = x_train.values
    x_test = x_test.values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    x_train = x_train.reshape(len(x_train), x_train.shape[1],1)
    x_test = x_test.reshape(len(x_test), x_test.shape[1],1)
    tuner = kt.RandomSearch(HyperCNN(), objective='loss')
    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
            ModelCheckpoint(filepath='best_model.h5', monitor='loss', save_best_only=True), TensorBoard(log_dir="./logs")]
    tuner.search(x_train, y_train, epochs=1, callbacks=callbacks)
    best_model = tuner.get_best_models()[0]
    best_model.save("best_model_tuning")
    
    
def pipeline_training():
    #loading of the data
    # data = Data("public/X_train.csv", "public/y_train.csv", sampling=True)
    # data.preprocessing()
    # all_signals = data.all_signals_resampled
    with open("all_signals_resampled.pickle", "rb") as f:
        all_signals = pickle.load(f)
    y = all_signals["label"]
    X = all_signals.drop("label", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=109) 
    x_train = x_train.values
    x_test = x_test.values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    x_train = x_train.reshape(len(x_train), x_train.shape[1],1)
    x_test = x_test.reshape(len(x_test), x_test.shape[1],1)
    model, history = CNN1(x_train,y_train,x_test,y_test, epochs=10, batch_size=64)
    model = tf.keras.models.load_model("best_model.h5")
    y_pred = model.predict(x_test)
    y_pred = transform_predictions(y_pred)
    print(f1_score(y_test, y_pred))
    pdb.set_trace()

pipeline_tuning()