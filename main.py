import pickle
from data_processing import Data, Features
from model import SVC_ECG, CNN1
from scoring import f1_score
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pdb

def pipeline():
    #loading of the data
    data = Data("public/X_train.csv", "public/y_train.csv")
    data.preprocessing()
    all_signals = data.all_signals_resampled
    # with open("all_signals_resampled.pickle", "rb") as f:
    #     all_signals = pickle.load(f)
    y = all_signals["label"]
    X = all_signals.drop("label", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=109) 
    x_train = x_train.values
    x_test = x_test.values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    x_train = x_train.reshape(len(x_train), x_train.shape[1],1)
    x_test = x_test.reshape(len(x_test), x_test.shape[1],1)
    model, history = CNN1(x_train,y_train,x_test,y_test, epochs=1)
    pdb.set_trace()
    
    

    # # featuring
    # featuring_process = Features(data)
    # x, y = featuring_process.featuring()
    # # resampling
    # ### to do
    # # split
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=109) 
    # # model
    # model_builder = SVC_ECG()
    # model_builder.model.fit(x_train, y_train)
    # model_builder.save('model/models/svc.sav')
    # # prediction
    # print(f1_score(y_test, model_builder.model.predict(x_test)))
    # return

pipeline()