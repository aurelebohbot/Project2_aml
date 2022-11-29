import pickle
from data_processing import Data, Features
from model import SVC_ECG, CNN1
from scoring import f1_score
from sklearn.model_selection import train_test_split
import pdb
​
def pipeline():
    # loading of the data
    with open("all_signals_resampled.pickle", "rb") as f:
        all_signals = pickle.load(f)
    pdb.set_trace()
    y = all_signals["labels"]
    X = all_signals.drop("labels")
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=109) 
    pdb.set_trace()
​
    
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
​
pipeline()