import sklearn
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier

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