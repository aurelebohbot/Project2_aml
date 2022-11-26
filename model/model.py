import sklearn
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class SVC_ECG(SVC):
    def __init__(self) -> None:
        self.model = self.svm_multiclass()

    def svm_multiclass(self):
        return SVC(C=1, kernel="rbf")

    def save(self, filename):
        pickle.dump(self.model, open(filename, 'wb'))
