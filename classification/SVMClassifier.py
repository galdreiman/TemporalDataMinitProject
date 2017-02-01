from classification.BaseClassifier import BaseClassifier

from sklearn import svm

class SVMClassifier(BaseClassifier):

    def __init__(self,input_filename,disct_num_symb):
        BaseClassifier.__init__(self,input_filename,disct_num_symb)
        print ('init SVM Classifier')
        self.clf = svm.SVC(kernel='linear',probability=True, C=1)

    def train(self):
        print('SVM train')
        self.clf.fit(self.x_train, self.y_train)

    def classify(self, x_test):
        print('SVM classify')
        return self.clf.predict(x_test)