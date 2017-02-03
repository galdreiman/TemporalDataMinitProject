from classification.BaseClassifier import BaseClassifier

from sklearn.ensemble import RandomForestClassifier

class RFClassifier(BaseClassifier):

    def __init__(self,input_filename,disct_num_symb):
        BaseClassifier.__init__(self,input_filename,disct_num_symb)
        print ('init RandomForest Classifier')
        # self.clf = svm.SVC(kernel='linear',probability=True, C=1)
        self.clf = RandomForestClassifier(n_estimators=100)

    def train(self):
        print('RF train')
        self.clf.fit(self.x_train, self.y_train)

    def classify(self, x_test):
        print('RF classify')
        return self.clf.predict(x_test)