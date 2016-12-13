from classification.BaseClassifier import BaseClassifier

class SVMClassifier(BaseClassifier):

    def __init__(self):
        BaseClassifier.__init__(self)
        print ('init SVM Classifier')

    def train(self,train_x, train_y):
        print('SVM train')

    def classify(self, test_x):
        print('SVM classify')