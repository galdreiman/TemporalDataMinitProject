

class BaseClassifier(object):

    def __init__(self):
        print('init BaseClassifier')


    def train(self,train_x, train_y):
        raise NotImplementedError

    def classify(self, test_x):
        raise NotImplementedError