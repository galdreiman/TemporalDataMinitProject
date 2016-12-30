import numpy
from descritization.BaseDiscritization import BaseDiscritization


class myEWD(BaseDiscritization):
    def __init__(self, num_of_classes):
        BaseDiscritization.__init__(self)
        self.num_of_classes = num_of_classes
        print('initializing EWD with {} classes'.format(num_of_classes))

    def perform_discritization(self, prices):
        print('perform_discritization_EWD: {}'.format(self.num_of_classes))
        self.bins = numpy.linspace(min(prices), max(prices), self.num_of_classes + 1)
        digitized = numpy.digitize(prices, self.bins)
        print(digitized)

    def discretize(self, value):
        print('discretize value {}'.format(value))
        return chr(96 + numpy.digitize(value, self.bins))  # chr(97) = 'a'
