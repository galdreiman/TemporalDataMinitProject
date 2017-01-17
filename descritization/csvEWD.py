import numpy
from descritization.BaseDiscritization import BaseDiscritization


class csvEWD(BaseDiscritization):
    def __init__(self, data, min_length, num_of_classes, out_filename):
        BaseDiscritization.__init__(self)
        print('initializing csvEWD with {} classes'.format(num_of_classes))
        self.num_of_classes = num_of_classes
        self.data = data
        self.max_value = max((max(data[key]) for key in data))
        self.min_value = min((min(data[key]) for key in data))
        self.bins = numpy.linspace(int(self.min_value), int(self.max_value), self.num_of_classes + 1)
        with open(out_filename, 'w') as f:
            f.write('@MAX_VALUE' + str(self.max_value) + '\n')
            f.write('@MIN_VALUE' + str(self.min_value) + '\n')
            for idx in range(1, len(self.bins)):
                f.write('@ITEM=' + str(idx) + '=[' + str(self.bins[idx-1]) + ',' + str(self.bins[idx]) + ']\n')
            for user, prices in self.data.items():
                if (len(prices) >= min_length):
                    digitized = numpy.digitize(prices, self.bins)
                    f.write('@NAME=' + user+ '\n')
                    f.write(' -1 '.join(str(v) for v in digitized) + ' -2\n')


    # def perform_discritization(self, prices):
    #     print('perform_discritization_EWD: {}'.format(self.num_of_classes))
    #     self.bins = numpy.linspace(min(prices), max(prices), self.num_of_classes + 1)
    #     digitized = numpy.digitize(prices, self.bins)
    #     print(digitized)
    #
    # def discretize(self, value):
    #     print('discretize value {}'.format(value))
    #     return chr(96 + numpy.digitize(value, self.bins))  # chr(97) = 'a'
