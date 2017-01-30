import numpy
from descritization.BaseDiscritization import BaseDiscritization


class csvEFD(BaseDiscritization):
    def __init__(self, data, min_length, num_of_classes, out_filename):
        BaseDiscritization.__init__(self)
        print('initializing csvEFD with {} classes'.format(num_of_classes))
        self.num_of_classes = num_of_classes
        self.data = data
        self.prices = []
        for user, prices in self.data.items():
            if (len(prices) >= min_length):
                self.prices += list(map(int, prices))
        self.prices = sorted(self.prices)
        self.bins = []
        for i in range(self.num_of_classes):
            self.bins.append(self.prices[int(i * (len(self.prices) / self.num_of_classes))])
        self.bins.append(max(self.prices))

        with open(out_filename, 'w') as f:
            f.write('@NUM_OF_CLASSES' + str(self.num_of_classes) + '\n')
            for idx in range(1, len(self.bins)):
                f.write('@ITEM=' + str(idx) + '=[' + str(self.bins[idx-1]) + ',' + str(self.bins[idx]) + ']\n')
            idx2 = 1
            for user, prices in self.data.items():
                if len(prices) >= min_length:
                    digitized = numpy.digitize(prices, self.bins)
                    f.write('@NAME=' + user+ ',index='+ str(idx2) + ',last='+ str(digitized[-1]) + ',raw=[' + ':'.join(str(v) for v in digitized) + ']\n')
                    idx2 += 1
                    f.write(' -1 '.join(str(v) for v in digitized[:-1]) + ' -2\n')


    # def perform_discritization(self, prices):
    #     print('perform_discritization_EWD: {}'.format(self.num_of_classes))
    #     self.bins = numpy.linspace(min(prices), max(prices), self.num_of_classes + 1)
    #     digitized = numpy.digitize(prices, self.bins)
    #     print(digitized)
    #
    # def discretize(self, value):
    #     print('discretize value {}'.format(value))
    #     return chr(96 + numpy.digitize(value, self.bins))  # chr(97) = 'a'
