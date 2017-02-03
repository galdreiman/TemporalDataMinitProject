import numpy
from descritization.BaseDiscritization import BaseDiscritization


class csvEFD(BaseDiscritization):
    def __init__(self, data, min_length, num_of_classes, out_filename, gradient):
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
            if gradient:
                f.write('@GRADIENT=1000=DEC\n')
                f.write('@GRADIENT=1001=STABLE\n')
                f.write('@GRADIENT=1002=INC\n')
            idx2 = 1
            for user, prices in self.data.items():
                if len(prices) >= min_length:
                    digitized = numpy.digitize(prices, self.bins)
                    f.write('@NAME=' + user+ ',index='+ str(idx2) + ',last='+ str(digitized[-1]) + ',raw=[' + ':'.join(str(v) for v in digitized) + ']\n')
                    idx2 += 1
                    line = ''
                    prevPrice = None
                    gradValue = None
                    for currPrice in digitized[:-1]:
                        line += str(currPrice) + ' '
                        if gradient:
                            if prevPrice is None:
                                prevPrice = currPrice
                            else:
                                if prevPrice > currPrice:
                                    gradValue = 1000
                                elif prevPrice == currPrice:
                                    gradValue = 1001
                                else:
                                    gradValue = 1002
                                line += str(gradValue)+ ' '
                        line += "-1 "
                    line += "-2\n"
                    f.write(line)
                    # f.write(' -1 '.join(str(v) for v in digitized[:-1]) + ' -2\n')
