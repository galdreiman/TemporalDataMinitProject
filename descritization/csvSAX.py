import numpy as np
from descritization.BaseDiscritization import BaseDiscritization


class csvSAX(BaseDiscritization):
    def __init__(self, data, min_length, num_of_classes, out_filename, gradient):
        self.breakpoints = {'3' : [-0.43, 0.43],
                            '4' : [-0.67, 0, 0.67],
                            '5' : [-0.84, -0.25, 0.25, 0.84],
                            '6' : [-0.97, -0.43, 0, 0.43, 0.97],
                            '7' : [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
                            '8' : [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
                            '9' : [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
                            '10': [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
                            '11': [-1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34],
                            '12': [-1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38],
                            '13': [-1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43],
                            '14': [-1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47],
                            '15': [-1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5],
                            '16': [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53],
                            '17': [-1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56],
                            '18': [-1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59],
                            '19': [-1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1, 1.25, 1.62],
                            '20': [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64]
                            }
        BaseDiscritization.__init__(self)
        print('initializing csvSAX with {} classes'.format(num_of_classes))
        self.num_of_classes = num_of_classes
        self.data = data
        self.prices = []
        for user, prices in self.data.items():
            if (len(prices) >= min_length):
                self.prices += list(map(int, prices))
        self.bins = [-9999]
        self.bins.extend(self.breakpoints[str(num_of_classes)])
        self.bins.extend([9999])
        self.std = np.std(self.prices)
        self.mean = np.mean(self.prices)

        with open(out_filename, 'w') as f:
            f.write('@MEAN' + str(self.mean) + '\n')
            f.write('@STD' + str(self.std) + '\n')
            for idx in range(1, len(self.bins)):
                f.write('@ITEM=' + str(idx) + '=[' + str(self.bins[idx-1]) + ',' + str(self.bins[idx]) + ']\n')
            if gradient:
                f.write('@GRADIENT=1000=DEC\n')
                f.write('@GRADIENT=1001=STABLE\n')
                f.write('@GRADIENT=1002=INC\n')
            idx2 = 1
            for user, prices in self.data.items():
                if len(prices) >= min_length:
                    norm = self.normalize(prices)
                    digitized = np.digitize(norm, self.bins)
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



    def normalize(self, prices):
        """
        Function will normalize an array (give it a mean of 0, and a
        standard deviation of 1) unless it's standard deviation is below
        epsilon, in which case it returns an array of zeros the length
        of the original array.
        """
        ret = [(int(x) - self.mean)//self.std for x in prices]
        return ret
