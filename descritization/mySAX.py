from descritization.BaseDiscritization import BaseDiscritization
from descritization.saxpy import SAX


class mySAX(BaseDiscritization):

    def __init__(self):
        BaseDiscritization.__init__(self)
        print('initializing SAX')

    def perform_discritization(self, prices):
        s = SAX(4, 3, 1e-6)
        # print ('------------------')
        # print(prices)

        if len(prices) <2:
            return

        x1String = s.to_letter_strings(prices)
        # for s,i in ret: print('%s||%s' %(s,i))

        return x1String