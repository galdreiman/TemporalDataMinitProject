from descritization.BaseDiscritization import BaseDiscritization
from descritization.saxpy import SAX


class mySAX(BaseDiscritization):

    def __init__(self):
        BaseDiscritization.__init__(self)
        print('initializing SAX')

    def perporm_discritization(self, input_data):
        print('perporm_discritization...')

        s = SAX(6, 5, 1e-6)
        print(input_data[0])
        price_index = 4
        prices = [int(line[price_index]) for line in input_data if line[price_index].isdigit()]
        print(len(prices))
        (x1String, x1Indices) = s.to_letter_rep(prices)

        for s,i in zip(x1String,x1Indices): print('%s||%s' %(s,i))