from preprocess.DataPreperation import DataPreperation
from descritization.mySAX import mySAX
from descritization.myEFD import myEFD
from descritization.myEWD import myEWD
from seq_minig.Bibe import Bide
from classification.SVMClassifier import SVMClassifier


class TDM(object):
    def __init__(self):
        print('init')

    def preprocess_input(self, filename):
        print('------------ preprocess_input file: ' + filename + ' ------------ ')
        PRICE_INDEX = 3
        data_preparer = DataPreperation(filename)
        self.input_train_data = data_preparer.prepare_data()
        self.prices = [int(line[PRICE_INDEX]) for line in self.input_train_data if line[PRICE_INDEX].isdigit()]

    def discrit_data(self):
        print('------------  discritization ------------ ')
        self.sax_desc = mySAX()
        self.sax_desc.perform_discritization(self.input_train_data)
        self.ewd_desc = myEWD(10)
        self.ewd_desc.perform_discritization(self.prices)
        self.efd_desc = myEFD(10)
        self.efd_desc.perform_discritization(self.prices)

    def sequence_mining(self):
        print('------------  sequence mining ------------ ')
        miner = Bide()
        miner.mine_sequence()

    def classify_data(self):
        print('------------  classifying ------------')
        clssifier = SVMClassifier()
        clssifier.train('X_Train', 'Y_Train')
        clssifier.classify('X_Test')








    def run_sequence(self):
        print('running sequence')

        # preprocess
        input_filename = 'Data/buys_small.dat'
        self.preprocess_input(input_filename)

        # discritization
        self.discrit_data()

        # sequence mining
        self.sequence_mining()

        # classify
        self.classify_data()


if __name__ == "__main__":
    x = TDM()
    x.run_sequence()
