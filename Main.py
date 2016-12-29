
from preprocess.DataPreperation import DataPreperation
from descritization.mySAX import mySAX
from seq_minig.Bibe import Bibe
from classification.SVMClassifier import SVMClassifier


class TDM(object):

    def __init__(self):
        print ('init')


    def preprocess_input(self, filename):
        print ('------------ preprocess_input file: ' + filename + ' ------------ ')
        data_preparer = DataPreperation(filename)
        self.input_train_data = data_preparer.prepare_data()


    def discrit_data(self):
        print('------------  discritization ------------ ')
        desc = mySAX()
        desc.perform_discritization(self.input_train_data)

    def sequence_mining(self):
        print ('------------  sequence minig ------------ ')
        miner = Bibe()
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

        #discritization
        self.discrit_data()

        #sequence mining
        self.sequence_mining()

        #classify
        self.classify_data()


























if __name__ == "__main__":
    x = TDM()
    x.run_sequence()