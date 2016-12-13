
from preprocess.DataPreperation import DataPreperation
from descritization.SAX import SAX
from seq_minig.Bibe import Bibe
from classification.SVMClassifier import SVMClassifier

class TDM(object):

    def __init__(self):
        print ('init')


    def preprocess_input(self, filename):
        print ('------------ preprocess_input file: ' + filename + ' ------------ ')
        data_preparer = DataPreperation(filename)
        data_preparer.prepare_data()

    def discrit_data(self):
        print('------------  discritization ------------ ')
        desc = SAX()
        desc.perporm_discritization()

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
        filename = 'path'
        self.preprocess_input(filename)

        #discritization
        self.discrit_data()

        #sequence mining
        self.sequence_mining()

        #classify
        self.classify_data()


























if __name__ == "__main__":
    x = TDM()
    x.run_sequence()