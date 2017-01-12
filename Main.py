from preprocess.DataPreperation import DataPreperation
from descritization.mySAX import mySAX
from descritization.myEFD import myEFD
from descritization.myEWD import myEWD
from seq_minig.Bibe import Bide
from classification.SVMClassifier import SVMClassifier


class TDM(object):
    def __init__(self):
        print('init')
        self.sax_user_to_label_to_indices_list = dict()

    def preprocess_input(self, filename):
        print('------------ preprocess_input file: ' + filename + ' ------------ ')
        PRICE_INDEX = 3
        data_preparer = DataPreperation(filename)
        self.input_train_data = data_preparer.read_csv_data()
        self.prices = [int(line[PRICE_INDEX]) for line in self.input_train_data if line[PRICE_INDEX].isdigit()]

        self.user_to_prices_map = data_preparer.read_user_to_purchases_data()




    def discrit_data(self):
        print('------------  discritization ------------ ')
        self.sax_desc = mySAX()
        self.sax_user_to_labels_map = dict()

        label_sequences = []

        for session_id, user_prices in self.user_to_prices_map.items():
            if len(user_prices) < 2:
                pass
            # print ('%s | %s' %(session_id, str(user_prices)))
            int_user_prices = [int(x) for x in user_prices if x.isdigit()]
            label_seq = self.sax_desc.perform_discritization(int_user_prices)
            if(label_seq is not None):
                label_sequences.append(label_seq)

        print(label_sequences)
        return label_sequences



        # self.ewd_desc = myEWD(10)
        # self.ewd_desc.perform_discritization(self.prices)
        # self.efd_desc = myEFD(10)
        # self.efd_desc.perform_discritization(self.prices)

    def sequence_mining(self,label_sequences):
        print('------------  sequence mining ------------ ')
        miner = Bide()
        miner.mine_sequence(label_sequences)

    def classify_data(self):
        print('------------  classifying ------------')
        clssifier = SVMClassifier()
        clssifier.train('X_Train', 'Y_Train')
        clssifier.classify('X_Test')








    def run_sequence(self):
        print('running sequence')

        # preprocess
        input_filename = 'Data/buys_med.dat'
        self.preprocess_input(input_filename)

        # discritization
        label_sequences = self.discrit_data()

        # sequence mining
        self.sequence_mining(label_sequences)

        # classify
        self.classify_data()


if __name__ == "__main__":
    x = TDM()
    x.run_sequence()
