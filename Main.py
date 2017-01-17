from preprocess.DataPreperation import DataPreperation
from descritization.mySAX import mySAX
from descritization.myEFD import myEFD
from descritization.myEWD import myEWD
from seq_minig.Bibe import Bide
from classification.SVMClassifier import SVMClassifier
import csv

class TDM(object):
    def __init__(self):
        print('init')
        self.sax_user_to_label_to_indices_list = dict()
        self.input_filename = 'Data/buys_med.dat'
        self.spade_output_filename = 'buys_med_prices_after_SPADE.txt'
        self.csv_table = 'Data/seq_table.csv'

    def preprocess_input(self):
        print('------------ preprocess_input file: ' + self.input_filename + ' ------------ ')
        PRICE_INDEX = 3
        data_preparer = DataPreperation(self.input_filename)
        self.input_train_data = data_preparer.read_csv_data()
        self.prices = [int(line[PRICE_INDEX]) for line in self.input_train_data if line[PRICE_INDEX].isdigit()]

        self.user_to_prices_map = data_preparer.read_user_to_purchases_data()

    def print_preprocess_input(self):
        with open(self.input_filename.replace('.dat', '_prices.txt'), 'w') as f:
            for user, prices in self.user_to_prices_map.items():
                if(len(prices) > 1):
                    f.write(','.join(prices) + '\n')





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


    def convert_spade_output_to_table(self):
        print('converting spade output from file: '+ self.spade_output_filename +' to table...')

        SID_to_labels_map = dict()
        all_labels = []
        all_SIDs_lables = []

        lines = [line.rstrip('\n') for line in open('Data/' + self.spade_output_filename)]
        for line in lines:
            # print(line)
            parts = line.split(' #SID: ')
            SIDs = parts[1].split()
            # print (SIDs)
            labelsAndSup = parts[0]

            labels = str(labelsAndSup.split(' #SUP: ')[:-1]).replace('[', '').replace(']','').replace('\'','')
            all_labels.append(labels)

            for SID in SIDs:
                if SID in SID_to_labels_map.keys():
                    SID_to_labels_map[SID].append(labels)
                else:
                    SID_to_labels_map[SID] = [labels]

        all_SIDs_lables.append(all_labels)

        for sid in SID_to_labels_map.keys():
            row = []
            for label in all_labels:
                if label in SID_to_labels_map[sid]:
                    row.append(1)
                else:
                    row.append(0)
            all_SIDs_lables.append(row)

        for row in all_SIDs_lables: print (row)

        #save table
        with open(self.csv_table, 'w') as table:
            writer = csv.writer(table)
            for row in all_SIDs_lables:
                print(row)
                writer.writerow(row)





    def run_sequence(self):
        print('running sequence')

        # preprocess
        #self.preprocess_input()

        #print pricess data
        #self.print_preprocess_input()

        #convert SPADE's output to table for classifier
        self.convert_spade_output_to_table()

        # discritization
        #label_sequences = self.discrit_data()

        # sequence mining
        #self.sequence_mining(label_sequences)

        # classify
        self.classify_data()


if __name__ == "__main__":
    x = TDM()
    x.run_sequence()
