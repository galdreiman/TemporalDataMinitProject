from descritization.csvEFD import csvEFD
from descritization.csvEWD import csvEWD
from preprocess.DataPreperation import DataPreperation
from descritization.mySAX import mySAX
from descritization.myEFD import myEFD
from descritization.myEWD import myEWD
import re
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
        self.target_price_for_user = dict()
        for user,prices in self.user_to_prices_map.items():
            self.target_price_for_user[user] = prices[-1]
            print('user: %s, prices: %s'% ( user, prices))


    def store_prices_as_csv(self):
        with open(self.input_filename.replace('.dat', '_prices.txt'), 'w') as f:
            for user, prices in self.user_to_prices_map.items():
                if(len(prices) > 1):
                    f.write(','.join(prices) + '\n')

    def discrit_data_ewd(self):
        csvEFD(self.user_to_prices_map, 5, 10, self.input_filename.replace('.dat', '_EFD.txt'))
        csvEWD(self.user_to_prices_map, 5, 10, self.input_filename.replace('.dat', '_EWD.txt'))






    def discrit_data(self):
        print('------------  discritization ------------ ')
        self.sax_desc = mySAX()
        self.sax_user_to_labels_map = dict()

        self.user_to_label_sequence_map = dict()
        label_sequences = []

        for session_id, user_prices in self.user_to_prices_map.items():
            if len(user_prices) < 2:
                pass
            # print ('%s | %s' %(session_id, str(user_prices)))
            int_user_prices = [int(x) for x in user_prices if x.isdigit()]
            label_seq = self.sax_desc.perform_discritization(int_user_prices)
            if(label_seq is not None):
                label_sequences.append(label_seq)

                if(session_id in self.user_to_label_sequence_map.keys()):
                    self.user_to_label_sequence_map[session_id].append(label_seq)
                else:
                    self.user_to_label_sequence_map[session_id] = [label_seq]

        print(label_sequences)
        return label_sequences



        # self.ewd_desc = myEWD(10)
        # self.ewd_desc.perform_discritization(self.prices)
        # self.efd_desc = myEFD(10)
        # self.efd_desc.perform_discritization(self.prices)

    def sequence_mining(self,label_sequences):
        print('------------  sequence mining ------------ ')
        miner = Bide()
        sorted_freq_seqs = miner.mine_sequence(label_sequences)
        return sorted_freq_seqs

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
            # appending target value: the last price in the purchase sequence:
            print (sid)
            keys = [x for x in self.target_price_for_user.keys()]
            if(int(sid) in keys):
                print("sid [%d]   price [%s]" %(sid, self.target_price_for_user[str(sid)]))
                row.append(self.target_price_for_user[sid])
            all_SIDs_lables.append(row)

        for row in all_SIDs_lables: print (row)

        #save table
        with open(self.csv_table, 'w') as table:
            writer = csv.writer(table)
            for row in all_SIDs_lables:
                print(row)
                writer.writerow(row)

    def build_freq_table_for_users(self, sorted_freq_seqs):
        print('build_freq_table_for_users')
        print(sorted_freq_seqs)
        for session_id, user_seq in self.user_to_label_sequence_map.items():
            # print('user: %s  sequence: %s' % (session_id,user_seq))
            for tpl in sorted_freq_seqs:
                seq = '.*'.join(tpl[0])
                print (seq)
                # check using regex if seq matches the user_sequence:





    def run_sequence(self):
        print('running sequence')

        # --------- preprocess -----------
        self.preprocess_input()

        # --------- discritization -----------
        #print pricess data
        # self.store_prices_as_csv()
        label_sequences = self.discrit_data()
        # self.discrit_data_ewd()

        #convert SPADE's output to table for classifier
        # self.convert_spade_output_to_table()

        # --------- sequence mining -----------
        sorted_freq_seqs = self.sequence_mining(label_sequences)

        # --------- build a train table -----------
        self.build_freq_table_for_users(sorted_freq_seqs)

        # --------- classify -----------
        self.classify_data()


if __name__ == "__main__":
    x = TDM()
    x.run_sequence()
