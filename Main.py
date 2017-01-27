import os

from descritization.csvEFD import csvEFD
from descritization.csvEWD import csvEWD
from preprocess.DataPreperation import DataPreperation
from descritization.mySAX import mySAX
from descritization.myEFD import myEFD
from descritization.myEWD import myEWD
from seq_minig.Bibe import Bide
from classification.SVMClassifier import SVMClassifier
import csv
from subprocess import *


class TDM(object):

    def __init__(self):
        print('init')
        self.sax_user_to_label_to_indices_list = dict()
        self.data_dir = 'Data'
        self.input_filename = 'buys_med'
        self.input_extension = 'dat'
        self.disct_extension = 'txt'
        self.spade_output_filename = 'buys_med_prices_after_SPADE.txt'
        self.csv_table = 'Data/seq_table.csv'

    def preprocess_input(self):
        print('------------ preprocess_input file: ' + self.get_input_filename() + ' ------------ ')
        PRICE_INDEX = 3
        data_preparer = DataPreperation(self.get_input_filename())
        self.input_train_data = data_preparer.read_csv_data()
        self.prices = [int(line[PRICE_INDEX]) for line in self.input_train_data if line[PRICE_INDEX].isdigit()]

        self.user_to_prices_map = data_preparer.read_user_to_purchases_data()
        self.target_price_for_user = dict()
        for user,prices in self.user_to_prices_map.items():
            self.target_price_for_user[user] = prices[-1]
            print('user: %s, prices: %s'% ( user, prices))


    # def store_prices_as_csv(self):
    #     with open(self.input_filename.replace('.dat', '_prices.txt'), 'w') as f:
    #         for user, prices in self.user_to_prices_map.items():
    #             if(len(prices) > 1):
    #                 f.write(','.join(prices) + '\n')

    def get_input_filename(self):
        return os.path.join(self.data_dir, '{}.{}'.format(self.input_filename,self.input_extension))

    def get_discretized_filename(self,min_length, num_of_classes,alg):
        return os.path.join(self.data_dir, '{}_min{}_numSymb{}_disc_{}.{}'.format(self.input_filename,str(min_length), str(num_of_classes), alg, self.disct_extension))

    def get_seq_mine_filename(self, disc_alg, min_length, num_of_classes, seq_mine_alg, minsup):
        return os.path.join(self.data_dir, '{}_min{}_numSymb{}_discAlg_{}_seqAlg_{}_minSup{}.{}'.format(self.input_filename, str(min_length), str(num_of_classes), disc_alg, seq_mine_alg, minsup, self.disct_extension))

    def discretize_data(self,min_length, num_of_classes,alg):
        print('------------ discretize_data ------------ ')
        outfile = self.get_discretized_filename(min_length, num_of_classes,alg)
        if not os.path.isfile(outfile):
            result = {
                'EWD': lambda x: csvEWD(self.user_to_prices_map, min_length, num_of_classes, outfile),
                'EFD': lambda x: csvEFD(self.user_to_prices_map, min_length, num_of_classes, outfile)
            }[alg](x)
        else:
            print('------------ discretize_data file: ' + outfile + ' Exists - Skipping ------------ ')
        return outfile

    def jar_wrapper(self, *args):
        # os.chdir('Data')
        process = Popen(['java', '-jar'] + list(args), stdout=PIPE, stderr=PIPE)
        ret = []
        while process.poll() is None:
            line = process.stdout.readline()
            if line != '' and line.endswith(b'\n'):
                ret.append(line[:-1])
        stdout, stderr = process.communicate()
        ret += stdout.split(b'\n')
        if stderr != b'':
            ret += stderr.split(b'\n')
        ret.remove(b'')
        return ret

    def run_spmf(self, alg, infile, outfile, *args):
        print('------------ Running SPMF ------------ ')
        params = [alg, infile, outfile, *args]
        command = ['Data\DM.jar ', 'run'] + list(params)
        command_spaces = list(map(lambda t: ' {} '.format(t), command))
        print(self.jar_wrapper(command_spaces))


    #
    # def discrit_data(self):
    #     print('------------  discritization ------------ ')
    #     self.sax_desc = mySAX()
    #     self.sax_user_to_labels_map = dict()
    #
    #     self.user_to_label_sequence_map = dict()
    #     label_sequences = []
    #
    #     for session_id, user_prices in self.user_to_prices_map.items():
    #         if len(user_prices) < 2:
    #             pass
    #         # print ('%s | %s' %(session_id, str(user_prices)))
    #         int_user_prices = [int(x) for x in user_prices if x.isdigit()]
    #         label_seq = self.sax_desc.perform_discritization(int_user_prices)
    #         if(label_seq is not None):
    #             label_sequences.append(label_seq)
    #
    #             if(session_id in self.user_to_label_sequence_map.keys()):
    #                 self.user_to_label_sequence_map[session_id].append(label_seq)
    #             else:
    #                 self.user_to_label_sequence_map[session_id] = [label_seq]
    #
    #     print(label_sequences)
    #     return label_sequences


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





    def run_sequence(self,discret_alg,disct_min_length,disct_num_symb,spmf_alg,mining_min_sup,mining_max_length):
        print('running sequence')

        # --------- preprocess -----------
        self.preprocess_input()

        # --------- discritization to file -----------
        discret_file = self.discretize_data(disct_min_length, disct_num_symb, discret_alg)

        # --------- SPMF Wrapper ------------
        spmf_output_filename = self.get_seq_mine_filename(discret_alg, disct_min_length, disct_num_symb, spmf_alg, mining_min_sup)
        if not os.path.isfile(spmf_output_filename):
            self.run_spmf(spmf_alg,discret_file,spmf_output_filename, mining_min_sup,mining_max_length, 'true')

        #convert SPMF's output to table for classifier
        # self.convert_spade_output_to_table()

        # --------- build a train table -----------
        # self.build_freq_table_for_users(sorted_freq_seqs)

        # --------- classify -----------
        self.classify_data()


if __name__ == "__main__":
    x = TDM()
    x.run_sequence('EWD',4,8,'SPADE','30%',' ')#When no need for max - just put space
    x.run_sequence('EFD',5,6,'PrefixSpan','30%','-1')
    x.run_sequence('EWD',6,10,'BIDE+','30%','-1')
    x.run_sequence('EWD',3,10,'CloSpan','30%',' ')
    x.run_sequence('EWD',3,10,'MaxSP','30%','-1')
