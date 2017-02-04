import os

from descritization.csvEFD import csvEFD
from descritization.csvEWD import csvEWD
from preprocess.DataPreperation import DataPreperation
from descritization.mySAX import mySAX
from descritization.myEFD import myEFD
from descritization.myEWD import myEWD
from seq_minig.Bibe import Bide
from classification.SVMClassifier import SVMClassifier
from classification.RFClassifier import RFClassifier
import csv
from subprocess import *
import datetime


class TDM(object):

    def __init__(self):
        print('init')
        self.sax_user_to_label_to_indices_list = dict()
        self.data_dir = 'Data'
        self.input_filename = 'buys_full'
        self.summary_filename = 'experiment_summary'
        self.input_extension = 'dat'
        self.disct_extension = 'txt'
        self.classifier_extension = 'csv'

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
            # print('user: %s, prices: %s'% ( user, prices))


    # def store_prices_as_csv(self):
    #     with open(self.input_filename.replace('.dat', '_prices.txt'), 'w') as f:
    #         for user, prices in self.user_to_prices_map.items():
    #             if(len(prices) > 1):
    #                 f.write(','.join(prices) + '\n')

    def get_input_filename(self):
        return os.path.join(self.data_dir, '{}.{}'.format(self.input_filename,self.input_extension))

    def get_summary_filename(self):
        return os.path.join(self.data_dir, '{}.{}'.format(self.summary_filename,self.classifier_extension))

    def get_discretized_filename(self,min_length, num_of_classes,alg):
        return os.path.join(self.data_dir, '{}_min{}_numSymb{}_discAlg_{}.{}'.format(self.input_filename,str(min_length), str(num_of_classes), alg, self.disct_extension))

    def get_seq_mine_filename(self, disc_alg, min_length, num_of_classes, seq_mine_alg, minsup):
        return os.path.join(self.data_dir, '{}_min{}_numSymb{}_discAlg_{}_seqAlg_{}_minSup{}.{}'.format(self.input_filename, str(min_length), str(num_of_classes), disc_alg, seq_mine_alg.replace('+',''), minsup, self.disct_extension))

    def get_table_for_classifier_filename(self, disc_alg, min_length, num_of_classes, seq_mine_alg, minsup):
        return os.path.join(self.data_dir, '{}_min{}_numSymb{}_discAlg_{}_seqAlg_{}_minSup{}_forClassifier.{}'.format(self.input_filename, str(min_length), str(num_of_classes), disc_alg, seq_mine_alg, minsup, self.classifier_extension))

    # def get_classifier_output_filename(self, disc_alg, min_length, num_of_classes, seq_mine_alg, minsup, classifier_name, folds):
    #     return os.path.join(self.data_dir, '{}_min{}_numSymb{}_discAlg_{}_seqAlg_{}_minSup{}_classifier_{}_folds_{}.{}'.format(
    #         self.input_filename, str(min_length), str(num_of_classes), disc_alg, seq_mine_alg, minsup, classifier_name,
    #         folds, self.classifier_extension))

    def discretize_data(self,min_length, num_of_classes,alg):
        print('------------ discretize_data ------------ ')
        outfile = self.get_discretized_filename(min_length, num_of_classes,alg)
        if not os.path.isfile(outfile):
            result = {
                'EWD': lambda x: csvEWD(self.user_to_prices_map, min_length, num_of_classes, outfile, False),
                'EFD': lambda x: csvEFD(self.user_to_prices_map, min_length, num_of_classes, outfile, False),
                'EWDG': lambda x: csvEWD(self.user_to_prices_map, min_length, num_of_classes, outfile, True),
                'EFDG': lambda x: csvEFD(self.user_to_prices_map, min_length, num_of_classes, outfile, True),
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


    def sequence_mining(self,label_sequences):
        print('------------  sequence mining ------------ ')
        miner = Bide()
        sorted_freq_seqs = miner.mine_sequence(label_sequences)
        return sorted_freq_seqs

    def classify_data(self, input_filename, summary_filename, disct_num_symb, alg, folds):
        print('------------  classifying ------------')
        clssifier = {
            'SVM': lambda x: SVMClassifier(input_filename,disct_num_symb),
            'RF': lambda x: RFClassifier(input_filename,disct_num_symb),
        }[alg](x)

        # clssifier.train()
        # clssifier.classify()
        return clssifier.classify_with_CV(folds,summary_filename)


    def load_session_id_map(self, spmf_input_filename, import_spmf_delta):
        session_map = dict()
        lines = [line.rstrip('\n') for line in open(spmf_input_filename)]
        for line in lines:
            if "@NAME" not in line:
                continue
            parts = line.split(',')
            sid = parts[0].split('=')[1]
            idx = parts[1].split('=')[1]
            target_class = parts[2].split('=')[1]
            session_map[idx] = (sid,target_class)  #str(int(idx)+import_spmf_delta)
        return session_map

    def convert_spmf_output_to_table(self, spmf_input_filename, spmf_output_filename, table_for_classifier_filename, import_spmf_delta):
        print('converting spmf output from file: ' + spmf_output_filename + ' to CSV...')

        SID_to_labels_map = dict()
        all_labels = []
        all_SIDs_lables_to_classifier = []

        tmp_sid_to_original_sid_and_class_map = self.load_session_id_map(spmf_input_filename, import_spmf_delta)

        lines = [line.rstrip('\n') for line in open(spmf_output_filename)]

        all_labels.append('ID')
        for line in lines:
            # print(line)
            parts = line.split(' #SID: ')
            SIDs = parts[1].split()
            # print (SIDs)
            labelsAndSup = parts[0]

            labels = str(labelsAndSup.split(' #SUP: ')[:-1]).replace('[', '').replace(']','').replace('\'','')
            all_labels.append(labels)

            for SID in SIDs:
                SID = str(int(SID)+import_spmf_delta)
                if SID in SID_to_labels_map.keys():
                    SID_to_labels_map[SID].append(labels)
                else:
                    SID_to_labels_map[SID] = [labels]
        all_labels.append('Class')
        all_SIDs_lables_to_classifier.append(all_labels)

        keys = [x for x in self.target_price_for_user.keys()]
        for sid in SID_to_labels_map.keys():
            row = []
            # print(sid)
            row.append(tmp_sid_to_original_sid_and_class_map[sid][0])

            for label in all_labels[1:-1]:
                if label in SID_to_labels_map[sid]:
                    row.append(1)
                else:
                    row.append(0)
            # appending target value: the last price in the purchase sequence:
            # print (sid)

            if(int(sid) in keys):
                # print("sid [%d]   price [%s]" %(sid, self.target_price_for_user[str(sid)]))
                row.append(self.target_price_for_user[sid])
            all_SIDs_lables_to_classifier.append(row)
            row.append(tmp_sid_to_original_sid_and_class_map[sid][1])

        # for row in all_SIDs_lables_to_classifier: print (row)

        #save table
        with open(table_for_classifier_filename, 'w', newline='', encoding='utf8') as table:
            writer = csv.writer(table)
            for row in all_SIDs_lables_to_classifier:
                # print(row)
                writer.writerow(row)

    def build_freq_table_for_users(self, sorted_freq_seqs):
        print('build_freq_table_for_users')
        # print(sorted_freq_seqs)
        for session_id, user_seq in self.user_to_label_sequence_map.items():
            # print('user: %s  sequence: %s' % (session_id,user_seq))
            for tpl in sorted_freq_seqs:
                seq = '.*'.join(tpl[0])
                # print (seq)
                # check using regex if seq matches the user_sequence:


    def run_sequence(self,discret_alg,disct_min_length,disct_num_symb,spmf_alg,mining_min_sup,mining_max_length, import_spmf_delta, classifier_name, folds):
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
        classifier_input_filename = self.get_table_for_classifier_filename(discret_alg, disct_min_length, disct_num_symb, spmf_alg, mining_min_sup)
        if not os.path.isfile(classifier_input_filename):
            self.convert_spmf_output_to_table(discret_file, spmf_output_filename, classifier_input_filename, import_spmf_delta)

        # --------- classify -----------
        summary_filename = self.get_summary_filename()
        if not os.path.isfile(summary_filename):
            with open(summary_filename, "a") as myfile:
                myfile.write("TS, discret_alg, disct_min_length,disct_num_symb, spmf_alg, mining_min_sup, classifier_name, folds, Accuracy,STD,kappa_score,precision,recall"+ '\n')

        # classifier_output_filename = self.get_classifier_output_filename(discret_alg, disct_min_length,
        #                                                                    disct_num_symb, spmf_alg, mining_min_sup, classifier_name, folds)


        params = [discret_alg, disct_min_length,disct_num_symb, spmf_alg, mining_min_sup, classifier_name, folds]
        results = self.classify_data(classifier_input_filename, summary_filename, disct_num_symb, classifier_name, folds)

        output_params = []
        output_params += [datetime.datetime.utcnow()]
        output_params += params
        output_params += results
        with open(summary_filename, "a") as myfile:
            line = ",".join(list(map(str,output_params)))
            myfile.write(line + '\n')



if __name__ == "__main__":
    x = TDM()
    # x.run_sequence('EWD',4,4,'SPADE','30%',' ',0)#When no need for max - just put space
    # x.run_sequence('EWD',6,10,'BIDE+','30%','-1',1)
    # x.run_sequence('EFD',5,6,'PrefixSpan','30%','-1',1)
    # x.run_sequence('EWD',3,10,'CloSpan','30%',' ',0)

    dicrete_algs = ['EFD','EWD','EFDG','EWDG']
    minimum_length = [2, 3, 4, 5]
    number_of_symbols = [3,4,5,7,10,15]
    minimum_supports = ["30%","40%","50%"]
    #Bide and prefix span produce the same output
    # seq_mining_algs = [['SPADE',' ',0],['BIDE+','-1',1],['PrefixSpan','-1',1],['CloSpan',' ',0]] #[Alg, max pattern length, deltafix]
    seq_mining_algs = [['SPADE',' ',0],['BIDE+','-1',1],['CloSpan',' ',0]] #[Alg, max pattern length, deltafix]
    classifiers = ['RF','SVM']
    folds = [5, 10]

    for discrete_alg in dicrete_algs:
        for minLength in minimum_length:
            for nSymbol in number_of_symbols:
                for minSup in minimum_supports:
                    for seq_minig_alg in seq_mining_algs:
                        for classifier_alg in classifiers:
                            for fold in folds:
                                try:
                                    print('Running Discretization:{}  MinimumLength:{} Symbols:{} SeqMining:{}'.format(
                                        discrete_alg, minLength, nSymbol, seq_minig_alg))
                                    x.run_sequence(discrete_alg, minLength, nSymbol, seq_minig_alg[0], minSup,
                                                   seq_minig_alg[1],
                                                   seq_minig_alg[2], classifier_alg, fold)
                                except Exception:
                                    print('ERROR!!ERROR!!ERROR!!ERROR!!ERROR!!ERROR!!ERROR!! CYCLE SKIPPED')
                                    pass




