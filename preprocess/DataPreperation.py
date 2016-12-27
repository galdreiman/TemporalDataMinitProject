import csv


class DataPreperation(object):

    def __init__(self, dataset_filename):
        print ('initializing data preparer')
        self.dataset_filename = dataset_filename

    def prepare_data(self):
        print('preparing data')
        with open(self.dataset_filename, 'r') as input_csv_file:
            reader = csv.reader(input_csv_file)
            self.csv_file = [row for row in reader]

        return self.csv_file