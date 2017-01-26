import csv


class DataPreperation(object):

    def __init__(self, dataset_filename):
        print ('initializing data preparer')
        self.dataset_filename = dataset_filename

    def read_csv_data(self):
        print('preparing data')
        with open(self.dataset_filename, 'r') as input_csv_file:
            reader = csv.reader(input_csv_file)
            self.csv_file = [row for row in reader]

        return self.csv_file

    def read_user_to_purchases_data(self):
        user_to_csv_map = dict()
        PRICE_INDEX = 3

        csv_data = self.read_csv_data()
        for row in csv_data:
            session_id = row[0]
            price = row[PRICE_INDEX]
            if session_id == 'Session ID':
                continue

            if session_id in user_to_csv_map:
                user_to_csv_map[(session_id)].append(price)
            else:
                user_to_csv_map[session_id] = [price]

        return user_to_csv_map