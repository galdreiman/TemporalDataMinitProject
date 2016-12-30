from seq_minig.BaseMiner import BaseMiner


class Bide(BaseMiner):

    def __init__(self):
        BaseMiner.__init__(self)
        print ('init BIDE')

    def mine_sequence(self):
        print('Bide: nime_sequence')