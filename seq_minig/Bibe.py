from seq_minig.BaseMiner import BaseMiner


class Bibe(BaseMiner):

    def __init__(self):
        BaseMiner.__init__(self)
        print ('init BIBE')

    def mine_sequence(self):
        print('Bibe: nime_sequence')