from seq_minig.BaseMiner import BaseMiner
from seq_minig.bide.pymining import seqmining

class Bide(BaseMiner):

    def __init__(self):
        BaseMiner.__init__(self)
        print ('init BIDE')

    def mine_sequence(self, seqs):
        print('Bide: nime_sequence')

        #seqs = ( 'caabc', 'abcb', 'cabc', 'abbca')
        freq_seqs = seqmining.freq_seq_enum(seqs, 1)
        sorted_freq_seqs = sorted(freq_seqs)


        # for x in sorted_freq_seqs: print (x)

        return sorted_freq_seqs
