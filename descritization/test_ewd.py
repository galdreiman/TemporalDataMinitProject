from descritization.myEWD import myEWD


class TestEWD(object):
    def setUp(self):
        # All tests will be run with 5 classes
        self.ewd = myEWD(5)

    def test_to_letter_rep(self):
        prices = [10, 20, 30, 40, 50, 60]
        self.ewd.perform_discritization(prices)
        assert self.ewd.discretize(15) == 'a'
        assert self.ewd.discretize(25) == 'b'
        assert self.ewd.discretize(35) == 'c'
        assert self.ewd.discretize(45) == 'd'
        assert self.ewd.discretize(55) == 'e'
        assert self.ewd.discretize(65) == 'f'
