from descritization.myEFD import myEFD


class TestEFD(object):
    def setUp(self):
        # All tests will be run with 5 classes
        self.efd = myEFD(5)

    def test_to_letter_rep(self):
        prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.efd.perform_discritization(prices)
        assert self.efd.discretize(15) == 'a'
        assert self.efd.discretize(35) == 'b'
        assert self.efd.discretize(55) == 'c'
        assert self.efd.discretize(75) == 'd'
        assert self.efd.discretize(95) == 'e'
        assert self.efd.discretize(105) == 'f'
