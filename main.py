import DataUtils as ut
from LogisticRegression import LogisticRegression

def main():
    test_setosa()

def test_setosa():
    data = ut.get_setosa()
    ls = LogisticRegression(0.01, 1000, data, [0, 1, 2, 3])
    ls.train()
    print(ls.test())

if __name__ == '__main__':
    main()