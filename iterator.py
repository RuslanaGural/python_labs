class Sequr:
    def __init__(self, seq):
        self.seq = seq
        self.leng = len(seq)

    def __iter__(self):
        return self

    def __next__(self):
        if self.leng == 0:
            raise StopIteration
        self.leng = self.leng - 1
        return self.seq[self.leng]


seq = Sequr([1, 2, 3, 4, 5])

for i in seq:
    print(i)
