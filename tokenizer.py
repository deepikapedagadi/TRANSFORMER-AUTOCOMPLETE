class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<pad>": 0}
        self.idx2word = {0: "<pad>"}

    def encode(self, text):
        tokens = text.lower().split()
        ids = []
        for tok in tokens:
            if tok not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[tok] = idx
                self.idx2word[idx] = tok
            ids.append(self.word2idx[tok])
        return ids

    def decode(self, idx):
        #return " ".join(self.idx2word.get(i, "?") for i in ids)
        return self.idx2word.get(idx, "?")

    #@property
    #def vocab_size(self):
        #return len(self.word2idx)