import random
import torch
from torch.utils.data import IterableDataset


class LinearChainDataset(IterableDataset):
    """
    This is a dataset simulating sequence labeling in NLP.
    An item looks like:
        number1 operator number2 = number3
    where:
    - a number can be: 0.133/-0.133/.333/3.
    - the label segments the numbers and operators, following the BMES-style in sequence labeling
    """
    def __init__(self, length=25) -> None:
        super().__init__()

        self.length = length

        vocab_ = [str(i) for i in range(10)] + [".", "+", "-", "*", "/", "=", "^", "$", "@"]
        self.word2idx = {ele: idx for idx, ele in enumerate(vocab_)}
        self.idx2word = {idx: ele for idx, ele in enumerate(vocab_)}

        vocab_ = ["B", "M", "E", "S", "@"]
        self.label2idx = {ele: idx for idx, ele in enumerate(vocab_)}
        self.idx2label = {idx: ele for idx, ele in enumerate(vocab_)}

        self.pad = "@"

    def rand_word(self):
        max_num = 10**(self.length // 8)
        x = random.randint(0, 1)
        ret = ""
        if x == 1:
            ret += "-"
        else:
            ret += ""
        x = random.randint(0, 3)
        if x == 0:
            ret += str(random.randint(1, max_num))
        elif x == 1:
            ret += str(random.randint(1, max_num))
            ret += "."
        elif x == 2:
            ret += str(random.randint(1, max_num))
            ret += "."
            ret += str(random.randint(1, max_num))
        else:
            ret += "."
            ret += str(random.randint(1, max_num))
        return ret

    def label_from_length(self, num):
        if num == 1:
            return "S"
        else:
            return "B" + "M" * (num - 2) + "E"

    def generate(self):
        x1 = self.rand_word()
        x2 = self.rand_word()
        y = self.rand_word()
        op = "+-*/"[random.randint(0, 3)]
        sent = [x1, op, x2, "=", y]
        label = [self.label_from_length(len(ele)) for ele in sent]
        sent = "".join(sent)
        sent += (self.length - len(sent)) * self.pad
        label = "".join(label)
        label += (self.length - len(label)) * self.pad
        sent = torch.tensor([self.word2idx[ele] for ele in sent])
        label = torch.tensor([self.label2idx[ele] for ele in label])
        masks = sent != self.word2idx[self.pad]
        return sent, masks, label

    def __iter__(self):
        while True:
            yield self.generate()

    def to_sent(self, tensor):
        tensor = tensor[0] if len(tensor.shape) == 2 else tensor
        sent = [self.idx2word[ele] for ele in tensor.tolist()]
        return "".join(sent)

    def to_label(self, tensor):
        tensor = tensor[0] if len(tensor.shape) == 2 else tensor
        label = [self.idx2label[ele] for ele in tensor.tolist()]
        return "".join(label)


class SkipChainDataset(IterableDataset):
    """
    This is a dataset simulating a general graph.
    An item looks like:
        _7____________5___E2__A5_____7__5____22___D7_@@@@@
         |                           |             |
         D                           D             D
    where:
    - 7 is labeled as "D" since one of 7s is a successor of a "D"
    - similarly, 5 is labeled as "A" and 2 is labeled as "E"
    - 3 numbers (7, 5, 2) occurs in an item, and each occurs 3 times,
        
    For graphical models, we can build one ternary factor or 3 binary factors:
            7                   7
           / \         or       |
          7---7              7_/ \_7
    """
    def __init__(self, length, skip_as_ternary) -> None:
        super().__init__()

        self.length = length
        self.skip_as_ternary = skip_as_ternary

        vocab_ = "ABCDEFG1234567_@"
        self.word2idx = {ele: idx for idx, ele in enumerate(vocab_)}
        self.idx2word = {idx: ele for idx, ele in enumerate(vocab_)}

        vocab_ = "ABCDEFG_@"
        self.label2idx = {ele: idx for idx, ele in enumerate(vocab_)}
        self.idx2label = {idx: ele for idx, ele in enumerate(vocab_)}

        self.pad = "@"

    def generate(self):
        seqlen = random.randint(int(self.length * 0.6), self.length)
        sent = list("_" * seqlen + self.pad * (self.length - seqlen))
        label = list("_" * seqlen + self.pad * (self.length - seqlen))
        names = list("ABCDEFG")
        random.shuffle(names)
        names = names[:3]
        nicks = list("1234567")
        random.shuffle(nicks)
        nicks = nicks[:3]
        locs = list(range(seqlen))
        random.shuffle(locs)

        bin_edges = [(i, i + 1) for i in range(seqlen - 2)]
        ter_edges = []
        for i in range(3):
            loc1 = locs.pop()
            while loc1 - 1 not in locs:
                locs.insert(0, loc1)
                loc1 = locs.pop()
            locs.remove(loc1 - 1)
            loc2 = locs.pop()
            loc3 = locs.pop()
            sent[loc1 - 1] = names[i]
            sent[loc1], sent[loc2], sent[loc3] = nicks[i], nicks[i], nicks[i]
            label[loc1], label[loc2], label[loc3] = names[i], names[i], names[i]

            if not self.skip_as_ternary:
                bin_edges.append((loc1, loc2))
                bin_edges.append((loc2, loc3))
                bin_edges.append((loc1, loc3))
            else:
                ter_edges.append((loc1, loc2, loc3))

        bin_edges_real_len = len(bin_edges)
        bin_edges.extend([(0, 0) for _ in range(self.length - seqlen)])

        sent = "".join(sent)
        label = "".join(label)

        nodes = torch.tensor([self.word2idx[ele] for ele in sent])
        bin_edges = torch.tensor(bin_edges)
        ter_edges = torch.tensor(ter_edges)
        node_masks = nodes != self.word2idx[self.pad]
        bin_edge_masks = torch.zeros([len(bin_edges)]).bool()
        bin_edge_masks[:bin_edges_real_len] = True
        ter_edge_masks = torch.ones([len(ter_edges)]).bool()
        label = torch.tensor([self.label2idx[ele] for ele in label])
        return nodes, node_masks, bin_edges, bin_edge_masks, ter_edges, ter_edge_masks, label

    def __iter__(self):
        while True:
            yield self.generate()

    def to_sent(self, tensor):
        tensor = tensor[0] if len(tensor.shape) == 2 else tensor
        sent = [self.idx2word[ele] for ele in tensor.view(-1).tolist()]
        return "".join(sent)

    def to_label(self, tensor):
        tensor = tensor[0] if len(tensor.shape) == 2 else tensor
        label = [self.idx2label[ele] for ele in tensor.view(-1).tolist()]
        return "".join(label)
