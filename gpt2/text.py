import json
import regex as re
import random
from collections import namedtuple


SpecialTokensTuple = namedtuple('SpecialTokensTuple', ['pad', 'bos', 'eos'])


class BPEVocab:
    @classmethod
    def from_files(cls, vocab_path, codes_path, special_tokens):
        with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
            vocab = json.load(vocab_file)

        with open(codes_path, 'r', encoding='utf-8') as codes_file:
            codes = [c.strip() for c in codes_file.readlines()]

            if codes[0].startswith('#version:'):
                codes = codes[1:]

            codes = [tuple(c.split()) for c in codes if c]

        return cls(vocab, codes, special_tokens)

    @staticmethod
    def _bytes_to_unicode():
        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("¡"), ord("¬") + 1)) + \
             list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]

        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
                
        cs = [chr(n) for n in cs]
        mapping = dict(zip(bs, cs))

        return mapping

    @staticmethod
    def _get_pairs(sequence, dropout=0):
        if len(sequence) < 2:
            return set()
        return set(p for p in zip(sequence[:-1], sequence[1:])
                   if not dropout or random.random() > dropout)

    def __init__(self, vocab, codes, special_tokens):
        assert isinstance(special_tokens, SpecialTokensTuple)

        filtered_special_tokens = [t for t in special_tokens if t not in vocab]
        special_token2id = {t: i for i, t in enumerate(filtered_special_tokens)}
        token2id = {t: i + len(filtered_special_tokens) for t, i in vocab.items()}
        token2id.update(special_token2id)

        self.special_tokens = special_tokens
        self.n_new_tokens = len(filtered_special_tokens)
        self.token2id = token2id
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.bpe_ranks = dict(zip(codes, range(len(codes))))
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

        for token_name, token in special_tokens._asdict().items():
            setattr(self, token_name + '_id', self.token2id[token])

    def __len__(self):
        return len(self.token2id)

    @property
    def n_special_tokens(self):
        return self.n_new_tokens

    @property
    def special_tokens_ids(self):
        return [self.token2id[t] for t in self.special_tokens]

    def _bpe(self, token, dropout=0):
        if token in self.cache and not dropout:
            return self.cache[token]

        word = token
        pairs = self._get_pairs(word, dropout)

        if not pairs:
            return word

        while pairs:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)

            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word, dropout)

        self.cache[token] = word

        return word

    def text2ids(self, text, dropout=0):
        tokens = [''.join([self.byte_encoder[b] for b in t.encode('utf-8')]) for t in re.findall(self.pat, text)]
        bpe_tokens = sum([self._bpe(tuple(t), dropout) for t in tokens], tuple())
        ids = [self.token2id[t] for t in bpe_tokens]

        return ids

    def ids2text(self, ids):
        text = ''.join([self.id2token[id] for id in ids])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')

        return text


def get_vocab(vocab_path, codes_path):
    special_tokens = SpecialTokensTuple(pad='<pad>', bos='<|endoftext|>', eos='<|endoftext|>')
    vocab = BPEVocab.from_files(vocab_path, codes_path, special_tokens) 

    return vocab
