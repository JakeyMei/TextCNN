from itertools import chain
import json
from collections import Counter
from typing import List
import torch

from utils import pad_sents

class VocabEntry:
    
    def __init__(self, word2id=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<PAD>'] = 0
            self.word2id['<UNK>'] = 1
        self.unk_id = self.word2id['<UNK>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """
        获取word的idx
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """
        判断词表是否包含改单词
        """
        return word in self.word2id

    def __setitem__(self, key, val):
        raise ValueError('vovabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % (len(self.word2id))

    def add(self, word):
        """
        增加单词
        """
        if word not in self.word2id:
            wid = self.word2id[word] = len(self.word2id)
            self.id2word[wid] = word
            return wid
        else: 
            return self.word2id[word]

    def word2indices(self, sents):
        """
        编码: 将sents转为number index, 文字转下标
        """
        if type(sents[0]) == list:
            return [[self.word2id.get(w, self.unk_id) for w in sent] for sent in sents]
        else:
            return [self.word2id[w, self.unk_id] for w in sents]

    def indices2words(self, idx):
        """
        解码：下标转文字
        """
        return [self.id2word[id] for id in idx]

    def to_input_tensor(self, sents:List[List[str]], device: torch.device):
        """
        将原始句子list转为tensor, 同时将句子PAD为max_len_sent
        """
        sents = self.word2indices(sents)
        sents = pad_sents(sents, self.word2id['<PAD>'])
        sents_var = torch.tensor(sents, device=device)
        return sents_var
    
    @staticmethod
    def from_corpus(corpus, size, min_freq=3):
        """
        从给定的语料中创建VocabEntry
        """
        vocab_enty = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = word_freq.most_common(size-2) # 提取size-2的高频词汇和出现次数
        valid_words = [word for word, value in valid_words if value>=min_freq]
        for word in valid_words:
            vocab_enty.add(word)
        return vocab_enty

class Vocab:
    def __init__(self, src_vocab: VocabEntry, labels:dict):
        self.vocab = src_vocab
        self.labels = labels

    @staticmethod
    def build(src_sents, labels, vocab_size, min_freq):
        print('initialize source vocabulary...')
        src = VocabEntry.from_corpus(src_sents, vocab_size, min_freq)

        return Vocab(src, labels)

    def save(self, file_path):
        with open(file_path, 'w') as fin:
            json.dump(dict(src_word2id=self.vocab.word2id, labels=self.self.labels), fin, indent=2)

    @staticmethod
    def load(file_path):
        with open(file_path, 'r') as fout:
            entry = json.loads(fout)
        src_word2id = entry['src_word2id']
        labels = entry['labels']
        return Vocab(VocabEntry(src_word2id), labels)
    
    def __repr__(self):
        return 'Vocab(source %d words)' % (len(self.vocab))
