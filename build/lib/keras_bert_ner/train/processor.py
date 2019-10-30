# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import json
import numpy as np
from ..bert4keras.utils import Tokenizer

__all__ = ["Processor"]


class Processor(object):
    
    def __init__(self, train_path, token_dict):
        self.train_path = train_path
        self.tokenizer = Tokenizer(token_dict)
    
    def get_tags(self):
        tags = set()
        train_data = self.get_data(self.train_path)
        for item in train_data:
            for tag in item[1].split(" "):
                tags.add(tag)
        # PAD-tag用X
        self.tag2id = {list(tags)[i]:i for i in range(len(tags))}
        self.tag2id["X"] = len(self.tag2id)
        self.id2tag = {self.tag2id[i]:i for i in self.tag2id}
        return self.tag2id, self.id2tag
            
    def get_data(self, path):
        with codecs.open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def get_bert_inputs(self, path, max_len):
        srcs = self.get_data(path)
        src_data, src_tags = [], []
        for item in srcs:
            src_data.append(item[0])
            src_tags.append(item[1])
        tokens, segs, tags = [], [], []
        for item in src_data:
            res = self.tokenizer.encode(item, first_length=max_len)
            tokens.append(np.array(res[0]))
            segs.append(np.array(res[1]))
        max_len -= 2
        for item in src_tags:
            len_item = len(item.split(" "))
            if len_item >= max_len:
                tags.append(["X"]+item.split(" ")[:max_len]+["X"])
            else:
                tags.append(["X"]+item.split(" ")+["X"]*(max_len-len_item+1))
        tags = [[self.tag2id[item] for item in term[1:]] for term in tags]
        tags = np.expand_dims(tags, axis=-1)
        return tokens, segs, tags