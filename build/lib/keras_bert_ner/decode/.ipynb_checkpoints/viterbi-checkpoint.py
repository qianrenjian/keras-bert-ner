# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Viterbi(object):
    
    def __init__(self, model, numb_tags):
        self.model = model
        self.numb_tags = numb_tags
        self.trans = self.get_trans()

    def get_trans(self):
        trans = {}
        crf_weights = self.model.layers[-1].get_weights()[0]
        for i in range(self.numb_tags):
            for j in range(self.numb_tags):
                trans[str(i)+"-"+str(j)] = crf_weights[i,j]
        return trans
    
    def viterbi(self, nodes):
        paths = nodes[0]
        for l in range(1, len(nodes)):
            paths_old, paths = paths, {}
            for n, ns in nodes[l].items():
                max_path, max_score = "", -1e10
                for p, ps in paths_old.items():
                    score = ns + ps + self.trans[p.split("-")[-1]+"-"+str(n)]
                    if score > max_score:
                        max_path, max_score = p+"-"+n, score
                paths[max_path] = max_score
        return self.max_in_dict(paths)
    
    def max_in_dict(self, dic):
        dic_reverse = {dic[path]:path for path in dic}
        return dic_reverse[max(dic_reverse)]

    def decode(self, data):
        pred_res = np.array(self.model.predict(data))
        decode_res = []
        for pred_item in pred_res:
            nodes = [dict([[str(idx), item] for idx, item in enumerate(term)]) for term in pred_item]
            decode_res.append([int(item) for item in self.viterbi(nodes).split("-")])
        return np.array(decode_res)