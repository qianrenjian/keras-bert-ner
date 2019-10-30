# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../..")

import os
import codecs
import json
from keras_bert_ner.utils.predict_helper import get_predict_args_parser
from keras_bert_ner.utils.predict import build_trained_model, get_model_inputs

def run_test():
    args = get_predict_args_parser()
    if True:
        import sys
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    tokenizer, id2tag, viterbi_decoder = build_trained_model(args=args)
    with codecs.open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    tokens, segs = get_model_inputs(tokenizer, test_data, max_len=args.max_len)
    decode_res = viterbi_decoder.decode([tokens, segs])
    decode_res = [[id2tag[item] for item in term if id2tag[item] != "X"] for term in decode_res]
    res = [[data_item, decode_item] for data_item, decode_item in zip(test_data, decode_res)]
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    with codecs.open(os.path.join(args.output_path, "test_outputs.txt"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
    
if __name__ == '__main__':
    run_test()