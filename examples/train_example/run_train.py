# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../..")

from keras_bert_ner.train.train_helper import get_train_args_parser
from keras_bert_ner.train.train import train

def run_train():
    args = get_train_args_parser()
    if True:
        import sys
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    train(args=args)
    
if __name__ == '__main__':
    run_train()