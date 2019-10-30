# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import pickle
import numpy as np
import keras
from .processor import Processor
from .models import NER_Model
from .callbacks import NER_Callbacks
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

__all__ = ["train"]


def train(args):
    if args.device_map != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_map
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    token_dict = {}
    with codecs.open(args.bert_vocab, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            token_dict[token] = len(token_dict)
    
    processor = Processor(args.train_data, token_dict)
    
    tag2id, id2tag = processor.get_tags()
    numb_tags = len(tag2id)
    # save tag2id/id2tag to save_path
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    with codecs.open(os.path.join(args.save_path, "tag2id.pkl"), "wb") as f:
        pickle.dump(tag2id, f)
    with codecs.open(os.path.join(args.save_path, "id2tag.pkl"), "wb") as f:
        pickle.dump(id2tag, f)
    
    train_tokens, train_segs, train_tags = processor.get_bert_inputs(args.train_data, args.max_len)
    trains = [np.array(train_tokens), np.array(train_segs)]
    train_tags = np.array(train_tags)
    if args.do_eval:
        dev_tokens, dev_segs, dev_tags = processor.get_bert_inputs(args.dev_data, args.max_len)
        devs = [[np.array(dev_tokens), np.array(dev_segs)], np.array(dev_tags)]
    else:
        devs = None
        
    # build model
    if args.albert:
        bert_type = "ALBERT"
    else:
        bert_type = "BERT"
    if args.crf_only:
        model_type = "CRF"
    elif args.model_type == "cnn":
        model_type = "IDCNN-CRF"
    elif args.model_type == "rnn" and args.cell_type == "bilstm":
        model_type = "BILSTM-CRF"
    elif args.model_type == "rnn" and args.cell_type == "bigru":
        model_type = "BIGRU-CRF"
    else:
        raise ValueError("Expected to have model_type 'rnn' with cell_type 'bilstm'/'bigru' or model_type 'cnn' with cell_type 'idcnn', but got {}/{}".format(args.model_type, args.cell_type))
    save_path = os.path.join(args.save_path, "{}-{}.h5".format(bert_type,model_type))
    
    model_configs = {
        "bert_config": args.bert_config,
        "bert_checkpoint": args.bert_checkpoint,
        "albert": args.albert,
        "model_type": args.model_type,
        "cell_type": args.cell_type,
        "rnn_units": args.rnn_units,
        "rnn_layers": args.rnn_layers,
        "cnn_filters": args.cnn_filters,
        "cnn_kernel_size": args.cnn_kernel_size,
        "cnn_blocks": args.cnn_blocks,
        "crf_only": args.crf_only,
        "dropout_rate": args.dropout_rate,
        "max_len": args.max_len,
        "numb_tags": numb_tags
    }
    model = NER_Model(model_configs).build()
    model.compile(
        optimizer=keras.optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss=crf_loss,
        metrics=[crf_viterbi_accuracy]
    )
    
    if args.best_fit:
        callback_configs = {
            "early_stop_patience": args.early_stop_patience,
            "reduce_lr_patience": args.reduce_lr_patience,
            "reduce_lr_factor": args.reduce_lr_factor,
            "save_path": save_path
        }
        callbacks = NER_Callbacks(id2tag).best_fit_callbacks(callback_configs)
        epochs = args.max_epochs
    else:
        callbacks = NER_Callbacks(id2tag).callbacks()
        epochs = args.hard_epochs
    
    model.fit(
        x=trains,
        y=train_tags,
        batch_size=args.batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=devs
    )
    
    # save model if not in best_fit mode
    if not args.best_fit:
        model.save(save_path)