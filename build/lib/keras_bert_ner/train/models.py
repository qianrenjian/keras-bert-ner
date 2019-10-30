# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..bert4keras.bert import set_gelu, load_pretrained_model
from keras.layers import *
from keras.models import Model
from keras_contrib.layers import CRF
set_gelu("tanh")

class NER_Model(object):
    
    def __init__(self, model_configs):
        self.bert_config = model_configs.get("bert_config")
        self.bert_checkpoint = model_configs.get("bert_checkpoint")
        self.albert = model_configs.get("albert")
        self.model_type = model_configs.get("model_type")
        self.cell_type = model_configs.get("cell_type")
        self.rnn_units = model_configs.get("rnn_units")
        self.rnn_layers = model_configs.get("rnn_layers")
        self.cnn_filters = model_configs.get("cnn_filters")
        self.cnn_kernel_size = model_configs.get("cnn_kernel_size")
        self.cnn_blocks = model_configs.get("cnn_blocks")
        self.crf_only = model_configs.get("crf_only")
        self.dropout_rate = model_configs.get("dropout_rate")
        self.max_len = model_configs.get("max_len")
        self.numb_tags = model_configs.get("numb_tags")
        
    def build(self):
        # loading pretrained language model
        bert_model = load_pretrained_model(
            self.bert_config,
            self.bert_checkpoint,
            albert = self.albert
        )
        for l in bert_model.layers:
            l.trainable = True
            
        # build models
        token_in = Input(shape=(self.max_len,))
        seg_in = Input(shape=(self.max_len,))
        emb = bert_model([token_in, seg_in])
        emb = Lambda(lambda X: X[:, 1:])(emb)
        # loading downstream model
        if not self.crf_only:
            if self.model_type == "rnn":
                downstream_layers = self._rnn(emb)
            elif self.model_type == "cnn":
                downstream_layers = self._cnn(emb)
        else:
            downstream_layers = TimeDistributed(Dense(self.numb_tags, activation="relu"))(emb)
        crf = CRF(self.numb_tags, sparse_target=True)
        crf_out = crf(downstream_layers)
        model = Model([token_in, seg_in], crf_out)
        return model
    
    def _rnn(self, rnn):
        if self.cell_type == "bilstm":
            rnn_cell = LSTM
        elif self.cell_type == "bigru":
            rnn_cell = GRU
        for layer_idx in range(self.rnn_layers):
            rnn = Bidirectional(rnn_cell(units=self.rnn_units, return_sequences=True, recurrent_dropout=self.dropout_rate))(rnn)
        return rnn
    
    def _cnn(self, emb):
        stack_idcnn_layers = []
        for layer_idx in range(self.cnn_blocks):
            idcnn_block = self._idcnn_block()
            cnn = idcnn_block[0](emb)
            cnn = idcnn_block[1](cnn)
            cnn = idcnn_block[2](cnn)
            stack_idcnn_layers.append(cnn)
        stack_idcnn = concatenate(stack_idcnn_layers, axis=-1)
        return stack_idcnn
        
    def _dilation_conv1d(self, dilation_rate):
        return Conv1D(self.cnn_filters, self.cnn_kernel_size, padding="same", dilation_rate=dilation_rate)
    
    def _idcnn_block(self):
        idcnn_1 = self._dilation_conv1d(1)
        idcnn_2 = self._dilation_conv1d(1)
        idcnn_3 = self._dilation_conv1d(2)
        return [idcnn_1, idcnn_2, idcnn_3]