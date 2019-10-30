# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("../..")

import json
import logging
from flask import Flask, request
from keras_bert_ner.utils.predict import build_trained_model, get_model_inputs


class Args(object):
    def __init__(self, configs):
        self.model_path = configs.get("model_path")
        self.model_name = configs.get("model_name")
        self.bert_vocab = configs.get("bert_vocab")
        self.device_map = configs.get("device_map")
        self.max_len = configs.get("max_len")


configs = {
    "model_path": "../models",
    "model_name": "ALBERT-IDCNN-CRF.h5",
    "bert_vocab": "/home1/liushaoweihua/pretrained_lm/albert_tiny_250k/vocab.txt",
    "device_map": "cpu",
    "max_len": 512
}


args = Args(configs)
tokenizer, id2tag, viterbi_decoder = build_trained_model(args=args)


def parse(text):
    tokens, segs = get_model_inputs(tokenizer, [text], max_len=args.max_len)
    decode_res = viterbi_decoder.decode([tokens, segs])
    decode_res = [id2tag[item] for item in decode_res[0] if id2tag[item] != "X"]
    res = get_entities([list(text), decode_res])
    return "|".join(res)


def get_entities(inputs):
    text, tags = inputs
    entities = []
    entity = ""
    for text_item, tag_item in zip(text, tags):
        if tag_item == "B":
            entity += text_item
        elif tag_item == "I":
            if entity != "":
                entity += text_item
        else:
            if entity != "":
                entities.append(entity)
                entity = ""
    return entities


def create_app():
    app = Flask(__name__)
    @app.route("/", methods=["GET"])
    def callback():
        text = request.args.get("s") or "EOF"
        app.logger.info("[RECEIVE]: {}".format(text))
        res = parse(text)
        app.logger.info("[SEND]: {}".format(res))
        return json.dumps({"text": text, "entities": res}, ensure_ascii=False, indent=4)
    return app


app = create_app()


if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    app.logger.info("Initializing complete!")