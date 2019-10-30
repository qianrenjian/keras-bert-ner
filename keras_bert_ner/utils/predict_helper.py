# -*- coding: utf-8 -*-

import os
import argparse

__all__ = ["get_predict_args_parser"]

if os.name == "nt":
    bert_dir = ""
    root_dir = ""
else:
    bert_dir = "/home1/liushaoweihua/pretrained_lm/bert_chinese/"
    root_dir = "/home1/jupyterlab/Keras-Bert-Ner/"
    
def get_predict_args_parser():
    
    parser = argparse.ArgumentParser()
    
    data_group = parser.add_argument_group("Data File Paths", "Config the train/dev/test file paths")
    data_group.add_argument("-test_data", type=str, default=os.path.join(root_dir, "data", "test.txt"), required=True, help="(REQUIRED) Test data path")
    data_group.add_argument("-max_len", type=int, default=64, help="(OPTIONAL) Max sequence length. Default is 64")
    
    model_group = parser.add_argument_group("Model Paths", "Config the model paths")
    model_group.add_argument("-model_path", type=str, default=os.path.join(root_dir, "models"), required=True, help="(REQUIRED) Model path")
    model_group.add_argument("-model_name", type=str, default="BERT-BILSTM-CRF.h5", required=True, help="(REQUIRED) Model name")
    
    output_group = parser.add_argument_group("Output Paths", "Config the output paths")
    output_group.add_argument("-output_path", type=str, default=os.path.join(root_dir, "test_outputs"), help="(OPTIONAL) Output file paths")
    
    bert_group = parser.add_argument_group("BERT File paths", "Config the vocab of a pretrained or fine-tuned BERT model")
    bert_group.add_argument("-bert_vocab", type=str, default=os.path.join(bert_dir, "vocab.txt"), required=True, help="(REQUIRED) vocab.txt")
    
    action_group = parser.add_argument_group("Action Configs", "Config the actions during running")
    action_group.add_argument("-device_map", type=str, default="cpu", help="(OPTIONAL) Use CPU/GPU to train. If use CPU, then 'cpu'. If use GPU, then assign the devices, such as '0'. Default is 'cpu'")
    
    return parser.parse_args()