from collections import deque
from collections import OrderedDict 

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from src.ml.classifiers.LowMemConv import LowMemConvBase


def getParams():
    #Format for this is to make it work easily with Optuna in an automated fashion.
    #variable name -> tuple(sampling function, dict(sampling_args) )
    params = {
        'channels'     : ("suggest_int", {'name':'channels', 'low':32, 'high':1024}),
        'log_stride'   : ("suggest_int", {'name':'log2_stride', 'low':2, 'high':9}),
        'window_size'  : ("suggest_int", {'name':'window_size', 'low':32, 'high':512}),
        'embd_size'    : ("suggest_int", {'name':'embd_size', 'low':4, 'high':64}),
    }
    return OrderedDict(sorted(params.items(), key=lambda t: t[0]))


def initModel(**kwargs):
    new_args = {}
    for x in getParams():
        if x in kwargs:
            new_args[x] = kwargs[x]
            
    return MalConv(**new_args)


class MalConv(LowMemConvBase):
    def __init__(self, out_size:int=2, channels:int=128, window_size:int=512, stride:int=512, embd_size:int=8, log_stride:int=None, thresh:float=0.5, vocabulary_size:int=257, padding_idx:int=0):
        super(MalConv, self).__init__()
        self.out_size = out_size
        self.thresh = thresh
        self.vocabulary_size = vocabulary_size
        self.padding_idx = padding_idx
        self.embd = nn.Embedding(vocabulary_size, embd_size, padding_idx=padding_idx)
        if not log_stride is None:
            stride = 2**log_stride
    
        self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=stride, bias=True)
        self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=stride, bias=True)

        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
        
    
    def processRange(self, x):
        x = self.embd(x)
        x = torch.transpose(x,-1,-2)
         
        cnn_value = self.conv_1(x)
        gating_weight = torch.sigmoid(self.conv_2(x))
        
        x = cnn_value * gating_weight
        
        return x
    
    def forward(self, x):
        post_conv = x = self.seq2fix(x)
        
        penult = x = F.relu(self.fc_1(x))
        x = self.fc_2(x)

        if self.out_size == 1:
            x = torch.squeeze(x)
        
        return x, penult, post_conv

    def get_prob(self, outputs):
        if self.out_size == 2:
            probs = F.softmax(outputs, dim=-1)
        else:
            probs = torch.sigmoid(outputs)
        return probs

    def predict(self, x: torch.Tensor):
        outputs, _, _ = self.forward(x)
        y_prob, y_pred = self.predict_from_outputs(outputs)
        return y_prob, y_pred

    def predict_from_outputs(self, outputs: torch.Tensor):
        prob = self.get_prob(outputs).detach().numpy()
        if self.out_size == 1:
            return prob, 1 if prob >= self.thresh else 0
        elif self.out_size == 2:
            return prob[0, 1], 1 if prob[0, 1] >= self.thresh else 0

    def predict_label(self, x: torch.Tensor):
        if self.out_size == 1:
            with torch.no_grad():
                x, penult, post_conv = self(x)
                outputs = F.sigmoid(x)
            return outputs.detach().numpy() >= self.thresh
        elif self.out_size == 2:
            with torch.no_grad():
                x, penult, post_conv = self(x)
                outputs = F.softmax(x, dim=-1)
            return outputs.detach().numpy()[0, 1] >= self.thresh

    def predict_proba(self, x: torch.Tensor):
        if self.out_size == 1:
            with torch.no_grad():
                x, penult, post_conv = self(x)
                outputs = F.sigmoid(x)
            return outputs.detach().numpy()
        elif self.out_size == 2:
            with torch.no_grad():
                x, penult, post_conv = self(x)
                outputs = F.softmax(x, dim=-1)
            return outputs.detach().numpy()[0, 1]
