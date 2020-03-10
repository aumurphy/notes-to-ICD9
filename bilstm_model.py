#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 
bilstm_model.py: BiLSTM Model
Austin Murphy <amurphy5@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

from model_embeddings import ModelEmbeddings


class BiLSTM(nn.Module):
    """
    Simple Neural Multilabel Classification Model:
    - Bidirectional LSTM Encoder
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(BiLSTM, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        print("vocab.num_labels: ", vocab.num_labels)
        self.num_labels = vocab.num_labels
        
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size, 
                               bias=True, 
#                                dropout=self.dropout_rate,
                               bidirectional=True)
        
        self.dropout1 = nn.Dropout2d(p=self.dropout_rate)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)
        
        self.attention_projection = nn.Linear(in_features=2*hidden_size, 
                                                 out_features=self.num_labels, 
                                                 bias=False)
        self.attention_softmax = nn.Softmax(dim=0)
        self.labels_projection = nn.Linear(in_features=2*hidden_size, 
                                          out_features=1, 
                                          bias=False)
        
    def forward(self, in_sents: List[List[str]], target_labels: List[List[int]]):
        # Compute sentence lengths
        in_sents_lengths = [len(s) for s in in_sents]
        
        # Convert list of lists into tensors
        source_padded = self.vocab.notes_.to_input_tensor(in_sents, device=self.device)   # Tensor: (src_len, b)
        
#         print("len(in_sents): ", len(in_sents))
#         print("source_padded.shape: ", source_padded.shape)
#         print(target_labels.shape)

        X = self.model_embeddings.note_embeds(source_padded)
#         print("X.shape: ", X.shape)
        
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X)
#         print("enc_hiddens.shape: ", enc_hiddens.shape)
#         print("last_hidden.shape: ", last_hidden.shape)
#         print("last_cell.shape: ", last_cell.shape)

#         enc_hiddens = self.dropout1(enc_hiddens)
        
        alpha = self.attention_projection(enc_hiddens)
#         print("alpha.shape: ", alpha.shape)

        alpha = self.dropout1(alpha)
        
        alpha_soft = self.attention_softmax(alpha)
#         print(np.sum(alpha_soft.detach().numpy(),axis=0))
        
#         print("alpha_soft.shape: ", alpha_soft.shape)
#         print("alpha_permuted shape: ", alpha_soft.permute([2,1,0]).shape)
        
        M = torch.bmm(alpha_soft.permute([1,2,0]), enc_hiddens.permute([1,0,2]))
#         print("M.shape: ", M.shape)
#         torch.stack(combined_outputs, dim=0)

#         M = self.dropout2(M)

        scores = self.labels_projection(M)
#         print(scores)
#         print(F.sigmoid(scores))
        scores = torch.sigmoid(torch.squeeze(scores,-1))
#         print("scores.shape: ", scores.shape)
#         print("scores squeezed shape: ", torch.squeeze(scores,-1).shape)
#         print("scores: \n", scores)
#         scores = 0.
        
        return scores
    
    
    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass
        
    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.note_embeds.weight.device
    
    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
#         params = torch.load(model_path, map_location=lambda storage, loc: storage)
#         args = params['args']
#         model = NMT(vocab=params['vocab'], **args)
#         model.load_state_dict(params['state_dict'])
        
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = BiLSTM(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

#         params = {
#             'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
#             'vocab': self.vocab,
#             'state_dict': self.state_dict()
#         }
        
        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
        