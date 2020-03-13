#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 
node.py: Node (Parent) Model
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
from bilstm_model import BiLSTM


class Node2(nn.Module):
    """
    Node Class that inherits the BiLSTM models created in bilstm_model.py
    Beginning Node models have 2 BiLSTMs, future versions can have more
    
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate):
        super(Node2, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        print("vocab.num_labels: ", vocab.num_labels)
        self.num_labels = vocab.num_labels
        
        self.encoder0 = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size, 
                               bias=True, 
                               # dropout=self.dropout_rate,
                               bidirectional=True)

        self.encoder1 = nn.LSTM(input_size=embed_size,
                               hidden_size=hidden_size, 
                               bias=True, 
                               # dropout=self.dropout_rate,
                               bidirectional=True)
        
        self.dropout1 = nn.Dropout()
        
        self.attention_projection = nn.Linear(in_features=2*hidden_size, 
                                                 out_features=self.num_labels, 
                                                 bias=False)
        self.attention_softmax = nn.Softmax(dim=0)
        self.labels_projection = nn.Linear(in_features=2*hidden_size, 
                                          out_features=1, 
                                          bias=False)
        
    def forward(self, in_sents: List[List[str]], target_labels: List[List[int]]):
        
        # in_sents should be (1000, whatever)
        # split in half
        num_notes = len(in_sents)
        length_of_each_note = int(len(in_sents[0])/2)
        
        
        # Convert list of lists into tensors
        source_padded = self.vocab.notes_.to_input_tensor(in_sents, device=self.device)   
                        # Tensor: (src_len, b)
#         print(num_notes)

        X = self.model_embeddings.note_embeds(source_padded)
#         print("X.shape: ", X.shape) # (1000, 16, 256)
#         print(in_sents[0])
        X0 = X[:length_of_each_note,:,:]
        X1 = X[length_of_each_note:,:,:]
        
        enc_hiddens0, _ = self.encoder0(X0)
        enc_hiddens1, _ = self.encoder1(X1)
#         print(enc_hiddens0.shape)
#         print(enc_hiddens1.shape)
        
        enc_hiddens = torch.cat([enc_hiddens0, enc_hiddens1],0)
#         print(enc_hiddens.shape)
        
#         scores0 = self.first_bilstm(in_sents0,target_labels)
#         scores1 = self.second_bilstm(in_sents1,target_labels)
#         print(scores0[0])
#         print(scores1[0])

        alpha = self.attention_projection(enc_hiddens)
#         print("alpha.shape: ", alpha.shape)

        alpha = self.dropout1(alpha)
        
                
        alpha_soft = self.attention_softmax(alpha)
#         print(np.sum(alpha_soft.detach().numpy(),axis=0))
        
        M = torch.bmm(alpha_soft.permute([1,2,0]), enc_hiddens.permute([1,0,2]))
#         print("M.shape: ", M.shape)
#         torch.stack(combined_outputs, dim=0)

        scores = self.labels_projection(M)
    
        scores = torch.sigmoid(torch.squeeze(scores,-1))
    
#         print("scores.shape: ", scores.shape)
        
        return scores
        


        

        
# #         print("alpha_soft.shape: ", alpha_soft.shape)
# #         print("alpha_permuted shape: ", alpha_soft.permute([2,1,0]).shape)
        
#         M = torch.bmm(alpha_soft.permute([1,2,0]), enc_hiddens.permute([1,0,2]))
# #         print("M.shape: ", M.shape)
# #         torch.stack(combined_outputs, dim=0)

#         scores = self.labels_projection(M)
# #         print(scores)
# #         print(F.sigmoid(scores))
#         scores = torch.sigmoid(torch.squeeze(scores,-1))
# #         print("scores.shape: ", scores.shape)
# #         print("scores squeezed shape: ", torch.squeeze(scores,-1).shape)
# #         print("scores: \n", scores)
# #         scores = 0.
        
#         return scores
    
    
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
        model = Node(vocab=params['vocab'], **args)
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
        