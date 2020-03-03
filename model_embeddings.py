#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Project
model_embeddings.py: Embeddings for the AttentionXML
Austin Murphy <amurphy5@stanford.edu>

Helped by CS224n 2019-20 HW4
"""

import torch.nn as nn

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing note and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size

        # default values
        self.note_embeds = None

#         src_pad_token_idx = vocab.src['<pad>']
        src_pad_token_idx = 0

        ### YOUR CODE HERE (~2 Lines)
        ### TODO - Initialize the following variables:
        ###     self.notes_ (Embedding Layer for notes vocab)
        ###
        ### Note:
        ###     1. `vocab` object contains one vocabulary object:
        ###            `vocab.notes` for the progress notes
        ###     2. You can get the length of a specific vocabulary by running:
        ###             `len(vocab.<specific_vocabulary>)`
        ###     3. Remember to include the padding token for the specific vocabulary
        ###        when creating your Embedding.
        ###
        ### Use the following docs to properly initialize these variables:
        ###     Embedding Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
        
        self.note_embeds = nn.Embedding(num_embeddings=len(vocab.notes_), embedding_dim=self.embed_size, 
                                         padding_idx=src_pad_token_idx, max_norm=None, 
                                         norm_type=2.0, scale_grad_by_freq=False, 
                                         sparse=False, _weight=None)
         
        ### END YOUR CODE



