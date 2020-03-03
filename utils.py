#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 4
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import csv
import ast
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)

    max_length = 0
    for sent in sents:
        if len(sent) > max_length:
            max_length = len(sent)
    
    for sent in sents:
        for i in range(max_length-len(sent)):
            sent.append(pad_token)
        sents_padded.append(sent)


    ### END YOUR CODE

    return sents_padded


def read_corpus(file_path, column, sent_max_length, remove_stopwords):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """

    stop_words = set(stopwords.words('english'))
    dumb_words = set([',','dd','d.','ddd','-','d.d','dddddam','dd.d','ddd/dd','--','-d'])

    data = []
    all_labels = set()
    
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
#                 print(f'Column names are {", ".join(row[0])}')
                line_count += 1
            else:
                if column == 'text':
                    sent = nltk.word_tokenize(row[0])
                    if remove_stopwords:
                        sent = [i for i in sent if i not in stop_words]
                        sent = [i for i in sent if i not in dumb_words]
                    # cut off sentence lengths:
                    if sent_max_length > 0:
                        data.append(sent[:int(sent_max_length)])
                    else:
                        data.append(sent)
                else:
                    this_row_labs = ast.literal_eval(row[0])
                    data.append(this_row_labs)
                line_count += 1
        print(f'Processed {line_count} lines.')
        
    return data


def get_all_labels(file_path, KNOWN_NUM_LABELS):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """

    data = []
    all_labels = set()
    
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # skip over column names
                line_count += 1
            else:
                this_row_labs = ast.literal_eval(row[0])
                all_labels.update(this_row_labs)
                data.append(this_row_labs)
                line_count += 1
        print(f'Processed {line_count} lines.')
    
    return max(max(all_labels), len(all_labels), KNOWN_NUM_LABELS)


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


def ind_to_one_hot(labels_list, num_labels):
    """
    @param labels_list(List[List[int]]): 
    @param num_labels: Total number of labels
    """
    labels_torch = torch.zeros(len(labels_list), num_labels)
    for i in range(len(labels_list)):
        labels_torch[i,labels_list[i]] = 1
    return labels_torch