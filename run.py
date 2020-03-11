"""
CS224N 2019-20: Homework 4
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>

Usage:
    run.py train --notes_file=<file> --labels_file=<file> --dev-notes=<file> --dev-labels=<file> --vocab=<file> [options]
    run.py test --vocab=<file> [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py test --vocab=<file> [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --notes_file=<file>                     train notes file
    --labels_file=<file>                    train labels file
    --dev-notes=<file>                      dev notes file
    --dev-labels=<file>                     dev labels file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --sent_max_length=<int>                 Max length of training sentences [default: 0]
    --remove_stopwords                      Remove stopwords from nltk
"""
import math
import sys
import pickle
import time


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
# from nmt_model import Hypothesis, NMT
from bilstm_model import BiLSTM
from node_model import Node
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter, ind_to_one_hot
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils

from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from statistics import harmonic_mean 


def eval_on_val(model, dev_data, loss_func, num_labels = 19, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_labels = 0.
    
    epoch_TP = 0.
    epoch_FP = 0.
    epoch_FN = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for notes_docs, notes_labels in batch_iter(dev_data, batch_size=batch_size):

            example_scores = model(notes_docs, notes_labels) # (batch_size,)
            labels_torch = ind_to_one_hot(notes_labels, num_labels)
            assert(labels_torch.shape == example_scores.shape)
            batch_loss = loss_func(example_scores, labels_torch)
            loss = batch_loss / len(dev_data[0])
            
            predictions = torch.zeros(example_scores.shape)
            predictions[example_scores >= .5] = 1
#             if train_iter % 25 == 0:
#                 print("example_scores[0]: \n", example_scores[0])
#                 print("predictions[0]:  \n", predictions[0])
#                 print("labels_torch[0]: \n", labels_torch[0])
            
            TP, FP, FN = update_f1_metrics(predictions, labels_torch)
            epoch_TP += TP
            epoch_FP += FP
            epoch_FN += FN
            
            cum_loss += loss.item()

            tgt_labels_num_to_predict = sum(len(s[1:]) for s in notes_labels) 
            cum_tgt_labels += tgt_labels_num_to_predict
            
        if (epoch_TP + epoch_FP)>0 and (epoch_TP + epoch_FN) > 0:
            m_f1 = compute_mic_f1(epoch_TP, epoch_FP, epoch_FN)
        else:
            m_f1 = 0
#         micro_F1_scores.append(m_f1)
#         print("Hermonic mean: ", m_f1)

        ppl = np.exp(cum_loss / cum_tgt_labels)

    if was_training:
        model.train()

    return ppl, m_f1


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def update_f1_metrics(predictions, labels):
    TP = abs(int((predictions*labels).sum().item()))
    FP = abs(int((predictions*(labels-1)).sum().item()))
    FN = abs(int(((predictions-1)*labels).sum().item()))
    return TP, FP, FN

def compute_mic_f1(epoch_TP, epoch_FP, epoch_FN):
    prec_mic = epoch_TP / (epoch_TP + epoch_FP)
    rec_mic = epoch_TP / (epoch_TP + epoch_FN)
    m_f1 = harmonic_mean([prec_mic,rec_mic])
    return m_f1




def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    train_data_notes  = read_corpus(args['--notes_file'], column='text',
                                   sent_max_length=int(args['--sent_max_length']), 
                                   remove_stopwords=bool(args['--remove_stopwords']))
    train_data_labels = read_corpus(args['--labels_file'], column='label',
                                   sent_max_length=int(args['--sent_max_length']), 
                                   remove_stopwords=bool(args['--remove_stopwords']))
    dev_notes  = read_corpus(args['--dev-notes'], column='text',
                                   sent_max_length=int(args['--sent_max_length']), 
                                   remove_stopwords=bool(args['--remove_stopwords']))
    dev_labels = read_corpus(args['--dev-labels'], column='label',
                                   sent_max_length=int(args['--sent_max_length']), 
                                   remove_stopwords=bool(args['--remove_stopwords']))

    train_data = list(zip(train_data_notes, train_data_labels))
    dev_data = list(zip(dev_notes, dev_labels))

    train_batch_size = int(16)
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = Vocab.load(args['--vocab'])
    
#     model = BiLSTM(embed_size=int(args['--embed-size']),
#                 hidden_size=int(args['--hidden-size']),
#                 dropout_rate=float(args['--dropout']),
#                 vocab=vocab)
    model = Node(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)
    model.train()
    
    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    training_micro_F1_scores = []
    validation_micro_F1_scores = []
    train_time = begin_time = time.time()
    
    loss_func = torch.nn.BCELoss(reduction='sum')
    print('begin Maximum Likelihood training')
    
    while True:
        epoch += 1
#         print("Epoch: {}".format(epoch))
#         if epoch == 20:
#             exit(0)

        epoch_TP = 0.
        epoch_FP = 0.
        epoch_FN = 0.
        for notes_docs, notes_labels in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
#             print("train_iter: {}".format(train_iter))

            optimizer.zero_grad()

            batch_size = len(notes_docs)
#             print(model.model_embeddings.note_embeds.weight.device)
            example_scores = model(notes_docs, notes_labels) # (batch_size,num_labels)
            labels_torch = ind_to_one_hot(notes_labels, vocab.num_labels)
            predictions = torch.zeros(example_scores.shape)
            predictions[example_scores >= .5] = 1
            if train_iter % 2000 == 0:
                print("example_scores[0]: \n", example_scores[0])
                print("predictions[0]:  \n", predictions[0])
                print("labels_torch[0]: \n", labels_torch[0])
            
            TP, FP, FN = update_f1_metrics(predictions, labels_torch)
            epoch_TP += TP
            epoch_FP += FP
            epoch_FN += FN

            assert(labels_torch.shape == example_scores.shape)
            batch_loss = loss_func(example_scores, labels_torch)
            loss = batch_loss / batch_size
#             print("loss: ", loss)
#             print()
            

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()
            
            
            

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

#             tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
#             report_tgt_words += tgt_words_num_to_predict
#             cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' \
                      % (epoch, train_iter, report_loss / report_examples, \
                         #math.exp(report_loss / report_tgt_words),\
                         cum_examples, \
                         report_tgt_words / (time.time() - train_time),\
                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

#             exit(0)
            
            # perform validation
            if train_iter % valid_niter == 0:
#                 print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
#                                                                                          cum_loss / cum_examples,
#                                                                                          np.exp(cum_loss / cum_tgt_words),
#                                                                                          cum_examples), file=sys.stderr)
                print('epoch %d, iter %d, cum. loss %.2f, cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         
                                                                                         cum_examples), file=sys.stderr)



                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
#                 dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger

                dev_ppl, m_f1 = eval_on_val(model, dev_data, loss_func, num_labels=vocab.num_labels, batch_size=128)   # dev batch size can be a bit larger
                validation_micro_F1_scores.append(m_f1)
                valid_metric = dev_ppl

                print('validation: iter %d, dev. ppl      %f' % (train_iter, dev_ppl), file=sys.stderr)
                print('validation: iter %d, dev. micro-F1 %f' % (train_iter, m_f1), file=sys.stderr)

#                 is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                is_better = len(hist_valid_scores) == 0 or valid_metric < min(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('saving the current best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
#                 elif patience < int(args['--patience']):
#                     patience += 1
#                     print('hit patience %d' % patience, file=sys.stderr)

#                     if patience == int(args['--patience']):
#                         num_trial += 1
#                         print('hit #%d trial' % num_trial, file=sys.stderr)
#                         if num_trial == int(args['--max-num-trial']):
#                             print('early stop!', file=sys.stderr)
#                             exit(0)

#                         # decay lr, and restore from previously best checkpoint
#                         lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
#                         print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

#                         # load model
#                         params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
#                         model.load_state_dict(params['state_dict'])
#                         model = model.to(device)

#                         print('restore parameters of the optimizers', file=sys.stderr)
#                         optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

#                         # set new lr
#                         for param_group in optimizer.param_groups:
#                             param_group['lr'] = lr

#                         # reset patience
#                         patience = 0

            if epoch == int(args['--max-epoch']):
                print('reached maximum number of epochs!', file=sys.stderr)
                with open('results/training_micro_f1.txt', 'w') as filehandle:
                    for listitem in training_micro_F1_scores:
                        filehandle.write('%s\n' % listitem)
                with open('results/validation_micro_f1.txt', 'w') as filehandle:
                    for listitem in validation_micro_F1_scores:
                        filehandle.write('%s\n' % listitem)
                exit(0)
        if (epoch_TP + epoch_FP)>0 and (epoch_TP + epoch_FN) > 0:
            m_f1 = compute_mic_f1(epoch_TP, epoch_FP, epoch_FN)
        else:
            m_f1 = 0
        training_micro_F1_scores.append(m_f1)
        print("Training micro-F1: ", m_f1)



# def train(args: Dict):
#     """ Train the NMT Model.
#     @param args (Dict): args from cmd line
#     """
#     train_data_notes  = read_corpus(args['--notes_file'], column='text',
#                                    sent_max_length=int(args['--sent_max_length']), 
#                                    remove_stopwords=bool(args['--remove_stopwords']))
#     train_data_labels = read_corpus(args['--labels_file'], column='label',
#                                    sent_max_length=int(args['--sent_max_length']), 
#                                    remove_stopwords=bool(args['--remove_stopwords']))
#     dev_notes  = read_corpus(args['--dev-notes'], column='text',
#                                    sent_max_length=int(args['--sent_max_length']), 
#                                    remove_stopwords=bool(args['--remove_stopwords']))
#     dev_labels = read_corpus(args['--dev-labels'], column='label',
#                                    sent_max_length=int(args['--sent_max_length']), 
#                                    remove_stopwords=bool(args['--remove_stopwords']))

#     train_data = list(zip(train_data_notes, train_data_labels))
#     dev_data = list(zip(dev_notes, dev_labels))

#     train_batch_size = int(16)
#     clip_grad = float(args['--clip-grad'])
#     valid_niter = int(args['--valid-niter'])
#     log_every = int(args['--log-every'])
#     model_save_path = args['--save-to']

#     vocab = Vocab.load(args['--vocab'])
    
#     model = BiLSTM(embed_size=int(args['--embed-size']),
#                 hidden_size=int(args['--hidden-size']),
#                 dropout_rate=float(args['--dropout']),
#                 vocab=vocab)
#     model.train()
    
#     uniform_init = float(args['--uniform-init'])
#     if np.abs(uniform_init) > 0.:
#         print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
#         for p in model.parameters():
#             p.data.uniform_(-uniform_init, uniform_init)

#     device = torch.device("cuda:0" if args['--cuda'] else "cpu")
#     print('use device: %s' % device, file=sys.stderr)

#     model = model.to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

#     num_trial = 0
#     train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
#     cum_examples = report_examples = epoch = valid_num = 0
#     hist_valid_scores = []
#     training_micro_F1_scores = []
#     validation_micro_F1_scores = []
#     train_time = begin_time = time.time()
    
#     loss_func = torch.nn.BCELoss(reduction='sum')
#     print('begin Maximum Likelihood training')
    
#     while True:
#         epoch += 1
# #         print("Epoch: {}".format(epoch))
# #         if epoch == 20:
# #             exit(0)

#         epoch_TP = 0.
#         epoch_FP = 0.
#         epoch_FN = 0.
#         for notes_docs, notes_labels in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
#             train_iter += 1
# #             print("train_iter: {}".format(train_iter))

#             optimizer.zero_grad()

#             batch_size = len(notes_docs)
# #             print(model.model_embeddings.note_embeds.weight.device)
#             example_scores = model(notes_docs, notes_labels) # (batch_size,num_labels)
#             labels_torch = ind_to_one_hot(notes_labels, vocab.num_labels)
#             predictions = torch.zeros(example_scores.shape)
#             predictions[example_scores >= .5] = 1
#             if train_iter % 25 == 0:
#                 print("example_scores[0]: \n", example_scores[0])
#                 print("predictions[0]:  \n", predictions[0])
#                 print("labels_torch[0]: \n", labels_torch[0])
            
#             TP, FP, FN = update_f1_metrics(predictions, labels_torch)
#             epoch_TP += TP
#             epoch_FP += FP
#             epoch_FN += FN

#             assert(labels_torch.shape == example_scores.shape)
#             batch_loss = loss_func(example_scores, labels_torch)
#             loss = batch_loss / batch_size
# #             print("loss: ", loss)
# #             print()
            

#             loss.backward()

#             # clip gradient
#             grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

#             optimizer.step()
            
            
            

#             batch_losses_val = batch_loss.item()
#             report_loss += batch_losses_val
#             cum_loss += batch_losses_val

# #             tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
# #             report_tgt_words += tgt_words_num_to_predict
# #             cum_tgt_words += tgt_words_num_to_predict
#             report_examples += batch_size
#             cum_examples += batch_size

#             if train_iter % log_every == 0:
#                 print('epoch %d, iter %d, avg. loss %.2f' \
#                       'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' \
#                       % (epoch, train_iter, report_loss / report_examples, \
#                          #math.exp(report_loss / report_tgt_words),\
#                          cum_examples, \
#                          report_tgt_words / (time.time() - train_time),\
#                          time.time() - begin_time), file=sys.stderr)

#                 train_time = time.time()
#                 report_loss = report_tgt_words = report_examples = 0.

# #             exit(0)
            
#             # perform validation
#             if train_iter % valid_niter == 0:
# #                 print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
# #                                                                                          cum_loss / cum_examples,
# #                                                                                          np.exp(cum_loss / cum_tgt_words),
# #                                                                                          cum_examples), file=sys.stderr)
#                 print('epoch %d, iter %d, cum. loss %.2f, cum. examples %d' % (epoch, train_iter,
#                                                                                          cum_loss / cum_examples,
                                                                                         
#                                                                                          cum_examples), file=sys.stderr)



#                 cum_loss = cum_examples = cum_tgt_words = 0.
#                 valid_num += 1

#                 print('begin validation ...', file=sys.stderr)

#                 # compute dev. ppl and bleu
# #                 dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger

#                 dev_ppl, m_f1 = eval_on_val(model, dev_data, loss_func, num_labels=vocab.num_labels, batch_size=128)   # dev batch size can be a bit larger
#                 validation_micro_F1_scores.append(m_f1)
#                 valid_metric = dev_ppl

#                 print('validation: iter %d, dev. ppl      %f' % (train_iter, dev_ppl), file=sys.stderr)
#                 print('validation: iter %d, dev. micro-F1 %f' % (train_iter, m_f1), file=sys.stderr)

# #                 is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
#                 is_better = len(hist_valid_scores) == 0 or valid_metric < min(hist_valid_scores)
#                 hist_valid_scores.append(valid_metric)

#                 if is_better:
#                     patience = 0
#                     print('saving the current best model to [%s]' % model_save_path, file=sys.stderr)
#                     model.save(model_save_path)

#                     # also save the optimizers' state
#                     torch.save(optimizer.state_dict(), model_save_path + '.optim')
# #                 elif patience < int(args['--patience']):
# #                     patience += 1
# #                     print('hit patience %d' % patience, file=sys.stderr)

# #                     if patience == int(args['--patience']):
# #                         num_trial += 1
# #                         print('hit #%d trial' % num_trial, file=sys.stderr)
# #                         if num_trial == int(args['--max-num-trial']):
# #                             print('early stop!', file=sys.stderr)
# #                             exit(0)

# #                         # decay lr, and restore from previously best checkpoint
# #                         lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
# #                         print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

# #                         # load model
# #                         params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
# #                         model.load_state_dict(params['state_dict'])
# #                         model = model.to(device)

# #                         print('restore parameters of the optimizers', file=sys.stderr)
# #                         optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

# #                         # set new lr
# #                         for param_group in optimizer.param_groups:
# #                             param_group['lr'] = lr

# #                         # reset patience
# #                         patience = 0

#             if epoch == int(args['--max-epoch']):
#                 print('reached maximum number of epochs!', file=sys.stderr)
#                 with open('results/training_micro_f1.txt', 'w') as filehandle:
#                     for listitem in training_micro_F1_scores:
#                         filehandle.write('%s\n' % listitem)
#                 with open('results/validation_micro_f1.txt', 'w') as filehandle:
#                     for listitem in validation_micro_F1_scores:
#                         filehandle.write('%s\n' % listitem)
#                 exit(0)
#         if (epoch_TP + epoch_FP)>0 and (epoch_TP + epoch_FN) > 0:
#             m_f1 = compute_mic_f1(epoch_TP, epoch_FP, epoch_FN)
#         else:
#             m_f1 = 0
#         training_micro_F1_scores.append(m_f1)
#         print("Training micro-F1: ", m_f1)
        
        
        
def test(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test notes from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_notes  = read_corpus(args['TEST_SOURCE_FILE'], column='text',
                                   sent_max_length=int(args['--sent_max_length']), 
                                   remove_stopwords=bool(args['--remove_stopwords']))
    if args['TEST_TARGET_FILE']:
        print("load test labels from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_labels = read_corpus(args['TEST_TARGET_FILE'], column='label',
                                       sent_max_length=int(args['--sent_max_length']), 
                                       remove_stopwords=bool(args['--remove_stopwords']))
    
    test_data = list(zip(test_data_notes, test_data_labels))
    num_examples = len(test_data_labels)
    test_batch_size = int(16)
    num_batches = int(np.ceil(num_examples/test_batch_size))

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = BiLSTM.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))
    
    was_training = model.training
    model.eval()
    
    vocab = Vocab.load(args['--vocab'])    
    
    
    all_predictions = []
    epoch_TP = epoch_FP = epoch_FN = 0.
    test_iter = 0
    
    with torch.no_grad():
        for notes_docs, notes_labels in batch_iter(test_data, batch_size=test_batch_size, shuffle=False):
            test_iter += 1
            print("test_iter: {} / {}".format(test_iter,num_batches))

            example_scores = model(notes_docs, notes_labels) # (batch_size,num_labels)
            labels_torch = ind_to_one_hot(notes_labels, vocab.num_labels)
            predictions = torch.zeros(example_scores.shape)
            predictions[example_scores >= .5] = 1
            if test_iter % 25 == 0:
                print("example_scores[0]: \n", example_scores[0])
                print("predictions[0]:  \n", predictions[0])
                print("labels_torch[0]: \n", labels_torch[0])

            TP, FP, FN = update_f1_metrics(predictions, labels_torch)
            epoch_TP += TP
            epoch_FP += FP
            epoch_FN += FN
    
            for i in range(predictions.shape[0]):
                all_predictions.append(predictions[i].numpy())


    if (epoch_TP + epoch_FP)>0 and (epoch_TP + epoch_FN) > 0:
        m_f1 = compute_mic_f1(epoch_TP, epoch_FP, epoch_FN)
    else:
        m_f1 = 0
    print("Test micro-F1: ", m_f1)
    
    ## Save the predictions to file to analyze later. 
    
    all_p_np = np.stack(all_predictions)
    print("saving the predictions to [{}]".format(args['OUTPUT_FILE']))
    
    with open(args['OUTPUT_FILE'], 'w') as filehandle:
        for i in range(all_p_np.shape[0]):
            filehandle.write('%s\n' % all_p_np[i,:])
            
    if was_training: model.train(was_training)
    
    
def main():
    args = docopt(__doc__)
    
    # Check pytorch version
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    
    if args['train']:
        train(args)
    elif args['test']:
        test(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()