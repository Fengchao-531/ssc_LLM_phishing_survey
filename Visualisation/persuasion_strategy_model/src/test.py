import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset

from Config import Config
from DataLoader import Vocab, dataLoaderANN, dataLoaderUnann
from AttnModel import Request
from MessageLoss import MessageLoss

config = Config()

dataset_text = "./dataSetNew/persuasion_dataset_text.0720.public.csv"
dataset_with_annotation = "./dataSetNew/annotation_dataset.0720.public.csv"

vocab = Vocab(dataset_with_annotation = dataset_with_annotation, dataset_text = dataset_text)

print(vocab.vocab_size)


dataSet2 = dataLoaderANN(vocab, dataset_with_annotation, mode = 'test')
dataSet3 = dataLoaderANN(vocab, dataset_with_annotation, mode = 'dev')

loaderTest = Data.DataLoader(dataset = dataSet2, batch_size = 500, shuffle = False, num_workers = 1)
loaderDev = Data.DataLoader(dataset = dataSet3, batch_size = 500, shuffle = False, num_workers = 1)

request = torch.load('./model/model_attn_1.pkl')
request.cuda()

messageLoss = MessageLoss(w2 = 10)
messageLoss.cuda()

def testModel():
    global request
    request.eval()
    for step, (x, y, l, num, length) in enumerate(loaderTest):
        message_input = Variable(x.type(torch.LongTensor)).cuda()
        message_target = Variable(y.type(torch.FloatTensor)).cuda()
        sentence_label = Variable(l.type(torch.LongTensor)).cuda()
        sentence_out, message_out = request(message_input, num, length)

        loss, rmse, sent_loss, correct_dict, predict_dict, correct_total, correct, count, p, r = messageLoss(labeled_doc = message_out, target1 = message_target, labeled_sent = sentence_out, target2 = sentence_label, mode = 'test')

    if p + r != 0:
        f1 = (2*p*r)/(p+r)
    else:
        f1 = 0

    print("...")
    print("Test: total_loss: {0}, score_rmse_loss: {1}, cross_loss: {2}".format(loss, rmse, sent_loss))
    print("   : acc: {0}, P: {1}, R: {2}, F1: {3}".format(correct/count, p, r, f1))
    print("...")

def evaluation():
    global request
    request.eval()
    for step, (x, y, l, num, length) in enumerate(loaderDev):
        message_input = Variable(x.type(torch.LongTensor)).cuda()
        message_target = Variable(y.type(torch.FloatTensor)).cuda()
        sentence_label = Variable(l.type(torch.LongTensor)).cuda()

        sentence_out, message_out = request(message_input, num, length)

        loss, rmse, sent_loss, correct_dict, predict_dict, correct_total, correct, count, p, r = messageLoss(labeled_doc = message_out, target1 = message_target, labeled_sent = sentence_out, target2 = sentence_label, mode = 'dev')

    if p + r != 0:
        f1 = (2*p*r)/(p+r)
    else:
        f1 = 0

    print("...")
    print("Dev: total_loss: {0}, score_rmse_loss: {1}, cross_loss: {2}".format(loss, rmse, sent_loss))
    print("   : acc: {0}, P: {1}, R: {2}, F1: {3}".format(correct/count, p, r, f1))
    print("...")




evaluation()
testModel()