import pickle
import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from nltk.tokenize import regexp_tokenize, sent_tokenize, word_tokenize
from torch.utils.data import Dataset
from tqdm import tqdm

from email_preprocessing import (
    check_ack_word,
    split_clean_sentences,
    tokenize_sentence,
    transform,
)

class Vocab(object):
    def __init__(
        self,
        dataset_text=None,
        dataset_with_annotation=None,
        *,
        max_total_words: int = 3000,
    ):

        self.word2id = {}
        self.id2word = {}
        self.max_total_words = max_total_words

        self.pattern = r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """  

        self.english_punctuations = [ ] 
        self.build_vocab(dataset_text, dataset_with_annotation)

        self.vocab_size = len(self.word2id)


    def build_vocab(self, dataset_text = None, dataset_with_annotation = None):
        sentences = []
        
        words = []
        
        if dataset_with_annotation is not None:
            raw_data = pd.read_csv(dataset_with_annotation, encoding = 'utf-8', sep = ',', engine = 'python')
            raw_data = np.array(raw_data)
            
            np.random.seed(0)
            np.random.shuffle(raw_data)
        
            
            for i in tqdm(range(0, raw_data.shape[0]-800)):
                for j in range(1, 7):
                    if raw_data[i][2*j] is not None:
                        results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*',re.S)
                        dd=results.sub(" website ",raw_data[i][2*j])
                        a = regexp_tokenize(transform(dd), self.pattern)
                        temp = []
                        for k in range(0, len(a)):
                            if a[k] not in self.english_punctuations and check_ack_word(a[k]) == 1:
                                if a[k].isdigit():
                                    a[k] = 'number'
                                elif a[k][0] == '$':
                                    a[k] = 'money'
                                elif a[k][-1] == '%':
                                    a[k] = 'percentage'
                                temp.append(a[k].lower())
                                words.append(a[k].lower())

                        if len(temp) > 0:
                            sentences.append(temp)

        if dataset_text is not None:
            raw_data2 = pd.read_csv(dataset_text, encoding='utf-8', sep=',', engine = 'python')
            raw_data2 = np.array(raw_data2)

            for i in tqdm(range(0, raw_data2.shape[0])):
                try:
                    temp_sentences = split_clean_sentences(
                        subject="",
                        body=raw_data2[i][1],
                        max_total_words=self.max_total_words,
                    )

                    for sentence_tokens in temp_sentences:
                        if len(sentence_tokens) > 0:
                            sentences.append(sentence_tokens)
                            words.extend(sentence_tokens)
                            
                except:
                    pass
                
        
        word_frequency = {}
        for i in range(0, len(words)):
            if words[i] in word_frequency:
                 word_frequency[words[i]] += 1
            else:
                word_frequency[words[i]]  = 1
        
        
               
        self.word2id['<PAD>'] = 0
        self.id2word[0] = '<PAD>'
        self.word2id['<UNK>'] = 1
        self.id2word[1] = '<UNK>'
        
        self.unk_count = 0
        wid = 2
        for (u,v) in word_frequency.items():
            if v >= 5:
                self.word2id[u] = wid
                self.id2word[wid] = u
                wid += 1
            else:
                self.unk_count += 1
                    

    
class dataLoaderANN(Dataset):
    def __init__(
        self,
        vocab,
        dataset_with_annotation,
        mode='train',
        idx=0,
        *,
        target_pickle_path: str | Path | None = None,
    ):
        self.vocab = vocab
        self.target_pickle_path = Path(target_pickle_path) if target_pickle_path else None

        self.max_len = 0
        self.length = []
        self.mode = mode
        self.idx = idx
        
        
        self.max_sentence_num = 0

        
        self.pattern = r"""(?x)                  
                     (?:[A-Z]\.)+          
                     |\$?\d+(?:\.\d+)?%?    
                     |\w+(?:[-']\w+)*      
                     |\.\.\.               
                     |(?:[.,;"'?():-_`])    
                  """  

        self.english_punctuations = [ ] 
        self.label_set = ['Other', 'Concreteness', 'Commitment', 'Emotional', 'Identity', 'Impact', 'Scarcity' ]

        self.message_id_and_score = {}

        self.message = []
        self.message_id = []
        self.message_sentence_label = []

        self.message_length = []
        self.sentence_length = []

        self.message_all = []
        
        self.load_data(dataset_with_annotation)
        
        if self.max_len > 400:
            self.max_len = 400
        
        
        for i in range(0, len(self.message)):
            self.message_all.append((self.message_id[i], self.message[i], self.message_sentence_label[i],  self.message_length[i], self.sentence_length[i]))
         
        self.train = self.message_all[0:int((self.ann_num - 800))]
        self.test = self.message_all[self.ann_num-800:self.ann_num-300]
        self.dev = self.message_all[self.ann_num-300: self.ann_num]
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train)
        elif self.mode == 'test':
            return len(self.test)
        elif self.mode == 'dev':
            return len(self.dev)
        elif self.mode == 'test_debug':
            return 1
        
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            temp = self.train[idx]
        elif self.mode == 'test':
            temp = self.test[idx]
        elif self.mode == 'dev':
            temp = self.dev[idx]
        elif self.mode == 'test_debug':
            temp = self.test[self.idx]

        message_id = temp[0]
        message = temp[1]
        message_sentence_label = temp[2]
        message_length = temp[3]
        sentence_length = temp[4]

        message_target = self.lookup_score(message_id)
        
        message_vec = torch.LongTensor(self.message2id(message))

        labels = np.array([8] * self.max_sentence_num)
        
        for i in range(0, len(message_sentence_label)):
            labels[i] = message_sentence_label[i]
        labels = torch.LongTensor(labels)

        lengths = np.array([0] * self.max_sentence_num)
        for i in range(0, len(sentence_length)):
            lengths[i] = sentence_length[i]
            
        lengths = torch.LongTensor(lengths)
        
        #message,  message_id
        return (message_vec, message_target, labels, message_length, lengths)
    
    
    def message2id(self, message):
        X = np.zeros([self.max_sentence_num, self.max_len])
        for i in range(0, len(message)):
            for j, si in enumerate(message[i]):
                if i < self.max_sentence_num and j < self.max_len:
                    try:
                        id = self.vocab.word2id[si.lower()]
                        X[i][j] = id
                    except:
                        X[i][j] = 1
        return X
    
    
    def lookup_label_id(self, s):
        for i in range(0, len(self.label_set)):
            if s == self.label_set[i]:
                return i
        return 8
    
    def lookup_score(self, id):
        try:
            score = self.message_id_and_score[id]
            #+ 1# + self.message_id_and_score[id][1] 
            return -np.log(score+1)
        except:
            return 0
        
        
    def load_data(self, dataset_with_annotation):
        if self.target_pickle_path and self.target_pickle_path.exists():
            with self.target_pickle_path.open('rb') as f:
                self.message_id_and_score = pickle.load(f)
        else:
            self.message_id_and_score = {}
            
        raw_data = pd.read_csv(dataset_with_annotation, encoding='utf-8', sep=',', engine = 'python')
        raw_data = np.array(raw_data)
        
        np.random.seed(0)
        np.random.shuffle(raw_data)
        
        for i in tqdm(range(0, raw_data.shape[0])):
            sentence_temp = []
            sentence_label = []
            sentence_length_temp = []

            for j in range(1, 7):
                if raw_data[i][2*j] is not None:
                    results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*',re.S)
                    dd=results.sub(" website ",raw_data[i][2*j])
                    a = regexp_tokenize(transform(dd), self.pattern)
                    temp = []
                    
                    for k in range(0, len(a)):
                        if a[k] not in self.english_punctuations and check_ack_word(a[k]) == 1:
                            if a[k].isdigit():
                                a[k] = 'number'
                            elif a[k][0] == '$':
                                a[k] = 'money'
                            elif a[k][-1] == '%':
                                a[k] = 'percentage'
                            temp.append(a[k].lower())
                    
                    
                    sentence_temp.append(temp)
                    l = len(temp)
                    sentence_length_temp.append(l)

                    if self.max_len < l:
                        self.max_len = l

                    sentence_label.append(self.lookup_label_id(raw_data[i][2*j + 1]))
            
            l = len(sentence_temp)
            self.message_length.append(l)
            if self.max_sentence_num < l:
                self.max_sentence_num = l
            
            self.message.append(sentence_temp)
            self.message_id.append(raw_data[i][0])
            self.message_sentence_label.append(sentence_label)
            self.sentence_length.append(sentence_length_temp)

        self.ann_num = len(self.message)
        
        
class dataLoaderUnann(Dataset):
    def __init__(
        self,
        vocab,
        dataset_text,
        dataset_with_annotation=None,
        *,
        target_pickle_path: str | Path | None = None,
        max_total_words: int = 3000,
    ):
        self.vocab = vocab
        self.target_pickle_path = Path(target_pickle_path) if target_pickle_path else None
        self.max_total_words = max_total_words

        self.max_len = 1
        
        self.max_sentence_num = 1

        self.english_punctuations = [ ] 
            
        self.pattern = r"""(?x)                  
                     (?:[A-Z]\.)+          
                     |\$?\d+(?:\.\d+)?%?    
                     |\w+(?:[-']\w+)*      
                     |\.\.\.               
                     |(?:[.,;"'?():-_`])    
                  """  

        self.message_id_and_score = {}
 
        self.message = []
        self.message_id = []

        self.message_length = []
        self.sentence_length = []
        
        self.message_all = []

        self.load_data(dataset_text, dataset_with_annotation)

        for i in range(0, len(self.message)):
            self.message_all.append((self.message_id[i], self.message[i], self.message_length[i], self.sentence_length[i]))
            
            
    def __len__(self):
        return len(self.message)
    
    def __getitem__(self, idx):
        temp = self.message_all[idx]

        message_id = temp[0]
        message = temp[1]
        message_length = temp[2]
        sentence_length = temp[3]

        message_target = self.lookup_score(message_id)
    
        message_vec = self.message2id(message)

        labels = np.array([8]* self.max_sentence_num)

        lengths = np.array([0] * self.max_sentence_num)
        
        for i in range(0, min(self.max_sentence_num,len(sentence_length))):
            lengths[i] = sentence_length[i]
            
        lengths = torch.LongTensor(lengths)
        
        #message,  message_id
        return (message_vec, message_target, labels, message_length, lengths)
    
    
    
    def message2id(self, message):
        X = np.zeros([self.max_sentence_num, self.max_len])
        for i in range(0, len(message)):
            for j, si in enumerate(message[i]):
                if i < self.max_sentence_num and j < self.max_len:
                    try:
                        id = self.vocab.word2id[si.lower()]
                        X[i][j] = id
                    except:
                        X[i][j] = 1
        
        return X
    
    def lookup_score(self, id):
        try:
            score = self.message_id_and_score[id]
            #[0] # + self.message_id_and_score[id][1] 
            return -np.log(score+1)
          
        except:
            return 0
        
    def load_data(self, dataset_text, dataset_with_annotation = None):
        if self.target_pickle_path and self.target_pickle_path.exists():
            with self.target_pickle_path.open('rb') as f:
                self.message_id_and_score = pickle.load(f)
        else:
            self.message_id_and_score = {}
            
        
        raw_data = pd.read_csv(dataset_text, encoding = 'utf-8', sep = ',', engine = 'python')
        raw_data = np.array(raw_data)
        
        
        for i in tqdm(range(0, int(raw_data.shape[0]))):
            try:
                sentence_temp = split_clean_sentences(
                    subject="",
                    body=raw_data[i][1],
                    max_total_words=self.max_total_words,
                )
                sentence_length_temp = [len(sentence) for sentence in sentence_temp]
                
                if len(sentence_temp) > 0:
                    self.message_id.append(raw_data[i][0])
                    l = len(sentence_temp)
                    self.max_sentence_num = max(self.max_sentence_num, l)
                    for sentence in sentence_temp:
                        self.max_len = max(self.max_len, len(sentence))

                    self.message_length.append(l)
                    self.sentence_length.append(sentence_length_temp)
                    self.message.append(sentence_temp)
                    
            except:
                pass
            
     
    
if __name__ == '__main__':
    dataset_with_annotation = "./dataSetNew/annotation_dataset.0720.public.csv"
    dataset_text = "./dataSetNew/persuasion_dataset_text.0720.public.csv"

    vocab = Vocab(dataset_with_annotation = dataset_with_annotation, dataset_text = dataset_text)
    #with open('./data/vocab_new.pkl', 'wb') as f:
    #   pickle.dump(vocab, f)
        
    #with open('./data/vocab_new.pkl', 'rb') as f:
    #    vocab = pickle.load(f)
    print(vocab.vocab_size)
    
   
    dataSet = dataLoaderANN(vocab, dataset_with_annotation)
    dataSet2 = dataLoaderANN(vocab, dataset_with_annotation, mode = 'test')
    dataSet3 = dataLoaderANN(vocab, dataset_with_annotation, mode = 'dev')
    dataSet4 = dataLoaderUnann(vocab, dataset_text)

    loaderTrain = Data.DataLoader(dataset = dataSet, batch_size = 512, shuffle = False, num_workers = 4)
    loaderTest = Data.DataLoader(dataset = dataSet2, batch_size = 500, shuffle = False, num_workers = 4)
    loaderDev = Data.DataLoader(dataset = dataSet3, batch_size = 500, shuffle = False, num_workers = 4)    
    loaderTrain2 = Data.DataLoader(dataset = dataSet4, batch_size = 512, shuffle = False, num_workers = 4)

    
    
    for step, (x, y, l, num, length) in enumerate(loaderTrain):
        print(x.shape, y.shape, l.shape, num.shape, length.shape)
        
    for step, (x, y, l, num, length) in enumerate(loaderTrain2):
        print(x.shape, y.shape, l.shape, num.shape, length.shape)
        
    for step, (x, y, l, num, length) in enumerate(loaderTest):
        print(x.shape, y.shape, l.shape, num.shape, length.shape)
       
    for step, (x, y, l, num, length) in enumerate(loaderDev):
        print(x.shape, y.shape, l.shape, num.shape, length.shape)
  
                        
                    
    


        
        
        
        
    
            

        
        
    

    
    
    
    
    
    
    
    
    
    
                    
