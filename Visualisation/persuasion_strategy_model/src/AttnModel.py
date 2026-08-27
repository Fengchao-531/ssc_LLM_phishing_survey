import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Request(nn.Module):
    def __init__(self, config, vocab_size = None, pretrained_embedding = None):
        super(Request, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size

        self.feature_size = config.feature_size

        if vocab_size is not None and pretrained_embedding is not None:
            self.embed = nn.Embedding(vocab_size, self.embedding_size)
            pretrained_weight = np.array(pretrained_embedding)
            self.embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        else:
            self.embed = nn.Embedding(vocab_size, self.embedding_size)
            

        # word level
        self.word_GRU = nn.GRU(self.embedding_size, self.hidden_size, batch_first = True)
        self.w_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_context_vector = nn.Parameter(torch.randn([self.hidden_size, 1]).float())
        self.softmax = nn.Softmax(dim = 1)

        self.word_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_linear_out = nn.Linear(self.hidden_size, self.feature_size)


        # sent level
        self.output_size = config.output_size

        self.sent_GRU = nn.GRU(self.feature_size, self.hidden_size, batch_first = True, bias = False)
        self.s_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.s_context_vector = nn.Parameter(torch.randn([self.hidden_size, 1]).float())

        self.sent_linear = nn.Linear(self.hidden_size, self.output_size, bias = False)
        
        
    def forward(self, x, sentence_num_, sentence_length):
        
        # batch_size * sentence_num * sentence_len
        batch_size = x.shape[0]
        sentence_num = x.shape[1]

        x = x.view([x.shape[0] * x.shape[1], x.shape[2]])

        x_embed = self.embed(x)
        # batch_size*sentence_num * sentence_len * embedding_size

        x_out, _ = self.word_GRU(x_embed)
        # batch_size*sentence_num * sentence_len * hidden_size

        Hw = torch.tanh(self.w_proj(x_out))
        # batch_size*sentence_num * sentence_len * hidden_size
        w_score = self.softmax(Hw.matmul(self.w_context_vector))
        x_out = x_out.mul(w_score)
        x_out = torch.sum(x_out, dim = 1)

        x_out = F.relu(self.word_linear(x_out))
        x_out = self.word_linear_out(x_out)

        x_out_softmax = F.softmax(x_out, dim = 1)
        # batch_size*sentence_num * feature_size

        x_out_softmax = x_out_softmax.view([batch_size, sentence_num, x_out_softmax.shape[1]])
        
        x_out_message, _ = self.sent_GRU(x_out_softmax)
        
        Hs = torch.tanh(self.s_proj(x_out_message))
        s_score = self.softmax(Hs.matmul(self.s_context_vector))
        x_out_message = x_out_message.mul(s_score)
        x_out_message = torch.sum(x_out_message, dim = 1)
        
        x_out_message = self.sent_linear(x_out_message)

        
        return x_out, x_out_message
        
        
        
        
        
        
