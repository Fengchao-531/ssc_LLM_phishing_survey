import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MessageLoss(nn.Module):
    def __init__(self, w1 = 1, w2 = 10):
        super(MessageLoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.mse = nn.MSELoss()
        
    def forward(self, labeled_doc = None, target1 = None, labeled_sent = None, target2 = None, w1 = None, unlabeled_doc = None, target3 = None, mode = 'None'):
        if w1 is not None:
            self.w1 = w1
            
        if labeled_doc is not None:
            labeled_doc = labeled_doc.squeeze(1)
            labeled_doc_loss = self.mse(labeled_doc, target1)
        else:
            labeled_doc_loss = 0
            
        if unlabeled_doc is not None:
            unlabeled_doc = unlabeled_doc.squeeze(1)
            unlabeled_doc_loss = self.mse(unlabeled_doc, target3)
        else:
            unlabeled_doc_loss = 0
            
        labeled_sent_loss = 0
        count = 0
        correct = 0
        predict_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        correct_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
        correct_total = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}

        p = 0
        r = 0
        
        if target2 is not None:
            target2 = target2.view(target2.shape[0] * target2.shape[1])
            labeled_sent1 = F.log_softmax(labeled_sent, dim = 1)
            labeled_sent2 = torch.argmax(F.softmax(labeled_sent, dim = 1), dim = 1)

            

            for i in range(0, target2.shape[0]):
                if target2[i] == 8:
                    continue
                else:
                    count += 1
                    correct_total[target2[i].item()] += 1
                    predict_dict[labeled_sent2[i].item()] += 1
                    if labeled_sent2[i] == target2[i]:
                        correct += 1
                        correct_dict[target2[i].item()] += 1
                    labeled_sent_loss += (-1 * labeled_sent1[i][target2[i].item()])

            if count != 0:
                labeled_sent_loss = labeled_sent_loss / count
      

            
                
        if mode != 'train':
            for (u, v) in correct_dict.items():
                if predict_dict[u] == 0:
                    temp = 0
                else:
                    temp = v/predict_dict[u]
                temp2 = v/correct_total[u]
                p += temp
                r += temp2
                
                
        if mode == 'train':
            loss = self.w1 * labeled_doc_loss + (1-self.w1) * unlabeled_doc_loss + self.w2 * labeled_sent_loss
            return loss, labeled_sent_loss, correct_dict, predict_dict, correct_total, correct, count, p/7, r/7
        else:
            loss = labeled_doc_loss + self.w2 * labeled_sent_loss
            return loss, np.sqrt(labeled_doc_loss.cpu().detach().numpy()), labeled_sent_loss, correct_dict, predict_dict, correct_total, correct, count, p/7, r/7
                
                
                
                
                