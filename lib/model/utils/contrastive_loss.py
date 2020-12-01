import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb
import time

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        
        
    def forward(self, source_out, target_out, t=1.0):
        source_out = source_out.view(source_out.size(0), -1)
        target_out = target_out.view(target_out.size(0), -1)
        
#         t0 = time.time()
        
        all_vector = torch.cat((source_out, target_out),0)
        multiply_all_vector = torch.mm(all_vector, all_vector.t())
        norm_vector = torch.norm(all_vector, p=2, keepdim=True, dim=-1)
        multiply_norm_vector = torch.mm(norm_vector, norm_vector.t())
        multiply_norm_vector = torch.clamp(multiply_norm_vector, min=1e-8)
        cos_similarity = (multiply_all_vector / multiply_norm_vector) / t
        exp_cos_similarity = torch.exp(cos_similarity)
        loss = 0
        half = exp_cos_similarity.size(0)//2
        exp_cos_similarity0 = exp_cos_similarity[:, 0:half].sum(1)
        exp_cos_similarity1 = exp_cos_similarity[:, half:].sum(1)
        # original
#         for i in range(cos_similarity.size(0)):
#             if i < exp_cos_similarity.size(0)//2:
#                 loss += -torch.log(exp_cos_similarity[i,i+half]/(exp_cos_similarity[i].sum() -
#                                                                   exp_cos_similarity[i, i]))
#             else:
#                 loss += -torch.log(exp_cos_similarity[i,i-half]/(exp_cos_similarity[i].sum() -
#                                                                   exp_cos_similarity[i, i]))
        # half negative
        for i in range(cos_similarity.size(0)):
            if i < half:
                loss += -torch.log(exp_cos_similarity[i,i+half]/(exp_cos_similarity1[i]))
            else:
                loss += -torch.log(exp_cos_similarity[i,i-half]/(exp_cos_similarity0[i]))
        
        # L2 loss
#         t1 = time.time()
#         l2_norm = torch.norm(source_out[:, None]-target_out, dim=2, p=2) / (t*source_out.size(1))
#         exp_l2_norm = torch.exp(l2_norm)
#         for i in range(exp_l2_norm.size(0)):
#             loss += torch.log(exp_l2_norm[i,i]/(exp_l2_norm[i].sum()))
#             print('loss: {}'.format(tmp))
#         t2 = time.time()
#         print('contrast time: {}, l2 time: {}'.format(t1-t0,t2-t1))
#         pdb.set_trace()
        # random
#         for i in range(cos_similarity.size(0)):
# #             rand_index = np.random.randint(0, high=exp_cos_similarity.size(1))
# #             while ((rand_index == i) or (rand_index == i+half)):
# #                    rand_index = np.random.randint(0, high=exp_cos_similarity.size(1))
#             if i < exp_cos_similarity.size(0)//2:
#                 rand_index = np.random.randint(0, high=exp_cos_similarity.size(1))
#                 while ((rand_index == i) or (rand_index == i+half)):
#                        rand_index = np.random.randint(0, high=exp_cos_similarity.size(1))
#                 loss += -torch.log(exp_cos_similarity[i,i+half]/(exp_cos_similarity[i,i+half] + exp_cos_similarity[i,rand_index]))
#             else:
#                 rand_index = np.random.randint(0, high=exp_cos_similarity.size(1))
#                 while ((rand_index == i) or (rand_index == i-half)):
#                        rand_index = np.random.randint(0, high=exp_cos_similarity.size(1))
#                 loss += -torch.log(exp_cos_similarity[i,i-half]/(exp_cos_similarity[i,i+half] + exp_cos_similarity[i,rand_index]))
        
        return loss / exp_cos_similarity.size(0)
    
# class ContrastiveLoss(nn.Module):
#     def __init__(self):
#         super(ContrastiveLoss, self).__init__()
        
#     #triplet sim loss    
#     def forward(self, source_out, target_out, t=1.0):
#         all_vector = torch.cat((source_out, target_out),0)
#         multiply_all_vector = torch.mm(all_vector, all_vector.t())
#         norm_vector = torch.norm(all_vector, p=2, keepdim=True, dim=-1)
#         multiply_norm_vector = torch.mm(norm_vector, norm_vector.t())
#         multiply_norm_vector = torch.clamp(multiply_norm_vector, min=1e-8)
#         cos_similarity = (multiply_all_vector / multiply_norm_vector) / 1.0
#         half = cos_similarity.size(0)//2
#         loss = cos_similarity[0,2] - cos_similarity[0,3] + cos_similarity[1,3] - cos_similarity[1,2] + cos_similarity[2,0] -cos_similarity[2,1] + cos_similarity[3,1] - cos_similarity[3,0]        
#         return -loss / 4.0
    
