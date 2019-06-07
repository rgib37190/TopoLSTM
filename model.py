import torch
import torch.nn as nn
import numpy as np
import os

class TopoLSTMCell(nn.Module):
    
    def __init__(self,hidden_size):
        super(TopoLSTMCell,self).__init__()
        self.hidden_size = hidden_size
        self.W = np.concatenate([self.ortho_weight(hidden_size),
                                 self.ortho_weight(hidden_size),
                                 self.ortho_weight(hidden_size),
                                 self.ortho_weight(hidden_size)],axis=1)
        self.W = nn.Parameter(torch.tensor(self.W,dtype=torch.float))
        self.U = np.concatenate([self.ortho_weight(hidden_size),
                                 self.ortho_weight(hidden_size),
                                 self.ortho_weight(hidden_size),
                                 self.ortho_weight(hidden_size)],axis=1)
        self.U = nn.Parameter(torch.tensor(self.U,dtype=torch.float))
        self.b = nn.Parameter(torch.zeros(4*hidden_size,dtype=torch.float))

    def _slice(self,x,n,dim):
        return x[:,n * dim : (n+1) * dim]
    
    def forward(self, one_step_seq_mask, one_step_topo_mask,one_step_node_embedding,h_init,c_init):
        '''
        one_step_seq_mask = (1, n_samples)
        one_step_topo_mask = (n_samples, diffusion_path_lengths)
        node_embedding = (n_samples, dim_proj)
        h_init = (diffusion_path_lengths, n_samples, dim_proj)
        c_init = (diffusion_path_lengths, n_samples, dim_proj)
        '''
        h, c = h_init, c_init
        h_sum = (one_step_topo_mask[:,:,None] * h.permute(1,0,2)).sum(dim=1)
        c_sum = (one_step_topo_mask[:,:,None] * c.permute(1,0,2)).sum(dim=1)
        
        # W * x + Up * hp + Uq * hq  + b
        input_ = torch.mm(h_sum,self.U) + torch.mm(one_step_node_embedding,self.W) + self.b
        
        # lstm gate
        i = torch.sigmoid(self._slice(input_, 0, self.hidden_size))
        f = torch.sigmoid(self._slice(input_, 1, self.hidden_size))
        o = torch.sigmoid(self._slice(input_, 2, self.hidden_size))
        c = torch.tanh(self._slice(input_, 3, self.hidden_size))
        
        c = f * c_sum + i * c
        c = one_step_seq_mask[:,None] * c
        
        h = o * (torch.tanh(c))
        h = one_step_seq_mask[:,None] * h
        
        return h, c
    # orthogonalize LSTM initial weight because it can prevent gradient vanish or explosion
    def ortho_weight(self,hidden_size):
        W = np.random.randn(hidden_size,hidden_size)
        u, s, v = np.linalg.svd(W)
        return u

class TopoLSTM(nn.Module):
    def __init__(self,data_dir,node_index,hidden_size):
        super(TopoLSTM,self).__init__()
        self.topolstm_cell = TopoLSTMCell(hidden_size).cuda()
        self.hidden_size = hidden_size
        self.node_index = node_index
        self.node_embedding = nn.Parameter(torch.tensor(self.get_embedding(data_dir),dtype=torch.float))
        self.embedding = nn.Embedding(len(node_index),hidden_size)
        self.embedding.weight = self.node_embedding
        self.embedding.requires_grad = False
        self.receiver_embedding = nn.Linear(hidden_size,len(node_index),bias=True)
    def forward(self,sequence_matrix,sequence_masks_matrix,topo_masks):
        '''
        sequence_matrix = (n_timesteps, n_samples)
        sequence_masks_matrix = (n_timesteps, n_samples)
        topo_masks = (n_timesteps, n_samples, n_timesteps)
        '''
        embedding_sequence = self.embedding(sequence_matrix)
        n_timesteps = embedding_sequence.shape[0]
        n_samples = embedding_sequence.shape[1]
        self.h_init = torch.zeros(n_timesteps,n_samples,self.hidden_size).cuda()
        self.c_init = torch.zeros(n_timesteps,n_samples,self.hidden_size).cuda()
        # iterate every timesteps
        for timesteps, one_step_seq_mask, one_step_topo_mask, one_step_node_embedding in zip(range(n_timesteps), sequence_masks_matrix, topo_masks, embedding_sequence):
            one_step_seq_mask = one_step_seq_mask.cuda()
            one_step_topo_mask = one_step_topo_mask.cuda()
            one_step_node_embedding = one_step_node_embedding.cuda()
            h, c = self.topolstm_cell(one_step_seq_mask, one_step_topo_mask, one_step_node_embedding, self.h_init, self.c_init)
            self.h_init[timesteps] = h
            self.c_init[timesteps] = c
        # mean pooling of hidden states, h_mean.shape=(n_samples,dim_proj)
        h_sum = (sequence_masks_matrix[:,:,None] * self.h_init).sum(dim=0)
        lengths = sequence_masks_matrix.sum(dim=0)
        h_mean = h_sum / lengths[:,None]
        
        # decode h_mean into input to softmax
        decode = self.receiver_embedding(h_mean)
        return decode    
    def get_embedding(self,data_dir):
        embedding_path = os.path.join(data_dir,'embedding.emd')
        with open(embedding_path) as f:
            for line in f:
                if len(line.split(' ')) == 2:
                    line = line.split(' ')
                    node_number = int(line[0])
                    node_dim = int(line[1])
                    embedding_matrix = np.zeros(shape=(len(self.node_index),node_dim))
                else:
                    line = line.strip().split(' ')
                    node = line[0]
                    embedding = np.array([float(i) for i in line[1:]])
                    try:    
                        index = node_index[node]
                        embedding_matrix[index,:] = embedding
                    except:
                        pass
        return embedding_matrix 

# cross entropy add l2 penalty term
class Penalty_cross_entropy(nn.Module):
    def __init__(self):
        super(Penalty_cross_entropy,self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self, predict, target, penalty_term):
        loss = self.cross_entropy(predict, target)
        for weight in penalty_term:
            loss += 0.0005*(weight**2).sum()
        return loss  
