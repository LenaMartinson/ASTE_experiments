
import torch.nn as nn
import torch
from huggingface_hub import PyTorchModelHubMixin


class GCN(nn.Module, PyTorchModelHubMixin):

    def __init__(self, emb_dim=768, num_layers=1, gcn_dropout=0.7):
        super(GCN, self).__init__()
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        input_dim = self.emb_dim
        # gcn layer
        self.W = nn.ModuleList([nn.Linear(input_dim, input_dim) for i in range(self.layers)])
        self.gcn_drop = nn.Dropout(gcn_dropout)
        self.relu = nn.ReLU()


    def forward(self, adj, inputs, device):
        # gcn layer

        # adj (batch_size, len, len)
        # inputs (batch_size, len, emb_dim)

#         adj = adj.to_dense()
        if inputs.shape[1] < adj.shape[1]:
            adj = adj[:, :inputs.shape[1], :inputs.shape[1]]
        
        denom = adj.sum(2).unsqueeze(2) + 1                 # batch_size, len, 1
#         mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2) # batch_size, len, 1

        for layer in range(self.layers):
            Ax = torch.bmm(adj, inputs)        # batch_size, len, emb_dim
            AxW = self.W[layer](Ax)            # batch_size, len, emb_dim
            AxW = AxW + self.W[layer](inputs)  # self loop
            AxW = AxW.to(device) / denom
            gAxW = self.relu(AxW)              # batch_size, len, emb_dim
            if layer < self.layers - 1:
                inputs = self.gcn_drop(gAxW)
            else:
                inputs = gAxW
        return inputs, None # mask
