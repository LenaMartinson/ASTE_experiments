
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings
from gcn import GCN
from syntax_encoder import SyMuxEncoder


class base_model(nn.Module):
    def __init__(self, pretrained_model_path, hidden_dim, dropout, args, class_n=16, 
                 span_average = False, gcn_num_layers=1, gcn_dropout=0.7, use_gcn=False, use_symux=False):
        super().__init__()
        
        self.device = args.device
        self.use_gcn = use_gcn
        self.use_symux = use_symux
        
        # Encoder
        if use_symux:
            self.symux_encoder = SyMuxEncoder(args)
        else:
            self.bert = BertModel.from_pretrained(pretrained_model_path)

        self.dense = nn.Linear(self.bert.pooler.dense.out_features, hidden_dim)
        self.span_average = span_average

        # Classifier
        self.classifier = nn.Linear(hidden_dim * 3 , class_n)
        
        # dropout
        self.layer_drop = nn.Dropout(dropout)
        
        # GCN
        if use_gcn:
            self.gcn = GCN(num_layers=gcn_num_layers, gcn_dropout=gcn_dropout)
        
        
    def forward(self, inputs, weight=None):
        
        #############################################################################################
        # word representation
        bert_token = inputs['bert_token']
        attention_mask = (bert_token > 0).int()
        bert_word_mapback = inputs['bert_word_mapback']
        token_length = inputs['token_length']
        bert_length = inputs['bert_length']
                
        if self.use_symux:
            bert_out, dep_output = self.symux_encoder(
                input_ids=bert_token, 
                input_masks=attention_mask, #[:, :bert_token.shape[1]],
                simple_graph=inputs['adj_matrix'], #[:, :bert_token.shape[1], :bert_token.shape[1]],
                graph=inputs['adj_matrix_simple'], #[:, :bert_token.shape[1], :bert_token.shape[1]], 
                pos=inputs['pos_tags'], #[:, :bert_token.shape[1]], 
                output_attention=False
            )
        else:
            bert_output = self.bert(bert_token, attention_mask = attention_mask)
            if self.use_gcn:
                adj_martixes = inputs['adj_matrix_simple']
                h_gcn, _ = self.gcn(adj_martixes.to(self.device), bert_output.last_hidden_state, self.device)
            bert_out = bert_output.last_hidden_state + h_gcn 
        
        
        bert_seq_indi = sequence_mask(bert_length).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(bert_length) + 1, :] * bert_seq_indi.float()
        word_mapback_one_hot = (F.one_hot(bert_word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
        
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        bert_out = bert_out / wnt.unsqueeze(dim=-1)  # h_i
        #############################################################################################
        # span representation
        
        max_seq = bert_out.shape[1]
        
        token_length_mask = sequence_mask(token_length)
        candidate_tag_mask = torch.triu(
            torch.ones(max_seq,max_seq,dtype=torch.int64,device=bert_out.device),
            diagonal=0).unsqueeze(dim=0) * (token_length_mask.unsqueeze(dim=1) * token_length_mask.unsqueeze(dim=-1))
        
        boundary_table_features = torch.cat(
            [bert_out.unsqueeze(dim=2).repeat(1,1,max_seq,1), bert_out.unsqueeze(dim=1).repeat(1,max_seq,1,1)],
            dim=-1) * candidate_tag_mask.unsqueeze(dim=-1)  # h_i ; h_j 
        span_table_features = form_raw_span_features(
            bert_out, candidate_tag_mask, is_average = self.span_average) # sum(h_i,h_{i+1},...,h_{j})
        
        # h_i ; h_j ; sum(h_i,h_{i+1},...,h_{j})
        table_features = torch.cat([boundary_table_features, span_table_features],dim=-1)
       
        #############################################################################################
        # classifier
        logits = self.classifier(self.layer_drop(table_features)) * candidate_tag_mask.unsqueeze(dim=-1)
        
        outputs = {
            'logits':logits
        }
        
        if 'golden_label' in inputs and inputs['golden_label'] is not None:
            loss = calcualte_loss(logits, inputs['golden_label'],candidate_tag_mask, weight = weight)
            outputs['loss'] = loss
        
        return outputs


def sequence_mask(lengths, max_len=None):

    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) < (lengths.unsqueeze(1))

def form_raw_span_features(v, candidate_tag_mask, is_average = True):
    new_v = v.unsqueeze(dim=1) * candidate_tag_mask.unsqueeze(dim=-1)
    span_features = torch.matmul(new_v.transpose(1,-1).transpose(2,-1), candidate_tag_mask.unsqueeze(dim=1).float()).transpose(2,1).transpose(2,-1)
    
    if is_average:
        _, max_seq, _ = v.shape
        sub_v = torch.tensor(range(1,max_seq+1), device = v.device).unsqueeze(dim=-1)  - torch.tensor(range(max_seq),device = v.device)
        sub_v  = torch.where(sub_v > 0, sub_v, 1).T
        
        span_features = span_features / sub_v.unsqueeze(dim=0).unsqueeze(dim=-1)
        
    return span_features

def calcualte_loss(logits, golden_label, candidate_tag_mask, weight=None):
    loss_func = nn.CrossEntropyLoss(weight = weight, reduction='none')
    return (loss_func(logits.view(-1,logits.shape[-1]), 
                      golden_label.view(-1)
                      ).view(golden_label.size()) * candidate_tag_mask).sum()
    
