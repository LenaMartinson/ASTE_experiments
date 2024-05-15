
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings
from huggingface_hub import PyTorchModelHubMixin
# from gcn import GCN


class base_model(nn.Module, PyTorchModelHubMixin):
    def __init__(self, pretrained_model_path, hidden_dim, dropout, device, class_n=16, 
                 span_average = False, add_pos_enc=False, gcn_num_layers=1, gcn_dropout=0.7):
        super().__init__()
        
        self.device = device
        
        # Encoder
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        bert_config = self.bert.config
        if add_pos_enc:
            print("Change pos_embeddings to 1536 len...")

            # word_emb
            word_emb = self.bert.embeddings.word_embeddings.weight.data

            # token_type_emb
            token_type_emb = self.bert.embeddings.token_type_embeddings.weight.data

            # pos_enc
            pos_enc = self.bert.embeddings.position_embeddings.weight.data
            new_pos_enc = torch.concat((pos_enc, pos_enc * 2, pos_enc * 4), axis=0)
            # new_pos_enc = torch.repeat_interleave(pos_enc, 3, dim=0)

            # new config and embeddings structure
            bert_config.update({'max_position_embeddings': 1536})
            self.bert.embeddings = BertEmbeddings(bert_config)

            # return pretrained weights
            self.bert.embeddings.word_embeddings.weight.data = word_emb
            self.bert.embeddings.token_type_embeddings.weight.data = token_type_emb
            self.bert.embeddings.position_embeddings.weight.data = new_pos_enc

            print("Changed successful!")

        self.dense = nn.Linear(self.bert.pooler.dense.out_features, hidden_dim)
        self.span_average = span_average

        # Classifier
        self.classifier = nn.Linear(hidden_dim * 3 , class_n)
        
        # dropout
        self.layer_drop = nn.Dropout(dropout)
        
        # GCN
        # self.gcn = GCN(num_layers=gcn_num_layers, gcn_dropout=gcn_dropout)
        
        
    def forward(self, inputs, weight=None):
        
        #############################################################################################
        # word representation
        bert_token = inputs['bert_token']
        attention_mask = (bert_token > 0).int()
        bert_word_mapback = inputs['bert_word_mapback']
        token_length = inputs['token_length']
        bert_length = inputs['bert_length']
        
        # adj_martixes = inputs['adj_matrix']
                
        bert_output = self.bert(bert_token, attention_mask = attention_mask)
        # h_gcn, _ = self.gcn(adj_martixes.to(self.device), bert_output.last_hidden_state, self.device)
        bert_out = bert_output.last_hidden_state # + h_gcn # \hat{h}
        
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
    
