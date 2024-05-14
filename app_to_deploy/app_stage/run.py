
import os
import time
import torch
import random
import argparse
import numpy as np

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from app_stage.ASTE_dataloader import ASTE_End2End_Dataset, ASTE_collate_fn,load_vocab
from app_stage.span_tagging import form_label_id_map, form_sentiment_id_map
from app_stage.evaluate import evaluate_model



def totally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)

def form_weight_n(n):
    if n  > 6:
        weight = torch.ones(n)
        index_range = torch.tensor(range(n))
        weight = weight + ((index_range & 3) > 0)
    else:
        weight = torch.tensor([1.0,2.0,2.0,2.0,3.0,3.0])
    
    return weight


class Args:
    def __init__(self):
        self.max_span_length = 4
        self.device = 'cpu'
        self.version = '1D'
        self.max_len = 1408
        self.order_input = False
        self.random_shuffle = False
        self.train_batch_size = 1
        self.batch_size = 1
        self.init_vocab = 'ai-forever/ruBert-base'
        self.pretrained_model = 'ai-forever/ruBert-base'
        self.dataset_dir = ''
        self.dataset = ''
        self.saved_dir = ''
        self.with_weight = True
        self.span_generation = "Max"
        self.Only_token_head = False
        self.do_lower_case = True
        self.span_average = False

        self.add_pos_enc = True

        self.hidden_dim = 64
        self.dropout_rate = 0.5
        self.lr = 1e-4
        self.bert_lr = 2e-5
        self.l2 = 0.0
        self.dropout_rate = 0.5
        self.adam_epsilon = 1e-8
        self.class_n = 6

        self.upload_model = 'model_13_best.pkl'


def test(raw_text, model_func):

    args = Args()
    
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, do_lower_case=True)
    saved_dir = args.saved_dir + '/' + args.dataset
    ensure_dir(saved_dir)
     
    vocab = load_vocab(dataset_dir = '')

    label2id, id2label = form_label_id_map(args.version)
    senti2id, id2senti = form_sentiment_id_map()
    
    vocab['label_vocab'] = dict(label2id=label2id,id2label=id2label)
    vocab['senti_vocab'] = dict(senti2id=senti2id,id2senti=id2senti)

    class_n = len(label2id)
    args.class_n = class_n
    weight = None
    if args.with_weight is True:
        weight = form_weight_n(class_n).to(args.device)
    print('> label2id:', label2id)
    print('> weight:', args.with_weight, weight)
    print(args)

    print('> Load model...')
    base_model = model_func(pretrained_model_path = args.pretrained_model,
                            hidden_dim = args.hidden_dim,
                            dropout = args.dropout_rate,
                            args = args,
                            class_n = class_n,
                            span_average = args.span_average,
                            gcn_num_layers=3, 
                            gcn_dropout=0.5
                            ).to(args.device)
    
    if args.upload_model is not None:
        base_model = torch.load(args.upload_model, map_location=torch.device('cpu'))
        base_model.eval()
    
    test_dataset = ASTE_End2End_Dataset(raw_text,
                                        version = args.version,
                                        vocab = vocab,
                                        tokenizer = tokenizer,
                                        max_len = args.max_len)
    
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, collate_fn = ASTE_collate_fn, shuffle = False)


    _, test_results, saved_preds = evaluate_model(base_model, test_dataset, test_dataloader, 
                                             id2senti = id2senti, 
                                             device = args.device, 
                                             version = args.version, 
                                             weight = weight,
                                             saved_file="users_result.json")
    return saved_preds



def get_bert_optimizer(model, args):

    no_decay = ['bias', 'LayerNorm.weight']
    diff_part = ['bert.embeddings', 'bert.encoder']
    
    

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": args.l2,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.bert_lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": args.l2,
            "lr": args.lr
        },
        {
            "params": [p for n, p in model.named_parameters() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
            "weight_decay": 0.0,
            "lr": args.lr
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)

    return optimizer

def set_random_seed(seed):

    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic =True

def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_dir', type=str,default='./data/ASTE-Data-V2-EMNLP2020')
    parser.add_argument('--saved_dir', type=str, default='saved_models')
    parser.add_argument('--saved_file', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
    parser.add_argument('--dataset', type=str, default='14lap')
    parser.add_argument('--add_pos_enc', default=False)
    
    parser.add_argument('--version', type=str, default='3D', choices=['3D','2D','1D'])
    
    parser.add_argument('--seed', type=int, default=64)
    
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=256)
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bert_lr', type=float, default=2e-5)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    
    # loss
    parser.add_argument('--with_weight', default=True, action='store_true')
    parser.add_argument('--span_average', default=False, action='store_true')
    
    args = parser.parse_args()
    
    return args


def run(raw_text):
    from app_stage.model import base_model
    # args = get_parameters()
    # args.with_weight = True # default true here
        
    preds = test(raw_text, base_model)
    return preds
    

if __name__ == '__main__':
    run()
