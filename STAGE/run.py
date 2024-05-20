
import os
import time
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import wandb

from transformers import BertTokenizer
from torch.utils.data import DataLoader

from ASTE_dataloader import ASTE_End2End_Dataset, ASTE_collate_fn,load_vocab
from span_tagging import form_label_id_map, form_sentiment_id_map
from evaluate import evaluate_model,print_evaluate_dict


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

def train_and_evaluate(model_func, args, save_specific=False):
    print('=========================================================================================================')
    
    set_random_seed(args.seed)
    
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, do_lower_case=True)
    dataset_dir = args.dataset_dir + '/' + args.dataset
    saved_dir = args.saved_dir + '/' + args.dataset
    ensure_dir(saved_dir)
     
    vocab = load_vocab(dataset_dir = dataset_dir)

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
                            gcn_dropout=0.5,
                            use_gcn=args.use_gcn, 
                            use_symux=args.use_gcn
                            ).to(args.device)
    
    print('> # parameters', totally_parameters(base_model))
    
    print('> Load dataset...')
    train_dataset = ASTE_End2End_Dataset(file_name = os.path.join(dataset_dir, 'train_triplets.txt'),
                                        version = args.version,
                                        vocab = vocab,
                                        tokenizer = tokenizer,
                                        max_len = args.max_len,
                                        add_pos_mode = args.add_pos_mode,
                                        add_simple_mode = args.add_simple_mode)
    valid_dataset = ASTE_End2End_Dataset(file_name = os.path.join(dataset_dir, 'dev_triplets.txt'),
                                        version = args.version,
                                        vocab = vocab,
                                        tokenizer = tokenizer,
                                        max_len = args.max_len,
                                        add_pos_mode = args.add_pos_mode,
                                        add_simple_mode = args.add_simple_mode)
    test_dataset = ASTE_End2End_Dataset(file_name = os.path.join(dataset_dir, 'test_triplets.txt'),
                                        version = args.version,
                                        vocab = vocab,
                                        tokenizer = tokenizer,
                                        max_len = args.max_len,
                                        add_pos_mode = args.add_pos_mode,
                                        add_simple_mode = args.add_simple_mode)
    
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, collate_fn = ASTE_collate_fn, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, collate_fn = ASTE_collate_fn, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, collate_fn = ASTE_collate_fn, shuffle = False)


    optimizer = get_bert_optimizer(base_model, args)

    triplet_max_f1 = 0.0

    best_model_save_path = saved_dir +  '/' + args.dataset + '_' +  args.version + '_' + str(args.with_weight) +'_best.pkl'
    
    if args.use_wandb:
        wandb.init(
            project='aste-STAGE',
            config=args
        )
    
    scaler = torch.cuda.amp.GradScaler()
    
    print('> Training...')
    for epoch in range(1, args.num_epoch+1):
        train_loss = 0.
        total_step = 0
        
        epoch_begin = time.time()
        for batch in tqdm(train_dataloader, 'Epoch:{}'.format(epoch)):
            base_model.train()
            optimizer.zero_grad()
            
            inputs = {k:v.to(args.device) for k,v in batch.items()}
            outputs = base_model(inputs, weight)
            
            loss = outputs['loss']
            
            total_step += 1
            train_loss += loss.item()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        
        valid_loss, valid_results = evaluate_model(base_model, valid_dataset, valid_dataloader, 
                                                   id2senti = id2senti, 
                                                   device = args.device, 
                                                   version = args.version, 
                                                   weight = weight)
        
        if args.use_wandb:
            wandb.log({
                'train_loss': train_loss / total_step,
                'val_loss': valid_loss,
                'val_precision': valid_results[0]['triplet']['precision'],
                'val_recall':valid_results[0]['triplet']['recall'],
                'val_f1': valid_results[0]['triplet']['f1'],
            })

        print('\ttrain_loss:{:.4f}\tvalid_loss:{:.4f} [{:.4f}s]'.format(train_loss / total_step, 
                                                                        valid_loss,
                                                                        time.time()-epoch_begin))
                
        print('\ttriplet_precision:{:.4f} \ttriplet_recall:{:.4f} \ttriplet_f1:{:.4f}'.format( 
                                                    valid_results[0]['triplet']['precision'], 
                                                    valid_results[0]['triplet']['recall'], 
                                                    valid_results[0]['triplet']['f1'],
                                                    ))
            
    
    saved_file = (saved_dir + '/' + args.saved_file) if args.saved_file is not None else None
    
    print('> Testing...')
    # model performance on the test set
    _, test_results = evaluate_model(base_model, test_dataset, test_dataloader, 
                                             id2senti = id2senti, 
                                             device = args.device, 
                                             version = args.version, 
                                             weight = weight,
                                             saved_file= saved_file)
    

    print('------------------------------')
    
    print('Dataset:{}, test_f1:{:.2f} | version:{} lr:{} bert_lr:{} seed:{} dropout:{}'.format(args.dataset,test_results[0]['triplet']['f1'],
                                                                                                 args.version, args.lr, args.bert_lr, 
                                                                                                 args.seed, args.dropout_rate))
    print_evaluate_dict(test_results)

    if args.use_wandb:
        wandb.finish()

    return test_results




def get_bert_optimizer(model, args):

    no_decay = ['bias', 'LayerNorm.weight']
    diff_part = ['symux_encoder.bert.embeddings', 'symux_encoder.bert.encoder']
    
    

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
    parser.add_argument('--dataset_dir', type=str,default='')
    parser.add_argument('--saved_dir', type=str, default='saved_models')
    parser.add_argument('--saved_file', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default='bert-base-uncased')
    parser.add_argument('--dataset', type=str, default='SBN/datasets/sbn-dataset-sent')
    parser.add_argument('--add_pos_enc', default=False)
    
    # SyMuxEncoder
    parser.add_argument("--pos_num", default=18, type=int)
    parser.add_argument("--pos_dim", default=20, type=int)
    parser.add_argument("--dep_dim", default=20, type=int)
    parser.add_argument("--num_layer", default=1, type=int)
    parser.add_argument("--bert_dropout", default=0.1, type=float)
    parser.add_argument("--output_dropout", default=0.1, type=float)
    parser.add_argument("--w_size", default=5, type=int)
    parser.add_argument("--dep_num", default=20, type=int)
    parser.add_argument("--bert_feature_dim", default=768, type=int, help="feature dim for bert")
    
    parser.add_argument("--add_pos_mode", default='pos', type=str)
    parser.add_argument("--add_simple_mode", default=False)
    parser.add_argument("--use_wandb", default=False)
    parser.add_argument("--use_gcn", default=False)
    parser.add_argument("--use_symux", default=False)
    
    parser.add_argument('--version', type=str, default='3D', choices=['3D','2D','1D'])
    
    parser.add_argument('--seed', type=int, default=64)
    
    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_len', type=int, default=256)
    
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--bert_lr', type=float, default=2e-5)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    
    # loss
    parser.add_argument('--with_weight', default=True, action='store_true')
    parser.add_argument('--span_average', default=False, action='store_true')
    
    args = parser.parse_args()
    
    return args


def run():
    from model import base_model
    args = get_parameters()

    if args.use_gcn and args.use_symux:
        raise "Don't use GCN and SyMuxEncodre at the same time"
        
    train_and_evaluate(base_model, args)
    

if __name__ == '__main__':
    run()
