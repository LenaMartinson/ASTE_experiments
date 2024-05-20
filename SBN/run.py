
import os
import argparse
import tqdm
import torch
import torch.nn.functional as F
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup
from data_BIO_loader import DataTterator
from data_BIO_loader import MyDataset
from model import stage_2_features_generation, Step_1, Step_2_forward, Step_2_reverse, Loss
from Metric import Metric
from eval_features import unbatch_data
import wandb
from transformers.models.bert.modeling_bert import BertEmbeddings
from gcn import GCN
from syntax_encoder import SyMuxEncoder
from syntax_features import make_adj_matrix

import logging
from datetime import datetime

import time
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


sentiment2id = {'none': 0, 'positive': 1, 'negative': 2, 'neutral': 3, 'start': 4}

from datetime import datetime

now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = logging.getLogger("test")
logger.setLevel(level=logging.INFO)

handler = logging.FileHandler("/kaggle/working/log/"+now+".log", encoding='utf-8')
handler.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)



def eval(bert_model, step_1_model, step_2_forward, step_2_reverse, dataset, args, gcn_model=None, symux_encoder=None, mode='val'):
    with torch.no_grad():
        if args.use_gcn:
            gcn_model.eveal()
        if args.use_symux:
            symux_encoder.eval()
        else:
            bert_model.eval()
        step_1_model.eval()
        step_2_forward.eval()
        step_2_reverse.eval()
        '''真实结果'''
        gold_instances = []
        '''前向预测结果'''
        forward_stage1_pred_aspect_result, forward_stage1_pred_aspect_with_sentiment, \
        forward_stage1_pred_aspect_sentiment_logit, forward_stage2_pred_opinion_result, \
        forward_stage2_pred_opinion_sentiment_logit = [],[],[],[],[]

        '''反向预测结果'''
        reverse_stage1_pred_opinion_result, reverse_stage1_pred_opinion_with_sentiment, \
        reverse_stage1_pred_opinion_sentiment_logit, reverse_stage2_pred_aspect_result, \
        reverse_stage2_pred_aspect_sentiment_logit = [], [], [], [], []
        
        tot_loss = 0
        tot_kl_loss = 0

        for j in range(dataset.batch_count):
            tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
            spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
            reverse_aspect_label_tensor, related_spans_tensor, sentence_length = dataset.get_batch(j)

            if not args.use_symux:
                bert_output = bert_model(input_ids=tokens_tensor, attention_mask=attention_mask)
                bert_out = bert_output.last_hidden_state
            
            if args.use_symux or args.use_gcn:
                sentence_adj = []
                sentence_pos = []
                diff_sentence_adj = []
                max_batch_len = 0
                for sent in sentence_length:
                    max_batch_len = max(max_batch_len, len(sent[0]))
                for sent in sentence_length:
                    sent_adj, diff_sent_adj, sent_pos_tags = make_adj_matrix(sent[0], 
                                                                            max_batch_len, 
                                                                            args.add_pos_mode, 
                                                                            args.add_simple_mode) 
                    sentence_adj.append(sent_adj)
                    sentence_pos.append(sent_pos_tags)
                    diff_sentence_adj.append(diff_sent_adj)
                sentence_adj = torch.cat([i.unsqueeze(0) for i in sentence_adj], axis=0).to_dense().to(args.device)
                diff_sentence_adj = torch.cat([i.unsqueeze(0) for i in diff_sentence_adj], axis=0).to_dense().to(args.device)
                sentence_pos = torch.cat([i.unsqueeze(0) for i in sentence_pos], axis=0).to(args.device)

                if args.use_symux:
                    encoder_output, dep_output = symux_encoder(
                        input_ids=tokens_tensor[:, :max_batch_len].to("cuda:0"), 
                        input_masks=attention_mask[:, :max_batch_len].to("cuda:0"),
                        simple_graph=sentence_adj,
                        graph=diff_sentence_adj, 
                        pos=sentence_pos, 
                        output_attention=False
                    )
                    bert_out = encoder_output

                if args.use_gcn:
                    h_gcn, _ = gcn_model(sentence_adj, bert_output.last_hidden_state, args.device)
                    bert_out = bert_output.last_hidden_state  + h_gcn
                    

            aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
                cnn_spans_mask_tensor = step_1_model(
                    bert_out, attention_mask, bert_spans_tensor, spans_mask_tensor,
                    related_spans_tensor, sentence_length)
            
            # CALCULATE LOSS
            if mode == 'val':
                all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
                all_spans_embedding, all_span_mask = stage_2_features_generation(bert_out,
                                                                             attention_mask, bert_spans_tensor,
                                                                             spans_mask_tensor, forward_embedding,
                                                                             spans_aspect_tensor,
                                                                             spans_opinion_label_tensor)
                all_reverse_aspect_tensor, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                reverse_spans_embedding, reverse_span_mask = stage_2_features_generation(bert_out,
                                                                                     attention_mask, bert_spans_tensor,
                                                                                     spans_mask_tensor, reverse_embedding,
                                                                                     reverse_opinion_tensor,
                                                                                     reverse_aspect_label_tensor)

                step_2_opinion_class_logits, opinion_attention = step_2_forward(all_spans_embedding, 
                                                                                     all_span_mask, all_span_aspect_tensor)
                step_2_aspect_class_logits, aspect_attention = step_2_reverse(reverse_spans_embedding,
                    reverse_span_mask, all_reverse_opinion_tensor)

                loss, kl_loss = Loss(spans_ner_label_tensor, aspect_class_logits, all_span_opinion_tensor, step_2_opinion_class_logits,
                            spans_mask_tensor, all_span_mask, reverse_ner_label_tensor, opinion_class_logits,
                            all_reverse_aspect_tensor, step_2_aspect_class_logits, cnn_spans_mask_tensor, reverse_span_mask,
                            spans_embedding, related_spans_tensor, args)

                tot_loss += loss.item()

            '''Batch Update'''
            pred_aspect_logits = torch.argmax(F.softmax(aspect_class_logits, dim=2), dim=2)
            pred_sentiment_ligits = F.softmax(aspect_class_logits, dim=2)
            pred_aspect_logits = torch.where(spans_mask_tensor == 1, pred_aspect_logits,
                                             torch.tensor(0).type_as(pred_aspect_logits))

            reverse_pred_stage1_logits = torch.argmax(F.softmax(opinion_class_logits, dim=2), dim=2)
            reverse_pred_sentiment_ligits = F.softmax(opinion_class_logits, dim=2)
            reverse_pred_stage1_logits = torch.where(spans_mask_tensor == 1, reverse_pred_stage1_logits,
                                             torch.tensor(0).type_as(reverse_pred_stage1_logits))

            '''true result synthesis'''
            gold_instances.append(dataset.get_instances(j))
            
            
            '''Bidirectional prediction'''
            if torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0] == 0:
                forward_stage1_pred_aspect_result.append(torch.full_like(spans_aspect_tensor, -1))
                forward_stage1_pred_aspect_with_sentiment.append(pred_aspect_logits)
                forward_stage1_pred_aspect_sentiment_logit.append(pred_sentiment_ligits)
                forward_stage2_pred_opinion_result.append(torch.full_like(spans_opinion_label_tensor, -1))
                forward_stage2_pred_opinion_sentiment_logit.append(
                    torch.full_like(spans_opinion_label_tensor.unsqueeze(-1).expand(-1, -1, len(sentiment2id)), -1))

            else:
                pred_aspect_spans = torch.chunk(torch.nonzero(pred_aspect_logits, as_tuple=False),
                                                torch.nonzero(pred_aspect_logits, as_tuple=False).shape[0], dim=0)
                pred_span_aspect_tensor = None
                for pred_aspect_span in pred_aspect_spans:
                    batch_num = pred_aspect_span.squeeze()[0]
                    span_aspect_tensor_unspilt_1 = bert_spans_tensor[batch_num, pred_aspect_span.squeeze()[1], :2]
                    span_aspect_tensor_unspilt = torch.tensor(
                        (batch_num, span_aspect_tensor_unspilt_1[0], span_aspect_tensor_unspilt_1[1])).unsqueeze(0)
                    if pred_span_aspect_tensor is None:
                        pred_span_aspect_tensor = span_aspect_tensor_unspilt
                    else:
                        pred_span_aspect_tensor = torch.cat((pred_span_aspect_tensor, span_aspect_tensor_unspilt),dim=0)

                all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
                    all_spans_embedding, all_span_mask = stage_2_features_generation(
                        bert_out, attention_mask, bert_spans_tensor, spans_mask_tensor,
                        forward_embedding, pred_span_aspect_tensor)


                step_2_opinion_class_logits, opinion_attention = step_2_forward(all_spans_embedding, all_span_mask,
                                                                         all_span_aspect_tensor)

                forward_stage1_pred_aspect_result.append(pred_span_aspect_tensor)
                forward_stage1_pred_aspect_with_sentiment.append(pred_aspect_logits)
                forward_stage1_pred_aspect_sentiment_logit.append(pred_sentiment_ligits)
                forward_stage2_pred_opinion_result.append(torch.argmax(F.softmax(step_2_opinion_class_logits, dim=2), dim=2))
                forward_stage2_pred_opinion_sentiment_logit.append(F.softmax(step_2_opinion_class_logits, dim=2))
            '''Reverse prediction'''
            if torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0] == 0:
                reverse_stage1_pred_opinion_result.append(torch.full_like(reverse_opinion_tensor, -1))
                reverse_stage1_pred_opinion_with_sentiment.append(reverse_pred_stage1_logits)
                reverse_stage1_pred_opinion_sentiment_logit.append(reverse_pred_sentiment_ligits)
                reverse_stage2_pred_aspect_result.append(torch.full_like(reverse_aspect_label_tensor, -1))
                reverse_stage2_pred_aspect_sentiment_logit.append(
                    torch.full_like(reverse_aspect_label_tensor.unsqueeze(-1).expand(-1, -1, len(sentiment2id)), -1))
            else:
                reverse_pred_opinion_spans = torch.chunk(torch.nonzero(reverse_pred_stage1_logits, as_tuple=False),
                                                torch.nonzero(reverse_pred_stage1_logits, as_tuple=False).shape[0], dim=0)
                reverse_span_opinion_tensor = None
                for reverse_pred_opinion_span in reverse_pred_opinion_spans:
                    batch_num = reverse_pred_opinion_span.squeeze()[0]
                    reverse_opinion_tensor_unspilt = bert_spans_tensor[batch_num, reverse_pred_opinion_span.squeeze()[1], :2]
                    reverse_opinion_tensor_unspilt = torch.tensor(
                        (batch_num, reverse_opinion_tensor_unspilt[0], reverse_opinion_tensor_unspilt[1])).unsqueeze(0)
                    if reverse_span_opinion_tensor is None:
                        reverse_span_opinion_tensor = reverse_opinion_tensor_unspilt
                    else:
                        reverse_span_opinion_tensor = torch.cat((reverse_span_opinion_tensor, reverse_opinion_tensor_unspilt), dim=0)
           
                all_reverse_aspect_tensor, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                reverse_spans_embedding, reverse_span_mask = stage_2_features_generation(
                        bert_out,
                        attention_mask,
                        bert_spans_tensor,
                        spans_mask_tensor,
                        reverse_embedding,
                        reverse_span_opinion_tensor)

                reverse_aspect_class_logits, reverse_aspect_attention = step_2_reverse(reverse_spans_embedding,
                                                                                reverse_span_mask,
                                                                                all_reverse_opinion_tensor)

                reverse_stage1_pred_opinion_result.append(reverse_span_opinion_tensor)
                reverse_stage1_pred_opinion_with_sentiment.append(reverse_pred_stage1_logits)
                reverse_stage1_pred_opinion_sentiment_logit.append(reverse_pred_sentiment_ligits)
                reverse_stage2_pred_aspect_result.append(torch.argmax(F.softmax(reverse_aspect_class_logits, dim=2), dim=2))
                reverse_stage2_pred_aspect_sentiment_logit.append(F.softmax(reverse_aspect_class_logits, dim=2))
            

        gold_instances = [x for i in gold_instances for x in i]
        forward_pred_data = (forward_stage1_pred_aspect_result, forward_stage1_pred_aspect_with_sentiment,
                             forward_stage1_pred_aspect_sentiment_logit, forward_stage2_pred_opinion_result,
                             forward_stage2_pred_opinion_sentiment_logit)
        forward_pred_result = unbatch_data(forward_pred_data)

        reverse_pred_data = (reverse_stage1_pred_opinion_result, reverse_stage1_pred_opinion_with_sentiment,
                             reverse_stage1_pred_opinion_sentiment_logit, reverse_stage2_pred_aspect_result,
                             reverse_stage2_pred_aspect_sentiment_logit)
        reverse_pred_result = unbatch_data(reverse_pred_data)

        metric = Metric(args, forward_pred_result, reverse_pred_result, gold_instances)
        aspect_result, opinion_result, apce_result, pair_result, triplet_result = metric.score_triples()

        
        logger.info(
            'aspect precision: {}\taspect recall: {:.8f}\taspect f1: {:.8f}'.format(aspect_result[0], aspect_result[1], aspect_result[2]))
        logger.info(
            'opinion precision: {}\topinion recall: {:.8f}\topinion f1: {:.8f}'.format(opinion_result[0],
                                                                                        opinion_result[1],
                                                                                        opinion_result[2]))
        logger.info('APCE precision: {}\tAPCE recall: {:.8f}\tAPCE f1: {:.8f}'.format(apce_result[0],
                                                                                apce_result[1], apce_result[2]))
        logger.info('pair precision: {}\tpair recall: {:.8f}\tpair f1: {:.8f}'.format(pair_result[0],
                                                                                          pair_result[1],
                                                                                          pair_result[2]))
        logger.info('triple precision: {}\ttriple recall: {:.8f}\ttriple f1: {:.8f}'.format(triplet_result[0],
                                                                                          triplet_result[1],
                                                                                          triplet_result[2]))

    return aspect_result, opinion_result, apce_result, pair_result, triplet_result, tot_loss


def train(args):

    if args.wandb_logging:
        wandb_ran = wandb.init(
            project='aste-SBN',
            config=args
        )

    if args.dataset_path == './datasets/BIO_form/':
        train_path = args.dataset_path + args.dataset + "/train.json"
        dev_path = args.dataset_path + args.dataset + "/dev.json"
        test_path = args.dataset_path + args.dataset + "/test.json"
    else:
        train_path = args.dataset_path + args.dataset + "/train_full.txt"
        dev_path = args.dataset_path + args.dataset + "/dev_full.txt"
        test_path = args.dataset_path + args.dataset + "/test_full.txt"

    print('-------------------------------')
    print('Start loading the test set')
    logger.info('Start loading the test set')
    test_datasets = MyDataset(args, test_path, if_train=False)
    testset = DataTterator(test_datasets, args)
    print('The test set is loaded')
    logger.info('The test set is loaded')
    print('-------------------------------')
    
    if args.use_gcn:
        gcn = GCN(emb_dim=args.bert_feature_dim).to(args.device)
        gcn_param_optimizer = list(gcn.named_parameters())

    if not args.use_symux:
        Bert = BertModel.from_pretrained(args.init_model)
        bert_config = Bert.config

        if args.add_pos_enc:
            print("Change pos_embeddings to 1536 len...")

            # word_emb
            word_emb = Bert.embeddings.word_embeddings.weight.data

            # token_type_emb
            token_type_emb = Bert.embeddings.token_type_embeddings.weight.data

            # pos_enc
            pos_enc = Bert.embeddings.position_embeddings.weight.data
            new_pos_enc = torch.concat((pos_enc, pos_enc * 2, pos_enc * 4), axis=0)
            # new_pos_enc = torch.repeat_interleave(pos_enc, 3, dim=0)

            # new config and embeddings structure
            bert_config.update({'max_position_embeddings': 1536})
            Bert.embeddings = BertEmbeddings(bert_config)

            # return pretrained weights
            Bert.embeddings.word_embeddings.weight.data = word_emb
            Bert.embeddings.token_type_embeddings.weight.data = token_type_emb
            Bert.embeddings.position_embeddings.weight.data = new_pos_enc

            print("Changed successful!")
        
        Bert.to(args.device)
        bert_param_optimizer = list(Bert.named_parameters())
    else:
        symux_encoder = SyMuxEncoder(args).to(args.device)
        symux_param_optimizer = list(symux_encoder.named_parameters())
        bert_config = symux_encoder.bert.config



    step_1_model = Step_1(args, bert_config)
    step_1_model.to(args.device)
    step_1_param_optimizer = list(step_1_model.named_parameters())

    step2_forward_model = Step_2_forward(args, bert_config)
    step2_forward_model.to(args.device)
    forward_step2_param_optimizer = list(step2_forward_model.named_parameters())

    step2_reverse_model = Step_2_reverse(args, bert_config)
    step2_reverse_model.to(args.device)
    reverse_step2_param_optimizer = list(step2_reverse_model.named_parameters())

    if args.use_symux:
        training_param_optimizer = [
            {'params': [p for n, p in symux_param_optimizer]},
            {'params': [p for n, p in step_1_param_optimizer], 'lr': args.task_learning_rate},
            {'params': [p for n, p in forward_step2_param_optimizer], 'lr': args.task_learning_rate},
            {'params': [p for n, p in reverse_step2_param_optimizer], 'lr': args.task_learning_rate}]
    elif args.use_gcn:
        training_param_optimizer = [
            {'params': [p for n, p in gcn_param_optimizer]},
            {'params': [p for n, p in bert_param_optimizer]},
            {'params': [p for n, p in step_1_param_optimizer], 'lr': args.task_learning_rate},
            {'params': [p for n, p in forward_step2_param_optimizer], 'lr': args.task_learning_rate},
            {'params': [p for n, p in reverse_step2_param_optimizer], 'lr': args.task_learning_rate}]
    else:
        training_param_optimizer = [
            {'params': [p for n, p in bert_param_optimizer]},
            {'params': [p for n, p in step_1_param_optimizer], 'lr': args.task_learning_rate},
            {'params': [p for n, p in forward_step2_param_optimizer], 'lr': args.task_learning_rate},
            {'params': [p for n, p in reverse_step2_param_optimizer], 'lr': args.task_learning_rate}]
    optimizer = AdamW(training_param_optimizer, lr=args.learning_rate)

    if args.model_to_upload != None:
        
        model_path = args.model_to_upload
        if args.device == 'cpu':
            state = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(model_path)
        
        new_state = {}
        for i in state['bert_model'].keys():
            new_state[i[7:]] = state['bert_model'][i]
        Bert.load_state_dict(new_state)
        
        new_state = {}
        for i in state['step_1_model'].keys():
            new_state[i[7:]] = state['step_1_model'][i]
        step_1_model.load_state_dict(new_state)
        
        new_state = {}
        for i in state['step2_forward_model'].keys():
            new_state[i[7:]] = state['step2_forward_model'][i]
        step2_forward_model.load_state_dict(new_state)
        
        new_state = {}
        for i in state['step2_reverse_model'].keys():
            new_state[i[7:]] = state['step2_reverse_model'][i]
        step2_reverse_model.load_state_dict(new_state)
        
        optimizer.load_state_dict(state['optimizer'])
        
        with torch.no_grad():
            Bert.eval()
            step_1_model.eval()
            step2_forward_model.eval()
            step2_reverse_model.eval()


    if args.multi_gpu:
        if args.use_symux:
            symux_encoder = torch.nn.DataParallel(symux_encoder)
        if args.use_gcn:
            gcn = torch.nn.DataParallel(gcn)
        if not args.use_symux:
            Bert = torch.nn.DataParallel(Bert)
        step_1_model = torch.nn.DataParallel(step_1_model)
        step2_forward_model = torch.nn.DataParallel(step2_forward_model)
        step2_reverse_model = torch.nn.DataParallel(step2_reverse_model)  
        
    if args.mode == 'train':
        print('-------------------------------')
        logger.info('Start loading the training and verification set')
        print('Start loading the training and verification set')
        train_datasets = MyDataset(args, train_path, if_train=True)
        trainset = DataTterator(train_datasets, args)
        print("Train features build completed")

        print("Dev features build beginning")
        dev_datasets = MyDataset(args, dev_path, if_train=False)
        devset = DataTterator(dev_datasets, args)
        print('The training set and verification set are loaded')
        logger.info('The training set and verification set are loaded')
        print('-------------------------------')
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        # scheduler
        if args.whether_warm_up:
            training_steps = args.epochs * trainset.batch_count
            warmup_steps = int(training_steps * args.warm_up)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)

        tot_loss = 0
        tot_kl_loss = 0
        for i in range(args.epochs):
            logger.info(('Epoch:{}'.format(i)))
            for j in tqdm.trange(trainset.batch_count):
                if args.use_gcn:
                    gcn.train()
                if args.use_symux:
                    symux_encoder.train()
                else:
                    Bert.train()
                step_1_model.train()
                step2_forward_model.train()
                step2_reverse_model.train()
                
                
                optimizer.zero_grad()

                tokens_tensor, attention_mask, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, \
                spans_aspect_tensor, spans_opinion_label_tensor, reverse_ner_label_tensor, reverse_opinion_tensor, \
                reverse_aspect_label_tensor, related_spans_tensor, sentence_length = trainset.get_batch(j)
                
                if not args.use_symux:
                    bert_output = Bert(input_ids=tokens_tensor, attention_mask=attention_mask)
                    bert_out = bert_output.last_hidden_state
            
                if args.use_symux or args.use_gcn:
                    sentence_adj = []
                    sentence_pos = []
                    diff_sentence_adj = []
                    max_batch_len = 0
                    for sent in sentence_length:
                        max_batch_len = max(max_batch_len, len(sent[0]))
                    for sent in sentence_length:
                        sent_adj, diff_sent_adj, sent_pos_tags = make_adj_matrix(sent[0], 
                                                                                max_batch_len, 
                                                                                args.add_pos_mode, 
                                                                                args.add_simple_mode) 
                        sentence_adj.append(sent_adj)
                        sentence_pos.append(sent_pos_tags)
                        diff_sentence_adj.append(diff_sent_adj)
                    sentence_adj = torch.cat([i.unsqueeze(0) for i in sentence_adj], axis=0).to_dense().to(args.device)
                    diff_sentence_adj = torch.cat([i.unsqueeze(0) for i in diff_sentence_adj], axis=0).to_dense().to(args.device)
                    sentence_pos = torch.cat([i.unsqueeze(0) for i in sentence_pos], axis=0).to(args.device)

                    if args.use_symux:
                        encoder_output, dep_output = symux_encoder(
                            input_ids=tokens_tensor[:, :max_batch_len].to("cuda:0"), 
                            input_masks=attention_mask[:, :max_batch_len].to("cuda:0"),
                            simple_graph=sentence_adj,
                            graph=diff_sentence_adj, 
                            pos=sentence_pos, 
                            output_attention=False
                        )
                        bert_out = encoder_output

                    if args.use_gcn:
                        h_gcn, _ = gcn(sentence_adj, bert_output.last_hidden_state, args.device)
                        bert_out = bert_output.last_hidden_state  + h_gcn
                
                aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
                    cnn_spans_mask_tensor = step_1_model(bert_out,
                                                        attention_mask,
                                                        bert_spans_tensor,
                                                        spans_mask_tensor,
                                                        related_spans_tensor,
                                                        sentence_length)

                '''Batch Update'''
                all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
                all_spans_embedding, all_span_mask = stage_2_features_generation(bert_out,
                                                                             attention_mask, bert_spans_tensor,
                                                                             spans_mask_tensor, forward_embedding,
                                                                             spans_aspect_tensor,
                                                                             spans_opinion_label_tensor)
                all_reverse_aspect_tensor, all_reverse_opinion_tensor, reverse_bert_embedding, reverse_attention_mask, \
                reverse_spans_embedding, reverse_span_mask = stage_2_features_generation(bert_out,
                                                                                     attention_mask, bert_spans_tensor,
                                                                                     spans_mask_tensor, reverse_embedding,
                                                                                     reverse_opinion_tensor,
                                                                                     reverse_aspect_label_tensor)

                step_2_opinion_class_logits, opinion_attention = step2_forward_model(all_spans_embedding, 
                                                                                     all_span_mask, all_span_aspect_tensor)
                step_2_aspect_class_logits, aspect_attention = step2_reverse_model(reverse_spans_embedding,
                    reverse_span_mask, all_reverse_opinion_tensor)
                

                loss, kl_loss = Loss(spans_ner_label_tensor, aspect_class_logits, all_span_opinion_tensor, step_2_opinion_class_logits,
                            spans_mask_tensor, all_span_mask, reverse_ner_label_tensor, opinion_class_logits,
                            all_reverse_aspect_tensor, step_2_aspect_class_logits, cnn_spans_mask_tensor, reverse_span_mask,
                            spans_embedding, related_spans_tensor, args)
                
                if args.accumulation_steps > 1:
                    loss = loss / args.accumulation_steps
                    loss.backward()
                    if ((j + 1) % args.accumulation_steps) == 0:
                        optimizer.step()
                        if args.whether_warm_up:
                            scheduler.step()
                else:
                    loss.backward()
                    optimizer.step()
                    if args.whether_warm_up:
                        scheduler.step()
                tot_loss += loss.item()
                tot_kl_loss += kl_loss
            
            
            logger.info(('Loss:', tot_loss))
            logger.info(('KL_Loss:', tot_kl_loss))
            

            print('Evaluating, please wait')
            aspect_result, opinion_result, apce_result, pair_result, triplet_result, val_tot_loss = eval(gcn, Bert, step_1_model,
                                                                                           step2_forward_model,
                                                                                           step2_reverse_model,
                                                                                           devset, args)
            if args.wandb_logging:
                wandb.log({
                    'Loss':tot_loss,
                    'KL_Loss':tot_kl_loss,
                    'Val_Loss':val_tot_loss,
                    'triple precision':triplet_result[0],
                    'triple recall':triplet_result[1],
                    'triple f1':triplet_result[2]
                })

            tot_loss = 0
            tot_kl_loss = 0

            print('Evaluating complete')


            if triplet_result[2] > 0.5:
                
                print("Test results...")
                eval(gcn, Bert, step_1_model, step2_forward_model, step2_reverse_model, testset, args, mode='test')

    logger.info("Features build completed")
    logger.info("Evaluation on testset:")

    eval(gcn, Bert, step_1_model, step2_forward_model, step2_reverse_model, testset, args, mode='test')
    if args.wandb_logging:
        wandb.finish()



def load_with_single_gpu(model_path):
    state_dict = torch.load(model_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    final_state = {}
    for i in state_dict:
        for k, v in state_dict[i].items():
            name = k[7:]
            new_state_dict[name] = v
        final_state[i] = new_state_dict
        new_state_dict = OrderedDict()
    return  final_state

def main():
    parser = argparse.ArgumentParser(description="Train scrip")
    parser.add_argument('--model_dir', type=str, default="savemodel/", help='model path prefix')
    parser.add_argument('--model_to_upload', type=str, default=None)
    parser.add_argument('--add_pos_enc', default=False)
    parser.add_argument('--device', type=str, default="cuda", help='cuda or cpu')
    parser.add_argument("--init_model", default="pretrained_models/bert-base-uncased", type=str, required=False,help="Initial model.")
    parser.add_argument("--init_vocab", default="pretrained_models/bert-base-uncased", type=str, required=False,help="Initial vocab.")

    parser.add_argument("--bert_feature_dim", default=768, type=int, help="feature dim for bert")
    parser.add_argument("--do_lower_case", default=True, action='store_true',help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=100, type=int,help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--drop_out", type=int, default=0.1, help="")
    parser.add_argument("--max_span_length", type=int, default=8, help="")
    parser.add_argument("--embedding_dim4width", type=int, default=200,help="")
    parser.add_argument("--task_learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--multi_gpu", default=False)
    parser.add_argument('--epochs', type=int, default=130, help='training epoch number')
    parser.add_argument("--train_batch_size", default=16, type=int, help="batch size for training")
    parser.add_argument("--RANDOM_SEED", type=int, default=2022, help="")

    # SyMuxEncoder
    parser.add_argument("--pos_num", default=18, type=int)
    parser.add_argument("--pos_dim", default=50, type=int)
    parser.add_argument("--dep_dim", default=50, type=int)
    parser.add_argument("--num_layer", default=1, type=int)
    parser.add_argument("--bert_dropout", default=0.1, type=float)
    parser.add_argument("--output_dropout", default=0.1, type=float)
    parser.add_argument("--w_size", default=1, type=int)
    parser.add_argument("--dep_num", default=20, type=int)
    
    parser.add_argument("--add_pos_mode", default='pos', type=str)
    parser.add_argument("--add_simple_mode", default=False)
    parser.add_argument("--wandb_logging", default=False)
    parser.add_argument("--use_gcn", default=False)
    parser.add_argument("--use_symux", default=False)

    '''修改了数据格式'''
    parser.add_argument("--dataset_path", default="",
                        help="")
    parser.add_argument("--dataset", default="", type=str,
                        help="specify the dataset")
    parser.add_argument('--mode', type=str, default="test", choices=["train", "test"], help='option: train, test')
    '''对相似Span进行attention'''
    # 分词中仅使用结果的首token
    parser.add_argument("--Only_token_head", default=False)
    # Choose the synthesis method of Span
    parser.add_argument('--span_generation', type=str, default="Max", choices=["Start_end", "Max", "Average", "CNN", "ATT"],
                        help='option: CNN, Max, Start_end, Average, ATT, SE_ATT')
    parser.add_argument('--ATT_SPAN_block_num', type=int, default=1, help="number of block in generating spans")

    # Whether to add a separation loss to the relevant span
    parser.add_argument("--kl_loss", default=True)
    parser.add_argument("--kl_loss_weight", type=int, default=0.5, help="weight of the kl_loss")
    parser.add_argument('--kl_loss_mode', type=str, default="KLLoss", choices=["KLLoss", "JSLoss", "EMLoss, CSLoss"],
                        help='选择分离相似Span的分离函数, KL散度、JS散度、欧氏距离以及余弦相似度')
    # Whether to use the filtering algorithm in the test
    parser.add_argument('--Filter_Strategy',  default=True, help='是否使用筛选算法去除冲突三元组')
    # Deprecated    Related Span attention
    parser.add_argument("--related_span_underline", default=False)
    parser.add_argument("--related_span_block_num", type=int, default=1, help="number of block in related span attention")

    # choose Cross Select the number of ATT blocks in Attention
    parser.add_argument("--block_num", type=int, default=1, help="number of block")
    parser.add_argument("--output_path", default='triples.json')
    # Enter and sort in the order of sentences
    parser.add_argument("--order_input", default=True, help="")
    '''Randomize input span sorting'''
    parser.add_argument("--random_shuffle", type=int, default=0, help="")
    # Verify model complexity
    parser.add_argument("--model_para_test", default=False)
    # Use Warm up to converge quickly
    parser.add_argument('--whether_warm_up', default=False)
    parser.add_argument('--warm_up', type=float, default=0.1)
    args = parser.parse_args()

    if args.use_gcn and args.use_symux:
        raise "Don't use GCN and SyMuxEncodre at the same time"

    for k,v in sorted(vars(args).items()):
        logger.info(str(k) + '=' + str(v))
    if args.mode == 'train':
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("keyboard break")
