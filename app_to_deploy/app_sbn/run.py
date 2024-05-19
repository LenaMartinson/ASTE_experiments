import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup, BertConfig
from app_sbn.data_BIO_loader import DataTterator, MyDataset
from app_sbn.model import stage_2_features_generation, Step_1, Step_2_forward, Step_2_reverse, Loss
from app_sbn.Metric import Metric
from app_sbn.eval_features import unbatch_data
from transformers.models.bert.modeling_bert import BertEmbeddings
from app_sbn.gcn import GCN
from app_sbn.syntax_features import make_adj_matrix
import streamlit as st


os.environ['CUDA_VISIBLE_DEVICES'] = '5'


sentiment2id = {'none': 0, 'positive': 1, 'negative': 2, 'neutral': 3, 'start': 4}



def eval(gcn_model, bert_model, step_1_model, step_2_forward, step_2_reverse, dataset, args, mode='val'):
    with torch.no_grad():
        gcn_model.eval()
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

            bert_output = bert_model(input_ids=tokens_tensor, attention_mask=attention_mask)
            
            sentence_adj = []
#             max_batch_len = 0
#             for sent in sentence_length:
#                 max_batch_len = max(max_batch_len, len(sent[0]))
            for sent in sentence_length:
                sent_adj, diff_sent_adj, sent_pos_tags = make_adj_matrix(sent[0], 
                                                                         args.max_seq_length, 
                                                                         'pos', 
                                                                         True) 
                sentence_adj.append(diff_sent_adj)
            sentence_adj = torch.cat([i.unsqueeze(0) for i in sentence_adj], axis=0).to_dense().to(args.device)
                        
            h_gcn, _ = gcn_model(sentence_adj, bert_output.last_hidden_state, args.device)
            bert_out = bert_output.last_hidden_state + h_gcn # \hat{h}

            aspect_class_logits, opinion_class_logits, spans_embedding, forward_embedding, reverse_embedding, \
                cnn_spans_mask_tensor = step_1_model(
                    bert_out, attention_mask, bert_spans_tensor, spans_mask_tensor,
                    related_spans_tensor, sentence_length)
            

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
        preds, my_triplets, bert_tokens = metric.score_triples()

    return preds, my_triplets, bert_tokens



class Args:
    def __init__(self):
        self.device = 'cpu'

        self.ATT_SPAN_block_num=1
        self.Filter_Strategy=True
        self.Only_token_head=False
        self.RANDOM_SEED=2022
        self.accumulation_steps=1
        self.add_pos_enc=True
        self.bert_feature_dim=768
        self.block_num=1
        self.do_lower_case=True
        self.drop_out=0.1
        self.embedding_dim4width=200
        self.epochs=25
        self.init_model="ai-forever/ruBert-base"
        self.init_vocab="ai-forever/ruBert-base"
        self.kl_loss=True
        self.kl_loss_mode="KLLoss"
        self.kl_loss_weight=0.5
        self.learning_rate=1e-05
        self.max_seq_length=1408
        self.max_span_length=3
        self.model_para_test=False
        self.model_to_upload=None
        self.multi_gpu=True
        self.order_input=True
        self.random_shuffle=1
        self.related_span_block_num=1
        self.related_span_underline=False
        self.span_generation="Max"
        self.task_learning_rate=0.0001
        self.train_batch_size=4
        self.warm_up=0.1
        self.whether_warm_up=False


@st.cache_resource
def sbn_models_init():

    args = Args()

    gcn_model = GCN(emb_dim=args.bert_feature_dim)

    bert_config = {
    "architectures": [
        "BertForMaskedLM"
    ],
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": None,
    "directionality": "bidi",
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 1536,
    "model_type": "bert",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 0,
    "pooler_fc_size": 768,
    "pooler_num_attention_heads": 12,
    "pooler_num_fc_layers": 3,
    "pooler_size_per_head": 128,
    "pooler_type": "first_token_transform",
    "position_embedding_type": "absolute",
    "transformers_version": "4.20.1",
    "type_vocab_size": 2,
    "use_cache": True,
    "vocab_size": 120138
    }

    print("changing Bert...")

    bert_c = BertConfig(**bert_config)
    
    Bert = BertModel(bert_c)

    Bert = Bert.from_pretrained("lmartinson/sbn_bert")

    print("changing gcn...")

    gcn_model = gcn_model.from_pretrained("lmartinson/sbn_gcn")

    print("changing step_2_forward...")

    step_2_forward = Step_2_forward(
        args.bert_feature_dim, 
        args.block_num, 
        bert_config['hidden_size'], bert_config['layer_norm_eps'], bert_config['hidden_dropout_prob'], 
        bert_config['num_attention_heads'], bert_config['attention_probs_dropout_prob'], 
        bert_config['intermediate_size'], bert_config['hidden_act'])

    step_2_forward = step_2_forward.from_pretrained("lmartinson/sbn_step_2_forward")

    print("changing step_2_reverse...")

    step_2_reverse = Step_2_reverse(
        args.bert_feature_dim, 
        args.block_num, 
        bert_config['hidden_size'], bert_config['layer_norm_eps'], bert_config['hidden_dropout_prob'], 
        bert_config['num_attention_heads'], bert_config['attention_probs_dropout_prob'], 
        bert_config['intermediate_size'], bert_config['hidden_act'])

    step_2_reverse = step_2_reverse.from_pretrained("lmartinson/sbn_step_2_reverse")

    print("changing step_1...")

    step_1 = Step_1(
        args.drop_out, args.max_span_length, args.embedding_dim4width,
        args.bert_feature_dim, args.ATT_SPAN_block_num, args.related_span_underline,
        args.related_span_block_num, args.block_num, 
        args.span_generation,
        bert_config['hidden_size'], bert_config['layer_norm_eps'], bert_config['hidden_dropout_prob'], 
        bert_config['num_attention_heads'], bert_config['attention_probs_dropout_prob'], 
        bert_config['intermediate_size'], bert_config['hidden_act'])

    step_1 = step_1.from_pretrained("lmartinson/sbn_step_1")

    return gcn_model, Bert, step_1, step_2_forward, step_2_reverse


def test(test_data, gcn_model, bert_model, step_1_model, step2_forward_model, step2_reverse_model):

    print("Model loading ended")

    args = Args()

    test_datasets = MyDataset(args, test_data, if_train=False)
    testset = DataTterator(test_datasets, args)

    _, my_triplets, bert_tokens = eval(gcn_model, bert_model, step_1_model, step2_forward_model, step2_reverse_model, testset, args)
    
    return my_triplets[0], bert_tokens[0]



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