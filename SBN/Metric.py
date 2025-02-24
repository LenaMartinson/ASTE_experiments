
from transformers import BertTokenizer
import numpy as np
import json
from tqdm import tqdm


id4validity = {0: 'none', 1: 'valid'}
id4sentiment = {0: 'none', 1: 'positive', 2: 'negative', 3:'neutral', 4:'start'}


class Metric():
    def __init__(self, args, forward_pred_result, reverse_pred_result, gold_instances):
        self.args = args
        self.gold_instances = gold_instances
        self.tokenizer = BertTokenizer.from_pretrained(args.init_vocab, do_lower_case=args.do_lower_case)

        self.pred_aspect = forward_pred_result[0]
        self.pred_aspect_sentiment = forward_pred_result[1]
        self.pred_aspect_sentiment_logit = forward_pred_result[2]

        self.pred_opinion = forward_pred_result[3]
        self.pred_opinion_sentiment_logit = forward_pred_result[4]

        '''Reverse evaluation'''
        self.reverse_pred_opinon = reverse_pred_result[0]
        self.reverse_pred_opinon_sentiment = reverse_pred_result[1]
        self.reverse_pred_opinon_sentiment_logit = reverse_pred_result[2]

        self.reverse_pred_aspect = reverse_pred_result[3]
        self.reverse_pred_aspect_sentiment_logit = reverse_pred_result[4]


    def P_R_F1(self, gold_num, pred_num, correct_num):
        precision = correct_num / pred_num if pred_num > 0 else 0
        recall = correct_num / gold_num if gold_num > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return (precision, recall, f1)

    def num_4_eval(self, gold, pred, gold_num, pred_num, correct_num):
        correct = set(gold) & set(pred)
        gold_num += len(set(gold))
        pred_num += len(set(pred))
        correct_num += len(correct)
        return gold_num, pred_num, correct_num

    def cal_triplet_final_result(self, forward_results, forward_spans, reverse_results, reverse_spans):

        pred_dicts = {}
        pred_spans = forward_spans + reverse_spans
        for index, result in enumerate(forward_results + reverse_results):
            if result in pred_dicts:
                score_dict = pred_dicts[result][2]
                score_new = pred_spans[index][2]
                if score_dict > score_new:
                    continue
                else:
                    pred_dicts[result] = pred_spans[index]
            else:
                pred_dicts[result] = pred_spans[index]
        history = []
        for i in pred_dicts:
            aspect_span_i = range(pred_dicts[i][0][0], pred_dicts[i][0][1])
            opinion_span_i = range(pred_dicts[i][1][0], pred_dicts[i][1][1])
            for j in pred_dicts:
                if (i,j) in history:
                    continue
                history.append((i, j))
                history.append((j, i))
                if i == j:
                    continue
                aspect_span_j = range(pred_dicts[j][0][0], pred_dicts[j][0][1])
                opinion_span_j = range(pred_dicts[j][1][0], pred_dicts[j][1][1])
                repeat_a_span = list(set(aspect_span_i) & set(aspect_span_j))
                repeat_o_span = list(set(opinion_span_i) & set(opinion_span_j))
                if len(repeat_a_span) == 0 or len(repeat_o_span) == 0:
                    continue
                elif len(repeat_a_span) <= min(len(aspect_span_i), len(aspect_span_j)) and \
                        len(repeat_o_span) <= min(len(opinion_span_i), len(opinion_span_j)):
                    i_score = pred_dicts[i][2]
                    j_score = pred_dicts[j][2]
                    if i_score >= j_score:
                        pred_dicts[j] = (pred_dicts[j][0], pred_dicts[j][1], 0)
                    else:
                        pred_dicts[i] = (pred_dicts[i][0], pred_dicts[i][1], 0)
                else:
                    raise(KeyboardInterrupt)
        return [_ for _ in pred_dicts if pred_dicts[_][2] != 0]


    def score_triples(self):
        correct_aspect_num,correct_opinion_num,correct_apce_num,correct_pairs_num,correct_num = 0,0,0,0,0
        gold_aspect_num,gold_opinion_num,gold_apce_num,gold_pairs_num,gold_num = 0,0,0,0,0
        pred_aspect_num,pred_opinion_num,pred_apce_num,pred_pairs_num,pred_num = 0,0,0,0,0

        if self.args.output_path:
            result = []
            aspect_text = []
            opinion_text = []
        print("Ready to for")
        for i in tqdm(range(len(self.gold_instances))):
            '''Entity length experiment'''

            bert_tokens = []
            spans = self.gold_instances[i]['spans']
            start2idx = []
            end2idx = []
            bert_tokens.append(self.tokenizer.cls_token)
            for token in self.gold_instances[i]['tokens']:
                start2idx.append(len(bert_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                if self.args.span_generation == "CNN":
                    bert_tokens.append(sub_tokens[0])
                elif self.args.Only_token_head:
                    bert_tokens.append(sub_tokens[0])
                else:
                    bert_tokens += sub_tokens
                end2idx.append(len(bert_tokens) - 1)
            bert_tokens.append(self.tokenizer.sep_token)
            bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
            gold_aspect, gold_opinion, gold_apce, gold_pairs, gold_triples = self.find_gold_triples(i, bert_spans,
                                                                                                    bert_tokens)
            pred_aspect, pred_opinion, pred_apce, pred_pairs, pred_triples, pred_spans = self.find_pred_triples(i, bert_spans,
                                                                                                    bert_tokens)

            # if len(gold_triples) < 5:
            #     continue

            reverse_aspect, reverse_opinion, reverse_apce, reverse_pairs, reverse_triples, reverse_spans = \
                self.find_pred_reverse_triples(i, bert_spans, bert_tokens)

            pred_aspect = list(set(pred_aspect) | set(reverse_aspect))
            pred_opinion = list(set(pred_opinion) | set(reverse_opinion))
            pred_apce = list(set(pred_apce) | set(reverse_apce))
            pred_pairs = list(set(pred_pairs) | set(reverse_pairs))
            if self.args.Filter_Strategy:
                pred_triples = self.cal_triplet_final_result(pred_triples, pred_spans, reverse_triples, reverse_spans)
            else:
                pred_triples = list(set(pred_triples) | set(reverse_triples))


            if self.args.output_path:
                result.append({"sentence": self.gold_instances[i]['sentence'],
                                     "triple_list_gold": [gold_triple for gold_triple in set(gold_triples)],
                                     "triple_list_pred": [pred_triple for pred_triple in set(pred_triples)],
                                    "new": [new_triple for new_triple in (set(pred_triples) - set(gold_triples))],
                                    "lack": [lack_triple for lack_triple in (set(gold_triples) - set(pred_triples))]
                                     })
                aspect_text.append({"sentence": self.gold_instances[i]['sentence'],
                                    'gold aspect': [gold_as for gold_as in set(gold_aspect)],
                                    'pred aspect': [pred_as for pred_as in set(pred_aspect)],
                                    "new": [new_as for new_as in (set(pred_aspect) - set(gold_aspect))],
                                    "lack": [lack_as for lack_as in (set(gold_aspect) - set(pred_aspect))]})
                opinion_text.append({"sentence": self.gold_instances[i]['sentence'],
                                    'gold aspect': [gold_op for gold_op in set(gold_opinion)],
                                    'pred aspect': [pred_op for pred_op in set(pred_opinion)],
                                    "new": [new_op for new_op in (set(pred_opinion) - set(gold_opinion))],
                                    "lack": [lack_op for lack_op in (set(gold_opinion) - set(pred_opinion))]})


            gold_aspect_num, pred_aspect_num, correct_aspect_num = self.num_4_eval(gold_aspect, pred_aspect,
                                                                                   gold_aspect_num,
                                                                                   pred_aspect_num, correct_aspect_num)

            gold_opinion_num, pred_opinion_num, correct_opinion_num = self.num_4_eval(gold_opinion, pred_opinion,
                                                                                   gold_opinion_num,
                                                                                   pred_opinion_num, correct_opinion_num)

            gold_apce_num, pred_apce_num, correct_apce_num = self.num_4_eval(gold_apce, pred_apce, gold_apce_num,
                                                                             pred_apce_num, correct_apce_num)

            gold_apce_num, pred_apce_num, correct_apce_num = self.num_4_eval(gold_apce, pred_apce, gold_apce_num,
                                                                             pred_apce_num, correct_apce_num)

            gold_pairs_num, pred_pairs_num, correct_pairs_num = self.num_4_eval(gold_pairs, pred_pairs, gold_pairs_num,
                                                                             pred_pairs_num, correct_pairs_num)

            gold_num, pred_num, correct_num = self.num_4_eval(gold_triples, pred_triples, gold_num,
                                                                                pred_num, correct_num)


        if self.args.output_path:
            F = open(self.args.dataset + 'triples.json', 'w', encoding='utf-8')
            json.dump(result, F, ensure_ascii=False, indent=4)
            F.close()

            F1 = open(self.args.dataset + 'aspect.json', 'w', encoding='utf-8')
            json.dump(aspect_text, F1, ensure_ascii=False, indent=4)
            F1.close()

            F2 = open(self.args.dataset + 'opinion.json', 'w', encoding='utf-8')
            json.dump(opinion_text, F2, ensure_ascii=False, indent=4)
            F2.close()


        aspect_result = self.P_R_F1(gold_aspect_num, pred_aspect_num, correct_aspect_num)
        opinion_result = self.P_R_F1(gold_opinion_num, pred_opinion_num, correct_opinion_num)
        apce_result = self.P_R_F1(gold_apce_num, pred_apce_num, correct_apce_num)
        pair_result = self.P_R_F1(gold_pairs_num, pred_pairs_num, correct_pairs_num)
        triplet_result = self.P_R_F1(gold_num, pred_num, correct_num)
        return aspect_result, opinion_result, apce_result, pair_result, triplet_result

    def find_token(self, bert_tokens, span):
        bert_tokens_4_span = bert_tokens[span[1]:span[2]+1]
        sub = ''
        for i, tokens in enumerate(bert_tokens_4_span):
            if i == 0:
                sub = tokens
            elif '##' in tokens:
                sub = sub + tokens.lstrip("##")
            else:
                sub = sub +" "+ tokens
        return sub

    def gold_token(self, tokens):
        sub = ''
        for i, token in enumerate(tokens):
            if i == 0:
                sub = token
            elif '##' in token:
                sub = sub + token.lstrip("##")
            else:
                sub = sub +" "+ token
        return sub

    def find_aspect_sentiment(self, sentence_index, bert_spans, span, aspect_sentiment, aspect_sentiment_logit):
        # span = [span[1], span[2], ]
        bert_span_index = [i for i,x in enumerate(bert_spans) if span[1] == x[0] and span[2] == x[1]]
        assert len(bert_span_index) == 1
        bert_span_index = bert_span_index[0]
        sentiment_index = aspect_sentiment[sentence_index][bert_span_index]
        # sentiment = id4sentiment[aspect_sentiment[sentence_index][bert_span_index]]
        sentiment = id4validity[aspect_sentiment[sentence_index][bert_span_index]]
        sentiment_logit = aspect_sentiment_logit[sentence_index][bert_span_index][sentiment_index]
        # all_sentiment_logit = sum(aspect_sentiment_logit[sentence_index][bert_span_index])
        # sentiment_precent = sentiment_logit / all_sentiment_logit
        # return sentiment, sentiment_precent

        return sentiment, sentiment_logit

    def find_opinion_sentiment(self, sentence_index, opinion_index, bert_spans, span, opinion_sentiment,
                               opinion_sentiment_logit):
        bert_span_index = [i for i, x in enumerate(bert_spans) if span[1] == x[0] and span[2] == x[1]]
        assert len(bert_span_index) == 1
        bert_span_index = bert_span_index[0]
        sentiment_index = opinion_sentiment[sentence_index][opinion_index][bert_span_index]
        sentiment = id4sentiment[opinion_sentiment[sentence_index][opinion_index][bert_span_index]]
        sentiment_logit = opinion_sentiment_logit[sentence_index][opinion_index][bert_span_index][sentiment_index]
        return sentiment, sentiment_logit

    # Code that uses raw data
    def find_gold_triples(self, sentence_index, bert_spans, bert_tokens):
        triples_list,pair_list = [],[]
        aspect_list,opinion_list,apce_list = [],[],[]
        triples = self.gold_instances[sentence_index]['triples']
        for keys in triples:
            aspect, opinion = keys.split('|')
            aspect_tokens = []
            for aspect_token in aspect.split( ):
                token = self.tokenizer.tokenize(aspect_token)
                if self.args.span_generation == "CNN":
                    aspect_tokens.append(token[0])
                elif self.args.Only_token_head:
                    aspect_tokens.append(token[0])
                else:
                    aspect_tokens += token
            new_aspect = self.gold_token(aspect_tokens)

            opinion_tokens = []
            for opinion_token in opinion.split( ):
                token = self.tokenizer.tokenize(opinion_token)
                if self.args.span_generation == "CNN":
                    opinion_tokens.append(token[0])
                elif self.args.Only_token_head:
                    opinion_tokens.append(token[0])
                else:
                    opinion_tokens += token
            new_opinion = self.gold_token(opinion_tokens)

            sentiment = triples[keys][2]

            triples_list.append((new_aspect, new_opinion, sentiment.lower()))

            aspect_list.append((new_aspect))
            opinion_list.append((new_opinion))

            apce_list.append((new_aspect, sentiment))
            pair_list.append((new_aspect, new_opinion))
        return aspect_list, opinion_list, apce_list, pair_list, triples_list

    def find_pred_triples(self, sentence_index, bert_spans, bert_tokens):
        triples_list, pair_list, span_list = [], [], []
        aspect_list, pred_opinion_list, apce_list = [], [], []
        pred_aspect_span = self.pred_aspect[sentence_index]
        # Remove duplicate aspects
        new_aspect_span = []
        for i, pred_aspect in enumerate(pred_aspect_span):
            if len(new_aspect_span) == 0:
                new_aspect_span.append(pred_aspect)
            else:
                if pred_aspect[1] == new_aspect_span[-1][1]:
                    new_aspect_span[-1] = pred_aspect
                else:
                    new_aspect_span.append(pred_aspect)
        for j, pred_aspect in enumerate(new_aspect_span):
            aspect = self.find_token(bert_tokens, pred_aspect)
            aspect_span_output = [pred_aspect[1], pred_aspect[2]+1]
            aspect_sentiment, aspect_sentiment_logit = self.find_aspect_sentiment(sentence_index, bert_spans,
                                                                                  pred_aspect,
                                                                                  self.pred_aspect_sentiment,
                                                                                  self.pred_aspect_sentiment_logit)
            aspect_list.append(aspect)

            opinion_list = []
            for opinion_index in list(np.where(np.array(self.pred_opinion[sentence_index][j]) != 0)[0]):
                opinion_list.append(opinion_index)
            opinion_spans = []
            for opinion_index in opinion_list:
                if opinion_index < len(bert_spans):
                    opinion_spans.append(bert_spans[opinion_index])
                else:
                    continue
            new_opinion_spans = []
            for i, pred_opinion in enumerate(opinion_spans):
                if len(new_opinion_spans) == 0:
                    new_opinion_spans.append(pred_opinion)
                else:
                    if pred_opinion[1] == new_opinion_spans[-1][1]:
                        new_opinion_spans[-1] = pred_opinion
                    else:
                        new_opinion_spans.append(pred_opinion)
            for opinion_span in new_opinion_spans:
                opinion_span = (opinion_span[2], opinion_span[0], opinion_span[1])
                opinion_span_output = [opinion_span[1], opinion_span[2]+1]
                opinion = self.find_token(bert_tokens, opinion_span)
                opinion_sentiment, opinion_sentiment_logit = self.find_opinion_sentiment(sentence_index, j, bert_spans,
                                                                                         opinion_span,
                                                                                         self.pred_opinion,
                                                                                         self.pred_opinion_sentiment_logit)
                # 筛选情感  弃用
                # if opinion_sentiment_logit > aspect_sentiment_logit:
                #     sentiment = opinion_sentiment
                # else:
                #     sentiment = aspect_sentiment

                pred_opinion_list.append(opinion)
                apce_list.append((aspect, opinion_sentiment))
                triples_list.append((aspect, opinion, opinion_sentiment))
                pair_list.append((aspect, opinion))
                span_list.append((aspect_span_output, opinion_span_output, opinion_sentiment_logit))
        return aspect_list, pred_opinion_list, apce_list, pair_list, triples_list, span_list



    def find_pred_reverse_triples(self, sentence_index, bert_spans, bert_tokens):
        triples_list, pair_list, span_list = [], [], []
        opinion_list, pred_aspect_list, apce_list = [], [], []
        pred_opinion_span = self.reverse_pred_opinon[sentence_index]

        new_opinion_span = []
        for i, pred_opinion in enumerate(pred_opinion_span):
            if len(new_opinion_span) == 0:
                new_opinion_span.append(pred_opinion)
            else:
                '''Take the long operation, the overlapping entity takes the longer part'''
                if pred_opinion[1] == new_opinion_span[-1][1]:
                    new_opinion_span[-1] = pred_opinion
                else:
                    new_opinion_span.append(pred_opinion)
        for j, pred_opinion in enumerate(new_opinion_span):
            opinion = self.find_token(bert_tokens, pred_opinion)
            opinion_span_output = [pred_opinion[1], pred_opinion[2] + 1]
            opinion_sentiment, opinion_sentiment_precent = self.find_aspect_sentiment(sentence_index,
                                                                                    bert_spans,
                                                                                    pred_opinion,
                                                                                    self.reverse_pred_opinon_sentiment,
                                                                                    self.reverse_pred_opinon_sentiment_logit)
            opinion_list.append((opinion))
            aspect_list = []
            for aspect_index in list(np.where(np.array(self.reverse_pred_aspect[sentence_index][j]) != 0)[0]):
                aspect_list.append(aspect_index)
            aspect_spans = []
            for aspect_index in aspect_list:
                if aspect_index < len(bert_spans):
                    aspect_spans.append(bert_spans[aspect_index])
                else: continue
            new_aspect_spans = []
            '''At the beginning of the same, choose a longer entity'''
            for i, pred_aspect in enumerate(aspect_spans):
                if len(new_aspect_spans) == 0:
                    new_aspect_spans.append(pred_aspect)
                else:

                    if pred_aspect[1] == new_aspect_spans[-1][1]:
                        new_aspect_spans[-1] = pred_aspect
                    else:
                        new_aspect_spans.append(pred_aspect)
            for aspect_span in new_aspect_spans:
                aspect_span = (aspect_span[2], aspect_span[0], aspect_span[1])
                aspect_span_output = [aspect_span[1], aspect_span[2] + 1]
                aspect = self.find_token(bert_tokens, aspect_span)
                aspect_sentiment, aspect_sentiment_precent = self.find_opinion_sentiment(sentence_index, j,
                                                                                       bert_spans, aspect_span,
                                                                                       self.reverse_pred_aspect,
                                                                                       self.reverse_pred_aspect_sentiment_logit)
                # if opinion_sentiment_precent > aspect_sentiment_precent:
                #     sentiment = opinion_sentiment
                # else:
                #     sentiment = aspect_sentiment
                pred_aspect_list.append((aspect))
                apce_list.append((aspect, aspect_sentiment))
                triples_list.append((aspect, opinion, aspect_sentiment))
                pair_list.append((aspect, opinion))
                span_list.append((aspect_span_output, opinion_span_output, aspect_sentiment_precent))
        return pred_aspect_list, opinion_list, apce_list, pair_list, triples_list, span_list

if __name__ == '__main__':
    test1 = ('boot time', 'fast', 'pos')
    test = ('boot time', 'boot')
    test2 = ('Boot time', 'fast', 'pos')
    set1  = set(test1) & set(test2)
    print(set(test))
