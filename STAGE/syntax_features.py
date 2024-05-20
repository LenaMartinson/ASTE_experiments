import numpy as np
import torch
from natasha import (
    Segmenter,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    
    Doc
)

pos2id = {'ADJ':1,
 'ADP':2,
 'ADV':3,
 'AUX':4,
 'CCONJ':5,
 'DET':6,
 'INTJ':7,
 'NOUN':8,
 'NUM':9,
 'PART':10,
 'PRON':11,
 'PROPN':12,
 'PUNCT':13,
 'SCONJ':14,
 'VERB':15,
 'X':16,
 'SYM':17,
}


rel2id = {'acl':1,
 'acl:relcl':2,
 'advcl':3,
 'advmod':4,
 'amod':5,
 'appos':6,
 'aux':7,
 'aux:pass':8,
 'case':9,
 'cc':10,
 'ccomp':11,
 'compound':12,
 'conj':13,
 'cop':14,
 'csubj':15,
 'csubj:pass':16,
 'det':17,
 'discourse':18,
 'expl':19,
 'fixed':20,
 'flat':21,
 'flat:foreign':22,
 'flat:name':23,
 'iobj':24,
 'list':25,
 'mark':26,
 'nmod':27,
 'nsubj':28,
 'nsubj:pass':29,
 'nummod':30,
 'nummod:entity':31,
 'nummod:gov':32,
 'obj':33,
 'obl':34,
 'obl:agent':35,
 'orphan':36,
 'parataxis':37,
 'punct':38,
 'root':39,
 'xcomp':40}


segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)


def make_adj_matrix(bert_tokens, tokenizer, max_len, sep_token, add_pos_mode='pos', add_simple_mode=True):
    sent = []
    for i in bert_tokens:
        sent.append(tokenizer.decode([i]))
        if i == sep_token:
            break
    new_sent = ""
    new_inds = [] # from tokens to poses in text
    new_inds_mapping = {}
    index = -1
    for idx, i in enumerate(sent[1:-1]):
        if i[:2] == '##':
            new_sent += i[2:]
        else:
            new_sent += " "
            new_sent += i
            index += 1
        new_inds.append(index)

    new_inds_mapping[0] = [0]
    for idx, i in enumerate(new_inds):
        if new_inds_mapping.get(i + 1):
            new_inds_mapping[i + 1].append(idx + 1)
        else:
            new_inds_mapping[i + 1] = [idx + 1]

    new_sent = new_sent.strip()
    text = new_sent
    
    splitted_text = text.split(" ")
    
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    
    doc_sents_lens = [0]
    for i in doc.sents:
        doc_sents_lens.append(doc_sents_lens[-1] + len(i.tokens))
    
    cnt = 0
    i = 0
    j = 0
    splitted_mapping = {0:[0]} # from poses from the text to segmented words
    while i < len(doc.tokens) and j < len(splitted_text):
        cur_nat_text = doc.tokens[i].text
        cur_our_text = splitted_text[j]
        if cur_nat_text == cur_our_text:
            splitted_mapping[i + 1] = [j + 1]
            i += 1
            j += 1
        else:
            splitted_mapping[i + 1] = [j + 1]
            if len(cur_nat_text) < len(cur_our_text):
                while cur_nat_text != cur_our_text:
                    i += 1
                    splitted_mapping[i + 1] = [j + 1]

                    cur_nat_text += doc.tokens[i].text
            elif len(cur_nat_text) > len(cur_our_text):
                while cur_nat_text != cur_our_text:
                    j += 1
                    splitted_mapping[i + 1].append(j + 1)
                    cur_our_text += splitted_text[j]
            else:
                raise "???"
            i += 1
            j += 1

    adj_matrix = np.zeros((max_len, max_len))
    if add_simple_mode:
        adj_matrix_simple = np.zeros((max_len, max_len))
    pos_tags = [0] * max_len

    cnt_pos = 0
    for i in doc.tokens:
        sent_id, cur_id = [int(j)  for j in i.id.split('_')]
        head_sent_id, head_id = [int(j) for j in i.head_id.split('_')]
        cur_words_ids = []
        cur_words_ids_rel = []
        for j in splitted_mapping[doc_sents_lens[sent_id - 1] + cur_id]:
            for k in new_inds_mapping[j]:
                cur_words_ids.append(k)
                if add_simple_mode:
                    cur_words_ids_rel.append((k, rel2id[i.rel]))
                if add_pos_mode == 'pos':
                    pos_tags[k] = pos2id[i.pos] 
                elif add_pos_mode == 'rel':
                    pos_tags[k] = rel2id[i.rel]
                cnt_pos += 1
        cur_words_head_ids = []
        for j in splitted_mapping[doc_sents_lens[head_sent_id - 1] + head_id]:
            for k in new_inds_mapping[j]:
                cur_words_head_ids.append(k)
        for x in cur_words_ids:
            for j in cur_words_head_ids:
                adj_matrix[x][j] = 1
        if add_simple_mode:
            for x, x_rel in cur_words_ids_rel:
                for j in cur_words_head_ids:
                    adj_matrix_simple[x][j] = x_rel

    pos_tags = torch.tensor(pos_tags)
    if not add_simple_mode:
        adj_matrix_simple = adj_matrix
    return torch.FloatTensor(adj_matrix), torch.FloatTensor(adj_matrix_simple), pos_tags
