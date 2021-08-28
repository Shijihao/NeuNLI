# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import random, os
from logger import Logger
import logging
from time import strftime, time
from torchtext import data
from transformers import BertTokenizer, BertForMaskedLM, RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import spacy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def tmp(text):
    dic = {'A':0, 'B':1, 'C':2, 'D':3}
    return [dic[each] for each in text]

def polarity_split(text):
    return [int(each) for each in text]


def lengths_to_masks(lengths, total_length=None):
    # lengths: LongTensor, (batch)
    # total_length: int|None
    if total_length is None:
        total_length = lengths.max().item()
    masks = torch.arange(total_length, device=lengths.device).expand(lengths.size(0), -1).lt(lengths.view(-1, 1).expand(-1, total_length))
    return masks


class RelationClassifier(nn.Module):
    def __init__(self, roberta_model, encoded_size, num_tags, loss_reduction='mean'):
        super(RelationClassifier, self).__init__()
        self.RoBERTa = RobertaModel.from_pretrained(roberta_model)
        self.classifier = nn.Linear(encoded_size, num_tags)
        self.criterion = nn.CrossEntropyLoss(reduction=loss_reduction)

    def forward(self, tokens_id, attention_mask):
        encoded_layers = self.RoBERTa(input_ids=tokens_id, attention_mask=attention_mask)[0]
        CLS_emb = encoded_layers[:, 0, :]
        output = self.classifier(CLS_emb)
        return output


class Projector(nn.Module):
    """
    project Euclidean to Hyperbolic space
    """
    def __init__(self, args, input_size):
        self.args = args
        super(Projector, self).__init__()
        self.W_in = nn.Linear(input_size, args.hyper_hidden_size, bias=args.bias == 1)
        self.hidden_layers = nn.ModuleList([nn.Linear(args.hyper_hidden_size, args.hyper_hidden_size, bias=args.bias == 1)
                                            for _ in range(args.hyper_hidden_layers)])
        self.W_out = nn.Linear(args.hyper_hidden_size, args.hyper_output_dims, bias=args.bias == 1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=args.projection_dropout)

        self.scaler = nn.Linear(input_size, 1, bias=args.bias == 1)
        self.sigmoid = nn.Sigmoid()

        for layer in [self.W_in, self.W_out] + [l for l in self.hidden_layers]:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, input):
        direction_vectors = self.get_direction_vector(input)
        scalers = self.get_scalers(input)
        return direction_vectors * scalers

    def get_direction_vector(self, input):
        hidden_state = self.dropout(self.relu(self.W_in(input)))
        for layer in self.hidden_layers:
            hidden_state = self.dropout(self.relu(layer(hidden_state)))

        output = self.W_out(hidden_state)  # batch x type_dims
        norms = output.norm(p=2, dim=1, keepdim=True)
        return output.div(norms.expand_as(output))

    def get_scalers(self, input):
        output = self.scaler(input)
        return self.sigmoid(output)

class BertMatch(nn.Module):
    def __init__(self, bert_model, encoded_size, args, loss_reduction='mean'):
        super(BertMatch, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.maskbert = BertForMaskedLM.from_pretrained(bert_model)
        self.criterion = nn.CrossEntropyLoss(reduction=loss_reduction)

        # predict seven lexical relationship
        self.RoBERTa_tokenizer = RobertaTokenizer.from_pretrained(args.RoBERTa_model)
        self.seven_relation = RelationClassifier(args.RoBERTa_model, args.RoBERTa_hidden, 7)
        state_dict = torch.load(args.pretrained_lexrel_RoBERTa, map_location=args.device)
        self.seven_relation.load_state_dict(state_dict)

        self.nonsense_id_ls = [2003, 1999, 1010, 1996, 1997, 2076, 1998, 2019, 1037, 2000, 2013, 2004, 2009, 2065, 2052,
                               2064, 2023, 2008, 2068, 2005, 2083, 2055, 2453, 2002, 2323, 2138, 2028, 2178, 2035, 2006,
                               2084, 2122, 2048, 2089, 2012, 2030, 2044, 2016, 2046, 2011, 2071, 2993, 2184, 2070, 3438,
                               3486, 2698, 2408, 2342, 2144, 2027, 2007, 2119, 2296, 2379, 2321, 2169, 2105, 2053, 2097,
                               2104, 2442, 2648, 1017, 2058, 1019, 2302, 2096, 2114, 2646, 2090, 1018, 1022, 2260, 2385,
                               2274, 2093, 1020, 3209]
        self.batch_size = args.batch_size
        self.encoded_size = args.encoded_size
        self.subword_topk = args.subword_topk
        self.barronKG_num = args.barronKG_num
        self.beam_num = args.beam_num
        self.device = args.device
        self.substitute_iteration = args.substitute_iteration

        # downpolarity
        self.downpolarity_dic = {0:0, 1:1, 2:3, 3:2, 4:5, 5:4, 6:6}
        # Euc2Hyp
        self.projector = Projector(args, args.encoded_size)
        # entailmentscore
        self.hyperdis_prob_classify = nn.Linear(1, 1)
        nn.init.xavier_normal_(self.hyperdis_prob_classify.weight)
        if self.hyperdis_prob_classify.bias is not None:
            nn.init.zeros_(self.hyperdis_prob_classify.bias)

    def forward(self, input_ids_A, input_ids_B, input_ids_C, input_ids_D, A_real_len, B_real_len, C_real_len, D_real_len,
                input_polarity_A, input_polarity_B, input_polarity_C, input_polarity_D,
                input_ids_premise, premise_real_len, gold_labels):
        A_pred_score = self.choice_batch_score(input_ids_A, A_real_len, input_polarity_A, input_ids_premise, premise_real_len)
        B_pred_score = self.choice_batch_score(input_ids_B, B_real_len, input_polarity_B, input_ids_premise, premise_real_len)
        C_pred_score = self.choice_batch_score(input_ids_C, C_real_len, input_polarity_C, input_ids_premise, premise_real_len)
        D_pred_score = self.choice_batch_score(input_ids_D, D_real_len, input_polarity_D, input_ids_premise, premise_real_len)
        hyperbolic_score_output = torch.cat((A_pred_score.permute(1, 0), B_pred_score.permute(1, 0), C_pred_score.permute(1, 0), D_pred_score.permute(1, 0)), dim=1)
        loss_choice = self.criterion(hyperbolic_score_output, gold_labels.squeeze(dim=1))

        return hyperbolic_score_output, loss_choice

    def choice_batch_score(self, input_ids_choice, choice_real_len, input_polarity_A,  input_ids_premise, premise_real_len):
        final_choice_ls = []
        for each_hypo, each_hypo_len, each_hypo_polarity, premises_ls, premise_len_ls in zip(input_ids_choice.chunk(self.batch_size, dim=0),
                                                                                            choice_real_len.chunk(self.batch_size, dim=0),
                                                                                            input_polarity_A.chunk(self.batch_size, dim=0),
                                                                                            input_ids_premise.chunk(self.batch_size, dim=0),
                                                                                            premise_real_len.chunk(self.batch_size, dim=0)):
            premises_ls = premises_ls.squeeze(dim=0)
            premises_len_ls = premise_len_ls.squeeze(dim=0)
            prem_score_ls = []
            for each_prem, each_prem_len in zip(premises_ls.chunk(self.barronKG_num, dim=0), premises_len_ls.chunk(self.barronKG_num, dim=0)):
                output = self.calculate_entail_score(each_hypo, each_hypo_len, each_hypo_polarity, each_prem, each_prem_len)
                output = output.unsqueeze(dim=0).unsqueeze(dim=0)
                prem_score_ls.append(output)
            final_choice_ls.append(max(prem_score_ls))
        return torch.cat(final_choice_ls, dim=1)

    def calculate_entail_score(self, hypo_token, hypo_len, hypo_polarity, pre_token, pre_len):

        # [5]: occupy position, corresponding to [CLS] [SEP]
        hypo_polarity = [5] + hypo_polarity.squeeze(dim=0).tolist() + [5]
        # tensortmp: new sentence after word mutation  idm_tag_ls: word_tag[insertion, deletion, mutation]
        # func{delete_one_word, insert_one_word} hypo_len calculated by stanfordcorenlp
        # substitute_one_word
        tensortmp_mutation, tensortmp_vector_mutation, idm_tag_ls_mutation, polarity_mutation = self.substitute_one_word(hypo_token, hypo_len, hypo_polarity)
        tensortmp_deletion, tensortmp_vector_deletion, idm_tag_ls_deletion, polarity_deletion = self.delete_one_word(hypo_token, hypo_polarity)
        tensortmp_insertion, tensortmp_vector_insertion, idm_tag_ls_insertion, polarity_insertion = self.insert_one_word(hypo_token, hypo_polarity)

        # padding mutation/deletion as long as insertion
        if len(idm_tag_ls_mutation) != 0:
            tensortmp_mutation = F.pad(tensortmp_mutation, (0,1), 'constant', 0)
            tensortmp_vector_mutation = F.pad(tensortmp_vector_mutation, (0,0,0,1), 'constant', 0)
        if len(idm_tag_ls_deletion) != 0:
            tensortmp_deletion = F.pad(tensortmp_deletion, (0,2), 'constant', 0)
            tensortmp_vector_deletion = F.pad(tensortmp_vector_deletion, (0,0,0,2), 'constant', 0)


        # Merge the result of Mutation/Insertion/Deletion
        tensortmp = torch.LongTensor([]).to(device)
        tensortmp_vector = torch.tensor([]).to(device)
        idm_tag_ls = []
        polarity_list = []

        if len(idm_tag_ls_mutation) != 0:
            tensortmp = torch.cat((tensortmp,tensortmp_mutation))
            tensortmp_vector = torch.cat((tensortmp_vector, tensortmp_vector_mutation))
            idm_tag_ls += idm_tag_ls_mutation
            polarity_list += polarity_mutation
        if len(idm_tag_ls_deletion) != 0:
            tensortmp = torch.cat((tensortmp, tensortmp_deletion))
            tensortmp_vector = torch.cat((tensortmp_vector, tensortmp_vector_deletion))
            idm_tag_ls += idm_tag_ls_deletion
            polarity_list += polarity_deletion
        if len(idm_tag_ls_insertion) != 0:
            tensortmp = torch.cat((tensortmp, tensortmp_insertion))
            tensortmp_vector = torch.cat((tensortmp_vector, tensortmp_vector_insertion))
            idm_tag_ls += idm_tag_ls_insertion
            polarity_list += polarity_insertion

        if len(idm_tag_ls) == 0:
            hypo_token_vector = self.maskbert.bert.embeddings.word_embeddings(hypo_token)
            hyperbolic_simscore = self.cal_hyperdistance(hypo_token_vector, hypo_len, pre_token, pre_len)
            rank_score, _ = hyperbolic_simscore.sort()
            rank_top1 = rank_score[-1]
            return rank_top1

        # hyperbolic distance -> simscore
        hyperbolic_simscore = self.cal_MutationInsertionDeletion_hyperdistance(tensortmp_vector, idm_tag_ls, pre_token, pre_len)
        # sort in ascending order
        rank_score, _ = hyperbolic_simscore.sort()
        rank_top1 = rank_score[-1]

        for i in range(self.substitute_iteration):
            if len(hyperbolic_simscore) > self.beam_num:
                _, indices_all = hyperbolic_simscore.sort(descending=True)
                indices = indices_all[:self.beam_num]
                hypo_ls = tensortmp[indices]
                hypo_ls_vector = tensortmp_vector[indices]
                idm_tag_ls_indices = np.array(idm_tag_ls)[np.array(indices.cpu())].tolist()
                tmp_idm_tag_ls = idm_tag_ls_indices.copy()
                polarity_list_indices = np.array(polarity_list)[np.array(indices.cpu())].tolist()
                tmp_polarity_list = polarity_list_indices.copy()
            else:
                hypo_ls = tensortmp
                hypo_ls_vector = tensortmp_vector
                tmp_idm_tag_ls = idm_tag_ls.copy()
                tmp_polarity_list = polarity_list.copy()

            assert hypo_ls.size()[0] == len(tmp_idm_tag_ls)

            # mutation/deletion/insertion
            tmpls_mutation, tmpls_vector_mutation, tag_mutation, polarity_list_mutation = self.MutationInsertionDeletion_nextsub(hypo_ls, hypo_ls_vector, tmp_idm_tag_ls, tmp_polarity_list)
            tmpls_deletion, tmpls_vector_deletion, tag_deletion, polarity_list_deletion = self.MutationInsertionDeletion_nextdeletion(hypo_ls, hypo_ls_vector, tmp_idm_tag_ls, tmp_polarity_list)
            tmpls_insertion, tmpls_vector_insertion, tag_insertion, polarity_list_insertion = self.MutationInsertionDeletion_nextinsertion(hypo_ls, hypo_ls_vector, tmp_idm_tag_ls, tmp_polarity_list)

            # no candidate sentences
            if len(tag_mutation) == 0 and len(tag_insertion) == 0 and len(tag_deletion) == 0:
                break

            # padding mutation/deletion as long as insertion
            if len(tag_mutation) != 0:
                tmpls_mutation = F.pad(tmpls_mutation, (0,1), 'constant', 0)
                tmpls_vector_mutation = F.pad(tmpls_vector_mutation, (0,0,0,1), 'constant', 0)
            if len(tag_deletion) != 0:
                tmpls_deletion = F.pad(tmpls_deletion, (0,2), 'constant', 0)
                tmpls_vector_deletion = F.pad(tmpls_vector_deletion, (0,0,0,2), 'constant', 0)

            # Merge the result of Mutation/Insertion/Deletion
            tensortmp = torch.LongTensor([]).to(device)
            tensortmp_vector = torch.tensor([]).to(device)
            idm_tag_ls = []
            polarity_list = []

            if len(tag_mutation) != 0:
                tensortmp = torch.cat((tensortmp, tmpls_mutation))
                tensortmp_vector = torch.cat((tensortmp_vector, tmpls_vector_mutation))
                idm_tag_ls += tag_mutation
                polarity_list += polarity_list_mutation
            if len(tag_deletion) != 0:
                tensortmp = torch.cat((tensortmp, tmpls_deletion))
                tensortmp_vector = torch.cat((tensortmp_vector, tmpls_vector_deletion))
                idm_tag_ls += tag_deletion
                polarity_list += polarity_list_deletion
            if len(tag_insertion) != 0:
                tensortmp = torch.cat((tensortmp, tmpls_insertion))
                tensortmp_vector = torch.cat((tensortmp_vector, tmpls_vector_insertion))
                idm_tag_ls += tag_insertion
                polarity_list += polarity_list_insertion

            hyperbolic_simscore = self.cal_MutationInsertionDeletion_hyperdistance(tensortmp_vector, idm_tag_ls, pre_token, pre_len)
            if hyperbolic_simscore.max() > rank_top1.max():
                rank_distance, _ = hyperbolic_simscore.sort()
                rank_top1 = rank_distance[-1]
            if i == self.substitute_iteration-1:
                print("进行了{}次替换，达到手动阈值，强制结束！".format(self.substitute_iteration))
        return rank_top1

    def substitute_one_word(self, hypo_tokens, hypo_len, hypo_polarity):
        real_hypo_len = hypo_len - 2
        tmpls = []
        tmpls_vector = []
        idm_tag_ls = []
        output_polarity = []
        
        hypo_tokens = hypo_tokens.squeeze(dim=0)
        for each_pos in range(real_hypo_len):
            idm_tag = [''] + ['idm'] * real_hypo_len.item() + ['']
            # [CLS] + each_pos
            substitute_pos = 1 + each_pos
            origin_word_id = hypo_tokens[substitute_pos]
            # ignore nonsense_id
            if origin_word_id in self.nonsense_id_ls:
                continue
            # 0 -> flat; 3 ->  result of wordpiece 
            substitute_polarity = hypo_polarity[substitute_pos]
            if substitute_polarity == 0:
                continue
            elif substitute_polarity == 3:
                continue
            mask_hypo_tokens = hypo_tokens.clone()
            mask_hypo_tokens[substitute_pos] = token_to_id['[MASK]']

            mask_hypo_tokens_emb = self.maskbert.bert.embeddings.word_embeddings(mask_hypo_tokens)
            # candidate word list
            sub_ls, sub_ls_vector = self.substitute_sent(mask_hypo_tokens.unsqueeze(dim=0), mask_hypo_tokens_emb.unsqueeze(dim=0), hypo_len, substitute_polarity, substitute_pos, origin_word_id)
            for each_sub, each_sub_vector in zip(sub_ls, sub_ls_vector):
                tmp_hypo_tokens = mask_hypo_tokens.clone()
                tmp_hypo_tokens_emb = mask_hypo_tokens_emb.clone()
                tmp_hypo_tokens[substitute_pos] = each_sub
                tmpls.append(tmp_hypo_tokens.unsqueeze(dim=0))
                tmp_hypo_tokens_emb[substitute_pos] = each_sub_vector
                tmpls_vector.append(tmp_hypo_tokens_emb.unsqueeze(dim=0))
                idm_tag[substitute_pos] = 'id'
                idm_tag_ls.append(idm_tag.copy())
                output_polarity.append(hypo_polarity.copy())

        if len(tmpls) == 0:
            return tmpls, tmpls_vector, idm_tag_ls, output_polarity
        tensortmp = torch.cat(tmpls)
        tensortmp_vector = torch.cat(tmpls_vector)

        return tensortmp, tensortmp_vector, idm_tag_ls, output_polarity

    def special_deal_PolarPostag(self, sent, origin_polar_ls, origin_postag_ls):
        sent = ' '.join(sent)
        sent = sent.replace('-LRB-', '(')
        if len(origin_polar_ls) == len(origin_postag_ls):
            return origin_polar_ls, origin_postag_ls
        else:
            polar_ls = []
            postag_ls = []
            sent_ls = sent.split()
            for index, each_word in enumerate(sent_ls):
                each_word_tokenid = tokenizer.encode(each_word, add_special_tokens=False)
                if len(each_word_tokenid) == 1:
                    polar_ls.append(origin_polar_ls[index])
                    postag_ls.append(origin_postag_ls[index])
                else:
                    for _ in range(len(each_word_tokenid)):
                        polar_ls.append('3')
                        postag_ls.append('ignore')
            return polar_ls, postag_ls

    def delete_one_word(self, hypo_tokens, hypo_polarity):
        """
        adj is downward, delete adj eg:“All small cats have legs”⊇“All cats have legs”
        """
        tmpls = []
        tmpls_vector = []
        idm_tag_ls = []
        output_polarity = []

        # filter downward
        if -1 not in hypo_polarity:
            return tmpls, tmpls_vector, idm_tag_ls, output_polarity

        sentstr = tokenizer.decode(hypo_tokens.squeeze(dim=0))
        sentenc = ' '.join(sentstr.split()[1:-1])
        sentenc = sentenc.replace('°f', 'f')
        doc = nlp_spacy(sentenc)
        postag_ls = [token.tag_ for token in doc]
        last_sen_ls = [str(token) for token in doc]
        polar_ls = hypo_polarity[1:-1]

        # filter 'JJ'
        if 'JJ' not in postag_ls:
            return tmpls, tmpls_vector, idm_tag_ls, output_polarity
        polarModify, postagModify = self.special_deal_PolarPostag(last_sen_ls, polar_ls, postag_ls)
        assert len(polarModify) == len(postagModify)
        polarModify = [int(each) for each in polarModify]
        polarModify = [5] + polarModify + [5]
        postagModify = ['cls'] + postagModify + ['sep']
        real_hypo_len = len(polarModify) - 2

        hypo_tokens = hypo_tokens.squeeze(dim=0)
        for idx, _ in enumerate(polarModify):
            idm_tag = [''] + ['idm'] * real_hypo_len + ['']
            if polarModify[idx] == -1 and postagModify[idx] == 'JJ':
                if (idx + 1) < len(idm_tag):
                    # deletion
                    delete_hypo_tokens = torch.cat((hypo_tokens[:idx], hypo_tokens[(idx+1):]))
                    delete_hypo_tokens_emb = self.maskbert.bert.embeddings.word_embeddings(delete_hypo_tokens)
                    tmpls.append(delete_hypo_tokens.unsqueeze(dim=0))
                    tmpls_vector.append(delete_hypo_tokens_emb.unsqueeze(dim=0))
                    # delete idx -> idx+1 can not insert
                    idm_tag[idx+1] = 'dm'
                    tmp_idm_tag = idm_tag[:idx] + idm_tag[(idx+1):]
                    idm_tag_ls.append(tmp_idm_tag.copy())
                    polar_tag = polarModify[:idx] + polarModify[(idx+1):]
                    output_polarity.append(polar_tag.copy())
                # idx + 1 == len(idm_tag)
                else:
                    delete_hypo_tokens = hypo_tokens[:idx]
                    delete_hypo_tokens_emb = self.maskbert.bert.embeddings.word_embeddings(delete_hypo_tokens)
                    tmpls.append(delete_hypo_tokens.unsqueeze(dim=0))
                    tmpls_vector.append(delete_hypo_tokens_emb.unsqueeze(dim=0))
                    tmp_idm_tag = idm_tag[:idx]
                    idm_tag_ls.append(tmp_idm_tag.copy())
                    polar_tag = polarModify[:idx]
                    output_polarity.append(polar_tag.copy())
        if len(tmpls) == 0:
            return tmpls, tmpls_vector, idm_tag_ls, output_polarity
        tensortmp = torch.cat(tmpls)
        tensortmp_vector = torch.cat(tmpls_vector)
        return tensortmp, tensortmp_vector, idm_tag_ls, output_polarity

    def insert_one_word(self, hypo_tokens, hypo_polarity):
        #noun upward, insert adj. eg：“cat eats animals”⊇ “cat eats small animals”
        tmpls = []
        tmpls_vector = []
        idm_tag_ls = []
        output_polarity = []

        sentstr = tokenizer.decode(hypo_tokens.squeeze(dim=0))
        sentenc = ' '.join(sentstr.split()[1:-1])
        sentenc = sentenc.replace('°f', 'f')

        doc = nlp_spacy(sentenc)
        postag_ls = [token.tag_ for token in doc]
        last_sen_ls = [str(token) for token in doc]
        polar_ls = hypo_polarity[1:-1]
        polarModify, postagModify = self.special_deal_PolarPostag(last_sen_ls, polar_ls, postag_ls)
        assert len(polarModify) == len(postagModify)
        polarModify = [int(each) for each in polarModify]
        polarModify = [5] + polarModify + [5]
        postagModify = ['cls'] + postagModify + ['sep']
        real_hypo_len = len(polarModify) - 2

        hypo_tokens = hypo_tokens.squeeze(dim=0)
        for idx, _ in enumerate(polarModify):
            if polarModify[idx] == 1 and (postagModify[idx] == "NN" or postagModify[idx] == "NNS"):
                insert_hypo_tokens = torch.cat((hypo_tokens[:idx], torch.tensor(token_to_id['[MASK]']).to(device).unsqueeze(dim=0), hypo_tokens[idx:]))
                insert_hypo_tokens_emb = self.maskbert.bert.embeddings.word_embeddings(insert_hypo_tokens)
                idm_tag = [''] + ['idm'] * (real_hypo_len+1) + ['']
                polar_tag = polarModify[:idx] + [1] + polarModify[idx:]
                insert_ls, insert_ls_vector = self.insert_sent(insert_hypo_tokens.unsqueeze(dim=0), insert_hypo_tokens_emb.unsqueeze(dim=0), len(idm_tag), idx)
                for each_insert, each_insert_vector in zip(insert_ls, insert_ls_vector):
                    tmp_hypo_tokens = insert_hypo_tokens.clone()
                    tmp_hypo_tokens_emb = insert_hypo_tokens_emb.clone()
                    tmp_hypo_tokens[idx] = each_insert
                    tmpls.append(tmp_hypo_tokens.unsqueeze(dim=0))
                    tmp_hypo_tokens_emb[idx] = each_insert_vector
                    tmpls_vector.append(tmp_hypo_tokens_emb.unsqueeze(dim=0))
                    # new insert can not delete
                    idm_tag[idx] = 'im'
                    idm_tag[idx+1] = 'md'
                    idm_tag_ls.append(idm_tag.copy())
                    output_polarity.append(polar_tag)
        if len(idm_tag_ls) == 0:
            return tmpls, tmpls_vector, idm_tag_ls, output_polarity
        tensortmp = torch.cat(tmpls)
        tensortmp_vector = torch.cat(tmpls_vector)
        return tensortmp, tensortmp_vector, idm_tag_ls, output_polarity

    def substitute_sent(self, hypo_token, hypo_token_emb, hypo_len, substitute_polarity, sub_pos, origin_word_id):
        first_sen = lengths_to_masks(hypo_len, total_length=hypo_token.size(-1))
        token_type_ids = first_sen.eq(torch.zeros_like(hypo_token, device=hypo_token.device).bool()).long()
        attention_mask = first_sen.long()
        prediction_scores = \
                self.maskbert(token_type_ids=token_type_ids, attention_mask=attention_mask, inputs_embeds=hypo_token_emb)[0]
        # prediction_scores.size()
        # (values, indices)
        predicted_distribution = prediction_scores[0, sub_pos]
        # original word -> -inf
        predicted_distribution[origin_word_id] = -float('inf')
        tmp_candidate_word = []
        tmp_candidate_vector = []
        for i in range(self.subword_topk):
            one_hot_vector = F.gumbel_softmax(predicted_distribution, tau=1, hard=True)
            word_id = torch.sum(torch.arange(30522, dtype=torch.float).to(device) * one_hot_vector).long()
            if word_id.unsqueeze(dim=0) in tmp_candidate_word:
                continue
            tmp_candidate_word.append(word_id.unsqueeze(dim=0))
            # differentiable
            word_emb = torch.sum(self.maskbert.bert.embeddings.word_embeddings.weight * one_hot_vector.unsqueeze(dim=1).repeat(1, self.encoded_size),dim=0)
            tmp_candidate_vector.append(word_emb.unsqueeze(dim=0))

        tmp_candidate_word_id = torch.cat(tmp_candidate_word)
        tmp_candidate_vector = torch.cat(tmp_candidate_vector)
        """
        predict lexical relation
        """
        origin_word = self.tokenizer.convert_ids_to_tokens(origin_word_id.unsqueeze(dim=0))
        tmp_candidate_word = self.tokenizer.convert_ids_to_tokens(tmp_candidate_word_id)
        lexical_pair = []
        max_len = 0
        for each_candidate in tmp_candidate_word:
            whole_sentence = origin_word[0] + ' </s> ' + each_candidate
            pair_ids = self.RoBERTa_tokenizer.encode(whole_sentence)
            lexical_pair.append(pair_ids)
            if len(pair_ids) > max_len:
                max_len = len(pair_ids)

        lexical_attention = []
        for each in lexical_pair:
            msk_pad = [1] * len(each) + [0] * (max_len - len(each))
            lexical_attention.append(msk_pad)

        for each in lexical_pair:
            # RoBERTa: <pad> -> 1
            while len(each) < max_len:
                each.append(1)

        lexical_pair = torch.LongTensor(lexical_pair).to(device)
        lexical_attention = torch.LongTensor(lexical_attention).to(device)
        output = self.seven_relation(lexical_pair, lexical_attention)
        output = F.softmax(output, dim=1)
        relation_indice = output.max(dim=1)[1].tolist()
        # downward
        if substitute_polarity == -1:
            relation_indice = [self.downpolarity_dic[i] for i in relation_indice]

        final_tokens = []
        final_tokens_vectors = []
        for i_relation, j_candidate, vector_candidate in zip(relation_indice, tmp_candidate_word_id, tmp_candidate_vector):
            if i_relation == 1:
                final_tokens.append(j_candidate.unsqueeze(dim=0))
                final_tokens_vectors.append(vector_candidate.unsqueeze(dim=0))
            elif i_relation == 3:
                final_tokens.append(j_candidate.unsqueeze(dim=0))
                final_tokens_vectors.append(vector_candidate.unsqueeze(dim=0))

        if len(final_tokens) == 0:
            return final_tokens, final_tokens_vectors
        final_tokens = torch.cat(final_tokens)
        final_tokens_vectors = torch.cat(final_tokens_vectors)

        return final_tokens, final_tokens_vectors

    def insert_sent(self, hypo_token, hypo_token_emb, hypo_len, insert_pos):
        hypo_len = torch.LongTensor([hypo_len]).to(device)
        first_sen = lengths_to_masks(hypo_len, total_length=hypo_token.size(-1))
        token_type_ids = first_sen.eq(torch.zeros_like(hypo_token, device=hypo_token.device).bool()).long()
        attention_mask = first_sen.long()
        prediction_scores = self.maskbert(token_type_ids=token_type_ids, attention_mask=attention_mask, inputs_embeds=hypo_token_emb)[0]
        predicted_distribution = prediction_scores[0, insert_pos]
        tmp_candidate_word = []
        tmp_candidate_vector = []
        for i in range(self.subword_topk):
            one_hot_vector = F.gumbel_softmax(predicted_distribution, tau=1, hard=True)
            word_id = torch.sum(torch.arange(30522, dtype=torch.float).to(device) * one_hot_vector).long()
            if word_id.unsqueeze(dim=0) in tmp_candidate_word:
                continue
            # 3685 -> cannot
            if word_id.item() in [18728, 3685]:
                continue
            wordstr = tokenizer.decode(word_id.unsqueeze(dim=0))
            word_postag = [token.tag_ for token in nlp_spacy(wordstr)][0]
            
            if word_postag == 'JJ':
                tmp_candidate_word.append(word_id.unsqueeze(dim=0))
                word_emb = torch.sum(
                    self.maskbert.bert.embeddings.word_embeddings.weight * one_hot_vector.unsqueeze(dim=1).repeat(1, self.encoded_size), dim=0)
                tmp_candidate_vector.append(word_emb.unsqueeze(dim=0))
        if len(tmp_candidate_word) == 0:
            return tmp_candidate_word, tmp_candidate_vector
        tmp_candidate_word_id = torch.cat(tmp_candidate_word)
        tmp_candidate_vector = torch.cat(tmp_candidate_vector)
        
        return tmp_candidate_word_id, tmp_candidate_vector

    def poincare_distance(self, u, v):
        """
        From: https://github.com/facebookresearch/poincare-embeddings/blob/master/model.py#L48
        """
        boundary = 1 - 1e-5
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, boundary)
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, boundary)
        sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    def cal_MutationInsertionDeletion_hyperdistance(self, hypo_token, idm_tag_ls, pre_token, pre_len):
        simscore_ls = []
        for each_hypo_token, each_idm_tag_ls in zip(hypo_token, idm_tag_ls):
            each_simscore = self.cal_hyperdistance(each_hypo_token.unsqueeze(dim=0), torch.LongTensor([len(each_idm_tag_ls)]).to(device), pre_token, pre_len)
            simscore_ls.append(each_simscore)
        return torch.cat(simscore_ls)

    def cal_hyperdistance(self, hypo_token, hypo_len, pre_token, pre_len):
        # [CLS] pre [SEP]
        pre = pre_token[0, :pre_len]
        # hyp [SEP]
        hyp = hypo_token[:, 1:hypo_len]
        pre = self.maskbert.bert.embeddings.word_embeddings(pre.expand(hyp.size()[0], -1))
        pre_hyp = torch.cat((pre, hyp), dim=1)

        first_sen = lengths_to_masks(pre_len, total_length=pre_hyp.size()[1])
        token_type_ids = first_sen.eq(torch.zeros_like(first_sen, device=first_sen.device).bool()).long()
        token_type_ids = token_type_ids.expand(pre_hyp.size()[0], -1)
        hidden = self.maskbert.bert(inputs_embeds=pre_hyp, token_type_ids=token_type_ids)[0]
        CLS_emb = hidden[:, 0, :]
        SEP_emb = hidden[:, (pre_len-1).item(), :]

        CLS_hyperbolic_vector = self.projector(CLS_emb)
        SEP_hyperbolic_vector = self.projector(SEP_emb)
        output = self.poincare_distance(CLS_hyperbolic_vector, SEP_hyperbolic_vector)

        sim_score = self.hyperdis_prob_classify(output.unsqueeze(dim=1))
        sim_score = sim_score.squeeze(dim=1)

        return sim_score

    def MutationInsertionDeletion_nextsub(self, hypo_ls, hypo_ls_vector, tmp_idm_tag_ls, tmp_polarity_list):
        output_hypo_ls = []
        output_hypo_vector_ls = []
        output_idm_tag_ls = []
        output_polarity_ls = []
        for each_hypo_ls, each_hypo_ls_vector, each_tmp, each_polarity in zip(hypo_ls, hypo_ls_vector, tmp_idm_tag_ls, tmp_polarity_list):
            each_mutation, each_vector_mutation, each_tag_ls_mutation, each_polarity_list_mutation = self.nextsub(each_hypo_ls, each_hypo_ls_vector, each_tmp, each_polarity)
            if len(each_tag_ls_mutation) == 0:
                continue
            output_hypo_ls.append(each_mutation)
            output_hypo_vector_ls.append(each_vector_mutation)
            output_idm_tag_ls += each_tag_ls_mutation
            output_polarity_ls += each_polarity_list_mutation
        if len(output_idm_tag_ls) == 0:
            return output_hypo_ls, output_hypo_vector_ls, output_idm_tag_ls, output_polarity_ls
        else:
            return torch.cat(output_hypo_ls), torch.cat(output_hypo_vector_ls), output_idm_tag_ls, output_polarity_ls

    def nextsub(self, hypo_padding, hypo_padding_vector, idm_each, polarity_each):
        tmp_idm_each = idm_each.copy()
        padding_num = len(hypo_padding) - len(tmp_idm_each)
        hypo_each = hypo_padding[:len(tmp_idm_each)]
        hypo_each_vector = hypo_padding_vector[:len(tmp_idm_each)]

        tmpls = []
        tmpls_vector = []
        idmls = []
        polarityls = []

        hypo_polarity = polarity_each
        hypo_len = len(hypo_polarity)
        real_hypo_len = hypo_len - 2
        for each_pos in range(real_hypo_len):
            substitute_pos = 1 + each_pos
            if 'm' not in tmp_idm_each[substitute_pos]:
                continue
            origin_word_id = hypo_each[substitute_pos]
            if origin_word_id in self.nonsense_id_ls:
                continue
            substitute_polarity = hypo_polarity[substitute_pos]
            if substitute_polarity == 0:
                continue
            elif substitute_polarity == 3:
                continue
            mask_hypo_tokens = hypo_each.clone()
            mask_hypo_tokens_vector = hypo_each_vector.clone()
            mask_hypo_tokens[substitute_pos] = token_to_id['[MASK]']
            mask_hypo_tokens_vector[substitute_pos] = self.maskbert.bert.embeddings.word_embeddings(torch.tensor(token_to_id['[MASK]']).to(device)).unsqueeze(dim=0)
            sub_ls, sub_ls_vector = self.substitute_sent(mask_hypo_tokens.unsqueeze(dim=0), mask_hypo_tokens_vector.unsqueeze(dim=0), torch.LongTensor([hypo_len]).to(device), substitute_polarity, substitute_pos, origin_word_id)
            
            if len(sub_ls) == 0:
                continue
            for each_sub, each_sub_vector in zip(sub_ls, sub_ls_vector):
                each_hypo_tokens = mask_hypo_tokens.clone()
                each_hypo_tokens[substitute_pos] = each_sub.unsqueeze(dim=0)
                
                tmpls.append(F.pad(each_hypo_tokens.unsqueeze(dim=0), (0, padding_num), 'constant', 0))
                each_hypo_tokens_vector = mask_hypo_tokens_vector.clone()
                each_hypo_tokens_vector[substitute_pos] = each_sub_vector.unsqueeze(dim=0)
                
                tmpls_vector.append(F.pad(each_hypo_tokens_vector.unsqueeze(dim=0), (0, 0, 0, padding_num), 'constant', 0))
                tmp_idm_each[substitute_pos] = 'id'
                idmls.append(tmp_idm_each.copy())
                polarityls.append(hypo_polarity.copy())

        if len(idmls) == 0:
            return tmpls, tmpls_vector, idmls, polarityls
        else:
            tensortmp = torch.cat(tmpls)
            tensortmp_vector = torch.cat(tmpls_vector)
            return tensortmp, tensortmp_vector, idmls, polarityls

    def MutationInsertionDeletion_nextdeletion(self, hypo_ls, hypo_ls_vector, tmp_idm_tag_ls, tmp_polarity_list):
        output_hypo_ls = []
        output_hypo_vector_ls = []
        output_idm_tag_ls = []
        output_polarity_ls = []
        for each_hypo_ls, each_hypo_ls_vector, each_tmp, each_polarity in zip(hypo_ls, hypo_ls_vector, tmp_idm_tag_ls, tmp_polarity_list):
            try:
                each_deletion, each_vector_deletion, each_tag_ls_deletion, each_polarity_list_deletion = self.nextdelete(each_hypo_ls, each_hypo_ls_vector, each_tmp, each_polarity)
            except:
                continue
            if len(each_tag_ls_deletion) == 0:
                continue
            output_hypo_ls.append(each_deletion)
            output_hypo_vector_ls.append(each_vector_deletion)
            output_idm_tag_ls += each_tag_ls_deletion
            output_polarity_ls += each_polarity_list_deletion
        if len(output_idm_tag_ls) == 0:
            return output_hypo_ls, output_hypo_vector_ls, output_idm_tag_ls, output_polarity_ls
        else:
            return torch.cat(output_hypo_ls), torch.cat(output_hypo_vector_ls), output_idm_tag_ls, output_polarity_ls

    def nextdelete(self, hypo_padding, hypo_padding_vector, idm_each, polarity_each):
        tmp_idm_each = idm_each.copy()
        padding_num = len(hypo_padding) - len(tmp_idm_each)

        hypo_each = hypo_padding[:len(tmp_idm_each)]
        hypo_each_vector = hypo_padding_vector[:len(tmp_idm_each)]
        tmpls = []
        tmpls_vector = []
        idmls = []
        polarityls = []

        sentstr = tokenizer.decode(hypo_each)
        sentenc = ' '.join(sentstr.split()[1:-1])
        sentenc = sentenc.replace('°f', 'f')
        doc = nlp_spacy(sentenc)
        postag_ls = [token.tag_ for token in doc]
        last_sen_ls = [str(token) for token in doc]
        polar_ls = polarity_each[1:-1]

        if -1 not in polar_ls:
            return tmpls, tmpls_vector, idmls, polarityls
        if 'JJ' not in postag_ls:
            return tmpls, tmpls_vector, idmls, polarityls
        polarModify, postagModify = self.special_deal_PolarPostag(last_sen_ls, polar_ls, postag_ls)
        assert len(polarModify) == len(postagModify)
        polarModify = [int(each) for each in polarModify]
        polarModify = [5] + polarModify + [5]
        postagModify = ['cls'] + postagModify + ['sep']
        assert len(polarModify) == len(tmp_idm_each)
        hypo_len = len(tmp_idm_each) - 2
        for each_pos in range(hypo_len):
            substitute_pos = 1 + each_pos
            if polarModify[substitute_pos] == -1 and postagModify[substitute_pos] == 'JJ' and 'd' in tmp_idm_each[substitute_pos]:
                if substitute_pos <= hypo_len:
                    delete_hypo_tokens = torch.cat((hypo_each[:substitute_pos], hypo_each[(substitute_pos+1):]))
                    delete_hypo_tokens_emb = torch.cat((hypo_each_vector[:substitute_pos], hypo_each_vector[(substitute_pos+1):]))
                    tmpls.append(F.pad(delete_hypo_tokens.unsqueeze(dim=0), (0, padding_num), 'constant', 0))
                    tmpls_vector.append(F.pad(delete_hypo_tokens_emb.unsqueeze(dim=0), (0, 0, 0, padding_num), 'constant', 0))
                    idm_each[substitute_pos+1] = 'dm'
                    tmp_idm_each = idm_each[:substitute_pos] + idm_each[(substitute_pos+1):]
                    idmls.append(tmp_idm_each.copy())
                    tmp_polarity_each = polarModify[:substitute_pos] + polarModify[(substitute_pos+1):]
                    polarityls.append(tmp_polarity_each.copy())

        if len(idmls) == 0:
            return tmpls, tmpls_vector, idmls, polarityls
        else:
            tensortmp = torch.cat(tmpls)
            tensortmp_vector = torch.cat(tmpls_vector)
            return tensortmp, tensortmp_vector, idmls, polarityls

    def MutationInsertionDeletion_nextinsertion(self, hypo_ls, hypo_ls_vector, tmp_idm_tag_ls, tmp_polarity_list):
        output_hypo_ls = []
        output_hypo_vector_ls = []
        output_idm_tag_ls = []
        output_polarity_ls = []
        for each_hypo_ls, each_hypo_ls_vector, each_tmp, each_polarity in zip(hypo_ls, hypo_ls_vector, tmp_idm_tag_ls, tmp_polarity_list):
            try:
                each_insertion, each_vector_insertion, each_tag_ls_insertion, each_polarity_list_insertion = self.nextinsert(each_hypo_ls, each_hypo_ls_vector, each_tmp, each_polarity)
            except:
                continue
            if len(each_tag_ls_insertion) == 0:
                continue
            output_hypo_ls.append(each_insertion)
            output_hypo_vector_ls.append(each_vector_insertion)
            output_idm_tag_ls += each_tag_ls_insertion
            output_polarity_ls += each_polarity_list_insertion
        if len(output_idm_tag_ls) == 0:
            return output_hypo_ls, output_hypo_vector_ls, output_idm_tag_ls, output_polarity_ls
        else:
            return torch.cat(output_hypo_ls), torch.cat(output_hypo_vector_ls), output_idm_tag_ls, output_polarity_ls

    def nextinsert(self, hypo_padding, hypo_padding_vector, idm_each, polarity_each):
        tmp_idm_each = idm_each.copy()
        padding_num = len(hypo_padding) - len(tmp_idm_each)

        hypo_each = hypo_padding[:len(tmp_idm_each)]
        hypo_each_vector = hypo_padding_vector[:len(tmp_idm_each)]

        tmpls = []
        tmpls_vector = []
        idmls = []
        output_polarity = []

        sentstr = tokenizer.decode(hypo_each)
        sentenc = ' '.join(sentstr.split()[1:-1])
        sentenc = sentenc.replace('°f', 'f')
        doc = nlp_spacy(sentenc)
        postag_ls = [token.tag_ for token in doc]
        last_sen_ls = [str(token) for token in doc]
        polar_ls = polarity_each[1:-1]

        polarModify, postagModify = self.special_deal_PolarPostag(last_sen_ls, polar_ls, postag_ls)
        polarModify = [int(each) for each in polarModify]
        polarModify = [5] + polarModify + [5]
        postagModify = ['cls'] + postagModify + ['sep']
        assert len(polarModify) == len(tmp_idm_each)
        hypo_polarity = polarModify
        hypo_len = len(hypo_polarity)
        real_hypo_len = hypo_len - 2

        for each_pos in range(real_hypo_len):
            substitute_pos = 1 + each_pos
            if 'i' in tmp_idm_each[substitute_pos] and (postagModify[substitute_pos] == 'NN' or postagModify[substitute_pos] == 'NNS') and polarModify[substitute_pos] == 1:
                insert_hypo_tokens = torch.cat((hypo_each[:substitute_pos], torch.tensor(token_to_id['[MASK]']).to(device).unsqueeze(dim=0), hypo_each[substitute_pos:]))
                mask_emb = self.maskbert.bert.embeddings.word_embeddings(torch.tensor(token_to_id['[MASK]']).to(device)).unsqueeze(dim=0)
                insert_hypo_tokens_emb = torch.cat((hypo_each_vector[:substitute_pos], mask_emb, hypo_each_vector[substitute_pos:]))
                new_tmp_idm_each = tmp_idm_each.copy()
                new_tmp_idm_each = new_tmp_idm_each[:substitute_pos] + ['im'] + new_tmp_idm_each[substitute_pos:]
                new_tmp_polarity_each = hypo_polarity.copy()
                new_tmp_polarity_each = new_tmp_polarity_each[:substitute_pos] + [1] + new_tmp_polarity_each[substitute_pos:]
                insert_ls, insert_ls_vector = self.insert_sent(insert_hypo_tokens.unsqueeze(dim=0), insert_hypo_tokens_emb.unsqueeze(dim=0), len(new_tmp_idm_each), substitute_pos)
                for each_insert, each_insert_vector in zip(insert_ls, insert_ls_vector):
                    tmp_hypo_tokens = insert_hypo_tokens.clone()
                    tmp_hypo_tokens_emb = insert_hypo_tokens_emb.clone()
                    tmp_hypo_tokens[substitute_pos] = each_insert
                    tmp_hypo_tokens_emb[substitute_pos] = each_insert_vector
                    tmpls.append(F.pad(tmp_hypo_tokens.unsqueeze(dim=0), (0, padding_num), 'constant', 0))
                    tmpls_vector.append(F.pad(tmp_hypo_tokens_emb.unsqueeze(dim=0), (0, 0, 0, padding_num), 'constant', 0))
                   
                    new_tmp_idm_each[substitute_pos] = 'im'
                    new_tmp_idm_each[substitute_pos + 1] = 'md'
                    idmls.append(new_tmp_idm_each.copy())
                    output_polarity.append(new_tmp_polarity_each.copy())
        if len(idmls) == 0:
            return tmpls, tmpls_vector, idmls, output_polarity
        else:
            tensortmp = torch.cat(tmpls)
            tensortmp_vector = torch.cat(tmpls_vector)
            return tensortmp, tensortmp_vector, idmls, output_polarity


def main():
    # use BERT vocab, use_vocab -> False
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=tmp)
    pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    TEXT = data.Field(sequential=True, use_vocab=False, batch_first=True, tokenize=tokenizer.encode, pad_token=pad_index, include_lengths=True)
    TEXT_POLARITY = data.Field(sequential=True, use_vocab=False, batch_first=True, preprocessing=polarity_split, pad_token=5, include_lengths=True)
    Candidate_TEXT = data.Field(sequential=True, use_vocab=False, batch_first=True, tokenize=tokenizer.encode,
                      pad_token=pad_index)
    CANDIDATES = data.NestedField(Candidate_TEXT, tokenize= lambda s: s.split('<split_each>'), include_lengths=True)

    train, dev, test = data.TabularDataset.splits(path= args.dataset_path, train=args.train_dataset, validation=args.dev_dataset, test=args.test_dataset, format='json',
                              fields={
                                      "answer":("answer",LABEL),
                                      "A":("A", TEXT),
                                      "polarity_A":("polarity_A", TEXT_POLARITY),
                                      "B":("B", TEXT),
                                      "polarity_B":("polarity_B", TEXT_POLARITY),
                                      "C":("C", TEXT),
                                      "polarity_C":("polarity_C", TEXT_POLARITY),
                                      "D":("D", TEXT),
                                      "polarity_D":("polarity_D", TEXT_POLARITY),
                                      "candidates":("candidates", CANDIDATES)
                                      })

    # use BERT tokenize, no need build_vocab
    train_iter, dev_iter = data.BucketIterator.splits((train, dev), batch_size=args.batch_size, sort_key=lambda x: len(x.A), sort_within_batch=False, shuffle=True, device=args.device)
    test_iter = data.Iterator(dataset=test, batch_size=args.batch_size, train=False, sort=False, shuffle=False, device=args.device)

    model = BertMatch(args.bert_model, args.encoded_size, args).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_epsilon)

    t_total = len(train_iter) // args.accumulation_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )


    if args.device == "cuda":
        model = nn.DataParallel(model)
    logger.cache_in(str(model), to_print=False)

    best_accuracy = 0

    for epoch in range(args.epochs):
        epoch += 1
        logger.cache_in('Epoch {}...'.format(epoch))
        
        model.train()
        correct = 0.0
        t0 = time()
        pred_ls = []
        epoch_loss = 0.0
        for step_i, batch in enumerate(tqdm(train_iter)):
            gold_labels = batch.answer
            input_ids_A, A_real_len = batch.A
            input_polarity_A, _ = batch.polarity_A
            input_ids_B, B_real_len = batch.B
            input_polarity_B, _ = batch.polarity_B
            input_ids_C, C_real_len = batch.C
            input_polarity_C, _ = batch.polarity_C
            input_ids_D, D_real_len = batch.D
            input_polarity_D, _ = batch.polarity_D
            input_ids_premise, _, premise_real_len = batch.candidates
            output, loss = model(input_ids_A, input_ids_B, input_ids_C, input_ids_D,
                                 A_real_len, B_real_len, C_real_len, D_real_len,
                                 input_polarity_A, input_polarity_B, input_polarity_C, input_polarity_D,
                                 input_ids_premise, premise_real_len, gold_labels)
            loss = loss.mean()
            epoch_loss += loss.item()

            loss = loss / args.accumulation_steps
            loss.backward()

            if (step_i+1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            _, pred_idx = output.max(dim=1)
            pred_ls.append(pred_idx)
            equal = torch.sum(torch.eq(pred_idx, gold_labels.squeeze()).int()).item()
            correct += equal
            torch.cuda.empty_cache()
        total = len(torch.cat(pred_ls))
        acc = correct / total
        logger.cache_in("Epoch_Loss: {}".format(epoch_loss / total))
        logger.cache_in("{} Accuarcy: {:.2f}%, correct_num: {},  total_num: {} , {:.2f} Seconds Used:".format('Train', acc * 100.0, correct, total, time() - t0))
        logger.cache_write()

        with torch.no_grad():
            model.eval()
            correct_num = 0.0
            pred_ls = []
            t0 = time()
            for _, batch in enumerate(tqdm(dev_iter)):
                gold_labels = batch.answer
                input_ids_A, A_real_len = batch.A
                input_polarity_A, _ = batch.polarity_A
                input_ids_B, B_real_len = batch.B
                input_polarity_B, _ = batch.polarity_B
                input_ids_C, C_real_len = batch.C
                input_polarity_C, _ = batch.polarity_C
                input_ids_D, D_real_len = batch.D
                input_polarity_D, _ = batch.polarity_D
                input_ids_premise, _, premise_real_len = batch.candidates
                with torch.no_grad():
                    output, loss = model(input_ids_A, input_ids_B, input_ids_C, input_ids_D,
                                         A_real_len, B_real_len, C_real_len, D_real_len,
                                         input_polarity_A, input_polarity_B, input_polarity_C, input_polarity_D,
                                         input_ids_premise, premise_real_len, gold_labels)
                _, pred_idx = output.max(dim=1)
                pred_ls.append(pred_idx)
                equal = torch.sum(torch.eq(pred_idx, gold_labels.squeeze()).int()).item()
                correct_num += equal
            total_num = len(torch.cat(pred_ls))
            dev_acc = correct_num / total_num
            logger.cache_in(
                "{} Accuarcy: {:.2f}%, correct_num: {},  total_num: {} , {:.2f} Seconds Used:".format('Dev',
                                                                                                      dev_acc * 100.0,
                                                                                                      correct_num, total_num,
                                                                                                      time() - t0))
            logger.cache_write()
            if dev_acc > best_accuracy:
                best_accuracy = dev_acc
                # 每个epoch模型保留下来，保存模型
                torch.save(model.state_dict(), os.path.join(save_path, 'model_epochs{}.pt'.format(epoch)))
                logger.cache_in('Model saved at {}'.format(os.path.join(save_path, 'model_epochs{}.pt'.format(epoch))))
                logger.cache_in('')
                logger.cache_write()

                model.eval()
                correct_num = 0.0
                pred_ls = []
                test_pred_ls = []
                t0 = time()
                for _, batch in enumerate(tqdm(test_iter)):
                    gold_labels = batch.answer
                    input_ids_A, A_real_len = batch.A
                    input_polarity_A, _ = batch.polarity_A
                    input_ids_B, B_real_len = batch.B
                    input_polarity_B, _ = batch.polarity_B
                    input_ids_C, C_real_len = batch.C
                    input_polarity_C, _ = batch.polarity_C
                    input_ids_D, D_real_len = batch.D
                    input_polarity_D, _ = batch.polarity_D
                    input_ids_premise, _, premise_real_len = batch.candidates
                    with torch.no_grad():
                        output, loss = model(input_ids_A, input_ids_B, input_ids_C, input_ids_D,
                                             A_real_len, B_real_len, C_real_len, D_real_len,
                                             input_polarity_A, input_polarity_B, input_polarity_C, input_polarity_D,
                                             input_ids_premise, premise_real_len, gold_labels)
                    _, pred_idx = output.max(dim=1)
                    pred_ls.append(pred_idx)
                    equal = torch.sum(torch.eq(pred_idx, gold_labels.squeeze()).int()).item()
                    test_pred_ls.append(equal)
                    correct_num += equal
                total_num = len(torch.cat(pred_ls))
                test_acc = correct_num / total_num
                logger.cache_in(
                    "{} Accuarcy: {:.2f}%, correct_num: {},  total_num: {} , {:.2f} Seconds Used:".format('Test',
                                                                                                          test_acc * 100.0,
                                                                                                          correct_num,
                                                                                                          total_num,
                                                                                                          time() - t0))
                logger.cache_in(str(test_pred_ls))
                logger.cache_write()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--bert_vocab', type=str, default='../data/bert-base-uncased/vocab.txt')
    parser.add_argument('--bert_model', type=str, default='../data/bert-base-uncased')

    parser.add_argument('--RoBERTa_model', type=str, default='../data/roberta-base')
    parser.add_argument('--RoBERTa_hidden', type=int, default=768)
    parser.add_argument('--pretrained_lexrel_RoBERTa', type=str, default='whole_CLS.pt')

    parser.add_argument('--dataset_path', type=str, default="../data/")
    parser.add_argument('--train_dataset', type=str, default="train.json")
    parser.add_argument('--dev_dataset', type=str, default="dev.json")
    parser.add_argument('--test_dataset', type=str, default="test.json")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--encoded_size', type=int, default=768)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument('--subword_topk', type=int, default=5)
    parser.add_argument('--barronKG_num', type=int, default=3)
    parser.add_argument('--beam_num', type=int, default=3)

    #Euclidean -> Hyperbolic
    parser.add_argument('--hyper_hidden_size', type=int, default=384)
    parser.add_argument('--hyper_hidden_layers', type=int, default=1)
    parser.add_argument("--bias", default=1, type=int, help="Whether to use bias in the linear transformation.")
    parser.add_argument("--projection_dropout", default=0.3, type=float, help="Dropout rate for projection")
    parser.add_argument("--hyper_output_dims", default=64, type=int)
    parser.add_argument('--substitute_iteration', type=int, default=20)

    args = parser.parse_args()
    device = torch.device(args.device)
    setup_seed(1234)

    save_path = "./results/{}-{}-epochs{}".format('model', strftime("%Y%m%d-%H%M%S"), args.epochs)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = Logger(os.path.join(save_path, 'log.txt'), log_type='w+')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

    token_to_id = BertTokenizer.from_pretrained(args.bert_vocab).vocab
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    projection_polarity_map = {"up": "1", "down": "-1", "flat": "0"}
    nlp_spacy = spacy.load('en_core_web_sm')

    main()
