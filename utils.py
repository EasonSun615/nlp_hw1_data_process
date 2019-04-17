# -*- coding: utf-8 -*-
import concurrent.futures
import pickle
import numpy as np
import random
import os
# from keras.preprocessing.sequence import pad_sequences

process_num = 8
word_dict = dict()
vectors=[]
neg_list = []

tot = 3924369

label_dict = {}
for i in range(10):
    label_dict[i] = [0 for x in range(10)]
    label_dict[i][i] = 1


def word2id(c):
        if c in word_dict:
            return word_dict[c]
        else:
            return 0

def wordEmbedding():
     # with open('..'+ff+'sgns.weibo.word', encoding='utf8') as f:
    global vectors
    global word_dict
    with open('./sgns.weibo.word', encoding='utf8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            line = line.split(' ')
            word_dict[line[0]] = i
            vectors.append(list(map(float, line[1:-1])))
        print('{}/195202 words dictionary is done. Now begin storing it'.format(len(lines)-1))
    #with open('..'+ff+'data'+ff+'embedding_matrix.pkl', "wb") as f:
    with open('./embedding.pkl', "wb") as f:
        vectors = np.array(vectors)
        pickle.dump(vectors, f)
    #with open('..'+ff+'data'+ff+'word_dict.pkl', "wb") as f:
    with open('./worddict.pkl', "wb") as f:
        pickle.dump(word_dict, f)
    print('embedding_matrix and dictionary is saved')


def build_train_data(lines, tid=0):
    cnt = 0
    history = []
    true_utt = []
    for line in lines:
        # fields = line.rstrip().lower().split('\t')
        # utterance = fields[1].split('###')
        utterance = line[:-1]
        history.append([list(map(word2id, each_utt.split())) for each_utt in utterance])
        true_utt.append(list(map(word2id, line[-1].split())))
        cnt += 1
        if cnt % 10000 == 0:
            print(tid, cnt)
    return history, true_utt

def build_neg_data(neglist, tid):
    cnt = 0
    result = []
    for line in neglist:
        # fields = line.rstrip().lower().split('\t')
        # utterance = fields[1].split('###')
        result.append(list(map(word2id, line.split())))
        cnt += 1
        if cnt % 10000 == 0:
            print(tid, cnt)
    return result


def build_val_data(convs, resps, tid = 0):
    cnt =0
    history = []
    resp = []
    for conv in convs:
        history.append([list(map(word2id, each_utt.split())) for each_utt in conv])
    for re in resps:
        resp.append(list(map(word2id, re.split())))
    return history, resp



def load_val_data():
    executor = concurrent.futures.ProcessPoolExecutor(process_num)
    conver_num = 0
    base = 0
    # conversation
    convers = []
    conver = []
    results = []

    resps = []
    conver_cnt = 0
    resps_cnt = 0
    history = []
    utt = []
    labels = []
    true_label = []
    with open('../nlp-hw1/valid/valid_context.txt',encoding="utf8") as f:
        conversations = f.readlines()
        for i,line in enumerate(conversations):
            if line == '\n':
                for i in range(10):
                    convers.append(conver)
                    conver_cnt += 1
                conver = []
                continue
            conver.append(line)
        conver_num = conver_cnt/10
        print("%d conversations"% conver_num )
    with open('../nlp-hw1/valid/valid_reply.txt', encoding="utf8") as f:
        responses = f.readlines()
        for i, line in enumerate(responses):
            if line == '\n':
                continue
            resps.append(line)
            resps_cnt += 1
    if conver_cnt != resps_cnt:
        print("ERROR: conversation count:%d, resps_cnt:%d"%(conver_cnt,resps_cnt))
    step = conver_cnt // process_num
    print('{} step'.format(step))
    low = 0
    while True:
        if low < conver_cnt:
            results.append(executor.submit(build_val_data, convers[low:low + step], resps[low: low+step], base))
        else:
            break
        base += 1
        low += step
    for result in results:
        h, t = result.result()
        history += h
        utt += t

    with open('../nlp-hw1/valid/valid_ground.txt', encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            label = line.strip('\n')
            labels.append(label)
        for label in labels:
            true_label += label_dict[int(label)]
    print("len of history: %d" % len(history))
    print("len of utt: %d" % len(utt))
    print("len of ture_label: %d" % len(true_label))

    with open("./evaluate.pkl", "wb") as f:
        # list
        pickle.dump([history, utt, true_label], f)







def get_sequences_length(sequences, maxlen):
    sequences_length = [min(len(sequence), maxlen) for sequence in sequences]
    return sequences_length

def load_data():
    executor = concurrent.futures.ProcessPoolExecutor(process_num)
    base = 0
    results = []
    results_neg = []
    history = []
    true_utt = []
    neg_examples = []
    # word_dict = dict()
    # vectors = []
    #ff = os.sep

    # train_file = '..'+ff+'ori_data'+ff+'train'+ff+'train_session.txt'
    train_file = '../nlp-hw1/train/train_session.txt'

    with open(train_file, encoding="utf8") as f:
        lines = f.readlines()
        total_num = 0
        us = []
        u = []
        for i, line in enumerate(lines):
            if line == '\n':
                total_num += 1
                us.append(u)
                u = []
                # generate negetive examples
                while True:
                    k = random.randint(0, tot)
                    if k!=i and lines[k]!='\n':
                        neg_list.append(lines[k])
                        break
                continue
            u.append(line)

        print('{} examples'.format(total_num))
        low = 0
        step = total_num // process_num
        print('{} step'.format(step))
        while True:
            if low < total_num:
                results.append(executor.submit(build_train_data, us[low:low + step], base))
                results_neg.append(executor.submit(build_neg_data,neg_list[low: low+step], base))
            else:
                break
            base += 1
            low += step

        for result in results:
            h, t = result.result()
            # history 三层list ，[[[每个字的index，...],每句话，...]，每轮话,...]
            history += h
            true_utt += t
        for result in results_neg:
            neg_examples += result.result()

    print("len(history)",len(history))
    print("len(true_utt)",len(history))
    print("len(neg_examples)",len(neg_examples))

    #with open(".."+ff+"data"+ff+"utterances.pkl", "wb") as f:
    with open("./utterances.pkl", "wb") as f:
        # list 
        pickle.dump([history, true_utt], f)
    with open("./responses.pkl", "wb") as f:
        pickle.dump(neg_examples,f)

def load_test_data():
    executor = concurrent.futures.ProcessPoolExecutor(process_num)
    conver_num = 0
    base = 0
    # conversation
    convers = []
    conver = []
    results = []

    resps = []
    conver_cnt = 0
    resps_cnt = 0
    history = []
    utt = []
    labels = []
    true_label = []
    with open('../nlp-hw1/test/test_context.txt',encoding="utf8") as f:
        conversations = f.readlines()
        for i,line in enumerate(conversations):
            if line == '\n':
                for i in range(10):
                    convers.append(conver)
                    conver_cnt += 1
                conver = []
                continue
            conver.append(line)
        conver_num = conver_cnt/10
        print("%d conversations"% conver_num )
    with open('../nlp-hw1/test/test_reply.txt', encoding="utf8") as f:
        responses = f.readlines()
        for i, line in enumerate(responses):
            if line == '\n':
                continue
            resps.append(line)
            resps_cnt += 1
    if conver_cnt != resps_cnt:
        print("ERROR: conversation count:%d, resps_cnt:%d"%(conver_cnt,resps_cnt))
    step = conver_cnt // process_num
    print('{} step'.format(step))
    low = 0
    while True:
        if low < conver_cnt:
            results.append(executor.submit(build_val_data, convers[low:low + step], resps[low: low+step], base))
        else:
            break
        base += 1
        low += step
    for result in results:
        h, t = result.result()
        history += h
        utt += t

    print("len of history: %d" % len(history))
    print("len of utt: %d" % len(utt))

    with open("./test.pkl", "wb") as f:
        # list
        pickle.dump([history, utt], f)


if __name__ == '__main__':
    wordEmbedding()
    load_data()
    load_val_data()
    load_test_data()

