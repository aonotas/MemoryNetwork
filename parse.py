#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

def file_list():
    basename = './tasks_1-20_v1-2/en/'
    files = os.listdir(basename)
    files = sorted([f for f in files if '.txt' in f])
    files = [(int(f.split('_')[0].replace('qa', '')) ,f) for f in files]
    train_list = []
    test_list = []
    for q_id, filename in files:
        if '_train' in filename:
            train_list.append((q_id, basename + filename))
        else:
            test_list.append((q_id, basename + filename))
    train_filenames = dict(train_list)
    test_filenames = dict(test_list)
    return train_filenames, test_filenames
# print train_filenames

def r(task_id, filename, vocab):
    batch_list = []
    item_fact = []
    item_qa = []
    lines = [l for l in open(filename)]
    for i, l in enumerate(lines):
        lsplit = l.strip().split('\t')
        text = lsplit[0].replace('?', ' ?').replace('!', ' !').replace('.', ' .').lower()
        s_id = int(text.split()[0])
        sentence_split = text.split()[1:]
        sentence_str = ' '.join(sentence_split)
        # print len(lsplit)

        if (s_id == 1 and i != 0) or i == len(lines):
            # print '***'
            # print item_fact
            # print item_qa
            batch_list.append((item_fact, item_qa))
            item_fact = []
            item_qa = []

        if len(lsplit) == 1:
            sentence_idx = build_vocab(vocab, sentence_split)
            item_fact.append((task_id, s_id, sentence_idx, sentence_str))
        else:
            # print lsplit

            # answer = lsplit[1].lower().split(',')
            answer = [lsplit[1].lower()]
            # print answer
            answer_hint = map(int, lsplit[2].split())
            sentence_idx = build_vocab(vocab, sentence_split)
            answer_idx = build_vocab(vocab, answer)
            item_qa.append((task_id, s_id, sentence_idx, sentence_str, answer_idx, answer, answer_hint))
        # print s_id, 
        # print sentence_str

        # print lsplit

    # print batch_list[-1]
    return batch_list

def build_vocab(vocab, word_list):
    index_list = []
    for w in word_list:
        if w not in vocab:
            vocab[w] = len(vocab)
        index_list.append(vocab[w])
    return index_list


def build_dataset():
    train_filenames, test_filenames = file_list()
    vocab = {}
    train_dataset = []
    test_dataset = []
    for task_id in range(1, 20 + 1):
        train_filename = train_filenames[task_id]
        train_batch_list = r(task_id, train_filename, vocab)
        for fact, qa in train_batch_list:
            train_dataset.append((fact, qa))
        test_filename = test_filenames[task_id]
        test_batch_list = r(task_id, test_filename, vocab)
        for fact, qa in test_batch_list:
            test_dataset.append((fact, qa))
        # print train_batch_list
        # train_dataset.append(train_batch_list)
        # test_dataset.append(test_batch_list)
    print len(set([i for i, _ in vocab.items()]))
    print len(set([i.lower() for i, _ in vocab.items()]))
    return train_dataset, test_dataset, vocab


if __name__ == '__main__':
    build_dataset()