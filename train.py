#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import parse
import net


def fill_batch(batch, fill_token=-1):
    max_len = max(len(x) for x in batch)
    return [x + [fill_token] * (max_len - len(x)) for x in batch]


if __name__ == '__main__':
    train_dataset, test_dataset, vocab = parse.build_dataset()
    n_vocab = len(vocab)
    memNN = net.MemoryNet(n_vocab=n_vocab, word_emb_size=50, nhop=3)
    opt = chainer.optimizers.Adam(alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-6)
    opt.setup(memNN)

    
    n_minibatch = 1
    for epoch in range(100):
        print '*epoch ', epoch
        sum_loss = 0.0
        mode = "train"
        n_sample = len(train_dataset)
        if epoch % 10 == 0 and epoch > 0:
            mode = "test"
            n_sample = len(test_dataset)
        
        iteration_list = range(0, n_sample, n_minibatch)
        iteration_list = np.random.permutation(iteration_list) 
        for i_count, i in enumerate(iteration_list):
            if mode == "train":
                batch_dataset = train_dataset[i:i + n_minibatch]
            else:
                batch_dataset = test_dataset[i:i + n_minibatch]
            # x_input = []
            # x_query = []
            if i_count % 1000 == 0:
                print i_count, ' / ', n_sample
            for fact_list, qa_list in batch_dataset:
                x_input_fact = []
                y_answer = []
                for fact in fact_list:
                    task_id, s_id, fact_idx, fact_str = fact
                    x_input_fact.append(fact_idx)
                x_input_fact = fill_batch(x_input_fact)
                
                x_input_query = []
                for qa in qa_list:
                    task_id, s_id, question_idx, question_str, answer_idx, answer, answer_hint = qa
                    x_input_query.append(question_idx)
                    y_answer.append(answer_idx[0])

                # print x_input_fact
                # print x_input_query
                # print batch_dataset
                x_input_query = fill_batch(x_input_query)
                # x_input.append(x_input_fact)
                # x_query.append(x_input_query)


            x_input = Variable(np.array(x_input_fact).astype(np.int32))
            x_query = Variable(np.array(x_input_query).astype(np.int32))
            y_answer = Variable(np.array(y_answer).astype(np.int32))

            # print x_input.data
            # print x_query.data
            # print y_answer.data
            loss = memNN.encode(x_input, x_query, y_answer)
            # print loss
            sum_loss += loss.data
            if mode == "train":
                opt.zero_grads()
                loss.backward()
                opt.update()

            # print '****'
        print mode,
        print '** loss=', sum_loss