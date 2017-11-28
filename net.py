#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import chainer
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers

"""
Memory Network

"""


class XP:
    __lib = None
    @staticmethod
    def set_library(args):
        if args.gpu >= 0:
          XP.__lib = cuda.cupy
          cuda.get_device(args.gpu).use()
          XP.xp = XP.__lib
        else:
          XP.__lib = np
          XP.xp = XP.__lib
    @staticmethod
    def __zeros(shape, dtype):
        return Variable(XP.__lib.zeros(shape, dtype=dtype))
    @staticmethod
    def fzeros(shape):
        return XP.__zeros(shape, XP.__lib.float32)
    @staticmethod
    def __array(array, dtype):
        return Variable(XP.__lib.array(array, dtype=dtype))
    @staticmethod
    def iarray(array):
        return XP.__array(array, XP.__lib.int32)
    @staticmethod
    def farray(array):
        return XP.__array(array, XP.__lib.float32)



class MemoryNet(chainer.Chain):
    def __init__(self, n_vocab, word_emb_size, nhop=3):
        super(MemoryNet, self).__init__(
            A=links.EmbedID(n_vocab, word_emb_size, ignore_label=-1), # for input
            B=links.EmbedID(n_vocab, word_emb_size, ignore_label=-1), # for query 
            C=links.EmbedID(n_vocab, word_emb_size, ignore_label=-1), # for output 
            W=links.Linear(word_emb_size, n_vocab), # for answer
            )

    def encode_input(self, x_input):
        # print functions.sum(self.A(x_input), axis=2).data
        return functions.sum(self.A(x_input), axis=1)

    def encode_query(self, x_query):
        return functions.sum(self.B(x_query), axis=1)

    def encode_output(self, x_input):
        return functions.sum(self.C(x_input), axis=1)

    def encode(self, x_input, x_query, answer):
        m = self.encode_input(x_input)
        u = self.encode_query(x_query)

        # print "m.data.shape", m.data.shape
        # print "u.data.shape", u.data.shape
        mu = functions.matmul(m, u, transb=True)
        # print "mu.data.shape", mu.data.shape
        # print "mu.data",  mu.data
        p = functions.softmax(mu)
        c = self.encode_output(x_input)
        # print "p.data.shape:", p.data.shape
        # print "c.data.shape:", c.data.shape
        # print "functions.swapaxes(c ,2, 1):", functions.swapaxes(c ,2, 1).data.shape
        o = functions.matmul(functions.swapaxes(c ,1, 0), p) # (2, 50, 1)
        o = functions.swapaxes(o ,1, 0) # (2, 50) 
        # print "u.data.shape:", u.data.shape
        # print "o.data.shape:", o.data.shape
        # print "u.data.shape:", u.data
        # print "o.data.shape:", o.data
        # print (u+o).data.shape
        predict = self.W(u + o)
        # print predict.data.shape
        loss = functions.softmax_cross_entropy(predict, answer)
        return loss


    def train(self, char_idx, target_embedding):
        h = self.infer(char_idx=char_idx)
        loss = functions.mean_squared_error(h, target_embedding)
        return loss, h

    def infer(self, char_idx):
        h = self.encoder.encode(char_idx=char_idx, char_emb=self.char_emb)
        # regression
        h = functions.sigmoid(self.l1(h))
        h = self.l2(h)
        return h
    def save(self):
        print 'save!'



if __name__ == '__main__':
    memNN = MemoryNet(n_vocab=50, word_emb_size=50, nhop=3)
    x_input = [
                
                [0, 1, 3, -1],
                [2, 3, 4, -1],
                [6, 4, -1, -1],
            
                # [
                #     [0, 1, 3],
                #     [2, 3, 2],
                #     [6, 4, -1],
                #     [6, 4, -1]

                # ],
              ]
    x_query = [
                
                [0, 1, 2],
                [1, 2, 5],
                [3, 7, 9],
                
              ]
    answer = [1, 2, 3]


    x_input = np.array(x_input).astype(np.int32)
    x_query = np.array(x_query).astype(np.int32)
    answer = np.array(answer).astype(np.int32)

    x_input = Variable(x_input)
    x_query = Variable(x_query)
    answer = Variable(answer)
    print x_input.data
    print x_query.data
    print x_input.data.shape
    print x_query.data.shape
    opt = chainer.optimizers.Adam(alpha=0.01, beta1=0.9, beta2=0.999, eps=1e-6)
    opt.setup(memNN)

    for _ in range(20):
        loss = memNN.encode(x_input, x_query, answer)
        print loss.data
        opt.zero_grads()
        # print 'loss_backward'
        loss.backward()
        # print 'opt.update'
        opt.update()
