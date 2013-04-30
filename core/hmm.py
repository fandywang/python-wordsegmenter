#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-wordsegmenter project.
# Author: Lifeng Wang (ofandywang@gmail.com)
# Desc: A Python Implementation of Chinese Word Segmenter, which is mainly
#       modified from jieba project (https://github.com/fxsjy/jieba).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

class HMM(object):
    """
    HMM 模型.

    TODO(fandywang): 增加模型训练代码, 主要分两种:
        1. 有指导学习: 在人工标注数据集基础上, 采用最大似然估计 (MLE) 方法得到
           n-gram 模型.
        2. 无指导学习: 不需要人工标注数据集, 采用 Baum-Welch 算法估计参数.
           Baum-Welch 算法又称前向-后向算法 (Forward-backward algorithm),
           属于一种典型的 EM 算法.
    """
    STATES_FILENAME = "states.dat"
    START_LOG_PROB_FILENAME = 'start_log_prob.dat'
    TRANS_LOG_PROB_FILENAME = 'trans_log_prob.dat'
    EMIT_LOG_PROB_FILENAME = 'emit_log_prob.dat'
    DEFAULT_LOG_PROB = -3.14E100  # 平滑

    def __init__(self):
        self.states = None  # 隐含状态集
        self.start_log_prob = None  # 起始概率矩阵
        self.trans_log_prob = None  # 状态转移概率矩阵
        self.emit_log_prob = None  # 发射概率矩阵

    def load(self, model_dir):
        """
        加载模型文件.
        """
        self.states = eval(open(model_dir + '/'
            + self.__class__.STATES_FILENAME, 'rb').read())
        self.start_log_prob = eval(open(model_dir + '/'
            + self.__class__.START_LOG_PROB_FILENAME, 'rb').read())
        self.trans_log_prob = eval(open(model_dir + '/'
            + self.__class__.TRANS_LOG_PROB_FILENAME, 'rb').read())
        self.emit_log_prob = eval(open(model_dir + '/'
            + self.__class__.EMIT_LOG_PROB_FILENAME, 'rb').read())

    def viterbi(self, obs):
        """
        HMM 解码: 基于动态规划的 Viterbi 算法.

        V[0][k] = start_log_prob[k] + emit_log_prob[k][obs[0]]
        V[t][k] = argmax_k0 { V[t-1][k] + trans_log_prob[k0][k] }
                  + emit_log_prob[k][obs[t]] }

        k, k0 表示隐含状态 states.
        """
        V = [{}]  # tabular
        path = {}

        for k in self.states:  # init
            V[0][k] = (self.start_log_prob[k]
                    + self.emit_log_prob[k].get(obs[0], self.__class__.DEFAULT_LOG_PROB))
            path[k] = [k]

        for t in xrange(1, len(obs)):
            V.append({})
            newpath = {}
            for k in self.states:
                (log_prob, state) = \
                        max([(V[t - 1][k0]
                            + self.trans_log_prob[k0].get(k, self.__class__.DEFAULT_LOG_PROB), k0)
                            for k0 in self.states])
                V[t][k] = log_prob + self.emit_log_prob[k].get(obs[t], self.__class__.DEFAULT_LOG_PROB)
                newpath[k] = path[state] + [k]
            path = newpath
        (log_prob, state) = max([(V[len(obs) - 1][k], k) for k in ('E', 'S')])

        return (log_prob, path[state])

