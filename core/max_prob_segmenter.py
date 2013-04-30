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

import logging
import pprint
import re

from hmm_segmenter import HMMSegmenter
from vocabulary import Vocabulary

class MaxProbSegmenter(object):
    """
    基于最大概率路径的中文分词 + 基于字标注的未登录词识别.

    分词步骤:
        0. 加载词典和模型数据, 支持用户自定义词典.
        1. 基于分词词典 (trie 树结构) 对文本做全切分, 构造有向无环图 (DAG),
           即词图.
        2. 基于动态规划算法计算最大概率路径, 得到基于词频的最大切分组合.
        3. 对于未登录词问题, 采用 HMMSegmnter 的字标注方法识别.
    """

    def __init__(self, vocabulary, hmm_segmenter):
        self.vocabulary = vocabulary
        self.hmm_segmenter = hmm_segmenter

        self.re_chinese = re.compile(ur"([\u4E00-\u9FA5a-zA-Z0-9+#&\._]+)")
        self.re_skip = re.compile(ur"(\s+)")

    def segment(self, text):
        """
        最大概率分词.

        NOTE: 支持自动识别 text 编码.
        """
        if not (type(text) is unicode):
            try:
                text = text.decode('utf-8')
            except:
                text = text.decode('gbk', 'ignore')

        blocks = self.re_chinese.split(text)
        for block in blocks:
            if self.re_chinese.match(block):
                for word in self._segment_block(block):
                    yield word
            else:
                fields = self.re_skip.split(block)
                for field in fields:
                    if self.re_skip.match(field):
                        yield ' '
                    else:
                        for ch in field:
                            yield ch

    def _segment_block(self, text):
        """
        unigram 模型

        TODO(fandywang): unigram  ->  bigram, trigram
        """
        DAG = self.vocabulary.gen_DAG(text)
        route = {}
        N = len(text)
        route[N] = (0.0, '')

        # 动态规划算法确定最大词频切分路径
        for i in xrange(N - 1, -1, -1):
            route[i] = max(
                    [(self.vocabulary.get_prob(text[i : j + 1]) + route[j + 1][0], j)
                        for j in DAG[i] ])

        buf = u''
        i = 0
        while i < N:
            j = route[i][1] + 1
            word = text[i : j]
            if j - i == 1:  # 单字
                buf += word
            else:
                if len(buf) > 0:
                    if len(buf) == 1:
                        yield buf
                    else:  # 未登录词识别
                        for w in self.hmm_segmenter.segment(buf):
                            yield w
                    buf = u''
                yield word
            i = j

        if len(buf) > 0:
            if len(buf) == 1:
                yield buf
            else:
                for w in self.hmm_segmenter.segment(buf):
                    yield w

