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

from core.hmm_pos_tagger import HMMPOSTagger
from core.hmm_segmenter import HMMSegmenter
from core.max_prob_segmenter import MaxProbSegmenter
from core.vocabulary import Vocabulary

class WordSegmenter(object):
    """A Chinese Word Segment Toolkit.

    分词算法:
        1. 加载词典和模型数据, 支持用户自定义词典.
        2. 基于分词词典 (trie 树结构) 对文本做全切分, 构造有向无环图 (DAG),
           即词图.
        3. 基于动态规划算法计算最大概率路径, 得到基于词频的最大切分组合.
        4. 对于未登录词问题, 采用 HMMSegmnter 的字标注方法识别.

    词性标注算法:
        1. 中文分词, 得到词序列.
        2. 基于 HMM模型解码算法实现词性标注.

    TODO(fandywang):
        1. 时间、数词、人名、地名、机构名、email、url识别.
        2. 繁简转换.
        3. 多粒度切分.
        4. 停用词识别.
    """
    VOCABULARY_FILENAME = 'vocabulary.dat'  # 基本分词词典
    CUSTOM_WORDS_DIR = "custom_words"  # 用户自定义词典
    HMM_SEGMENT_MODEL_DIR = 'hmm_segment_model'  # HMM 字标注中文分词模型
    HMM_POS_MODEL_DIR = 'hmm_pos_model'  # HMM n-gram 词性标注模型

    def __init__(self):
        self.vocabulary = Vocabulary()
        self.hmm_segmenter = HMMSegmenter()
        self.max_prob_segmenter = None
        self.hmm_pos_tagger = HMMPOSTagger()

    def load(self, data_dir):
        """
        加载词典和模型文件.
        """
        self.vocabulary.load(data_dir + '/'
                + self.__class__.VOCABULARY_FILENAME,
                data_dir + '/'
                + self.__class__.CUSTOM_WORDS_DIR)
        self.hmm_segmenter.load(data_dir + '/'
                + self.__class__.HMM_SEGMENT_MODEL_DIR)
        self.max_prob_segmenter = MaxProbSegmenter(
                self.vocabulary, self.hmm_segmenter)

        self.hmm_pos_tagger.load(data_dir + '/'
                + self.__class__.HMM_POS_MODEL_DIR);

    def segment(self, text):
        """
        切词, 返回切词序列.
        """
        return self.max_prob_segmenter.segment(text)

    def segment_with_pos(self, text):
        """
        切词 + 词性标注, 返回词和词性组成的元组序列.
        """
        return self.hmm_pos_tagger.pos_tag(
                self.max_prob_segmenter.segment(text))

