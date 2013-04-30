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

import re
from hmm import HMM

class HMMPOSTagger(object):
    """
    在中文分词结果基础上, 采用 HMM 模型实现词性标注 (Part-of-speech tagging).
    """

    def __init__(self):
        self.hmm = HMM()

        self.re_chinese = re.compile(ur"([\u4E00-\u9FA5]+)")  # 正则匹配汉字串
        self.re_skip = re.compile(ur"([\.0-9]+|[a-zA-Z0-9]+)")  # 正则匹配英文串和数字串

    def load(self, model_dir):
        """
        加载模型文件.
        """
        self.hmm.load(model_dir)

    def pos_tag(self, words):
        """
        基于 HMM 模型的词性标注.
        """
        log_prob, pos_list = self._viterbi(words)

        for i, w in enumerate(words):
            yield (w, pos_list[i])

