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
import math
import os

class Vocabulary(object):
    """
    分词词典.

    使用 Trie 树结构组织词典, 实现高效查找.
    """

    MAX_WORD_LENGTH = 16  # 词的最大长度

    def __init__(self):
        # TODO(fandywang): 实现 double array trie, 节省内存, 提高速度
        self.trie = {}  # trie 树结构组织词典, 实现高效查找
        self.words = {}  # word->(prob, pos), 词频率分布
        self.total_freq = 0.0
        self.min_prob = 1.0

    def load(self, vocabulary_file, custom_words_dir = None):
        """
        加载词典, 包括基本词典和用户自定义词典.
        """
        self._load_vocabulary(vocabulary_file)
        if custom_words_dir is not None:
            self._load_custom_words(custom_words_dir)

        for word, word_attr in self.words.iteritems():
            prob = math.log(word_attr[0] / self.total_freq)
            self.words[word] = (prob, word_attr[1])
            self.min_prob = min(self.min_prob, prob)
        # pprint.pprint(self.trie)
        # pprint.pprint(self.words)

    def _load_vocabulary(self, vocabulary_file):
        """
        加载基本分词词典, 构建 trie 树.
        """
        logging.info('Load vocabulary from %s.' % vocabulary_file)
        fp = open(vocabulary_file, 'rb')
        for line in fp.readlines():
            line = line.strip().decode('utf-8')
            # remove bom flag if it exists
            line = line.replace(u'\ufeff', u"")
            fields  = line.split('\t')
            if len(fields) < 3:
                logging.warning('Line format error, line: %s.' % line)
                continue
            word, freq, pos = fields[0], float(fields[1]), fields[2]
            if word in self.words:
                logging.warning('Duplicate word, line: %s' % line)
                continue

            self.words[word] = (freq, pos)
            self.total_freq += freq
            self.min_prob = min(self.min_prob, freq)
            self._insert_trie(word)
        fp.close()

    def _load_custom_words(self, custom_words_dir):
        """
        加载用户自定义词典, 丰富 trie 树.
        """
        logging.info('Load custom_words from %s.' % custom_words_dir)
        for root, dirs, files in os.walk(custom_words_dir):
            for f in files:
                filename = os.path.join(root, f)
                logging.info('Load filename %s.' % filename)
                fp = open(filename, 'rb')
                for line in fp.readlines():
                    line = line.strip().decode('utf-8')
                    # remove bom flag if it exists
                    line = line.replace(u'\ufeff', u"")
                    fields = line.split('\t')
                    if len(fields) < 3:
                        logging.warning('Line format error, line: %s.' % line)
                        continue

                    word, freq, pos = fields[0], float(fields[1]), fields[2]
                    self.words[word] = (freq, pos)
                    self.total_freq += freq
                    self._insert_trie(word)
                fp.close()

    def _insert_trie(self, word):
        ptr = self.trie
        for ch in word:
            if not ch in ptr:
                ptr[ch] = {}
            ptr = ptr[ch]
        ptr[''] = ''  # ending flag

    def get_prob(self, word):
        """
        获取 word 的概率, 如果 word 不在词典中, 返回最小概率.
        """
        return self.words.get(word, (self.min_prob, ''))[0]

    def get_pos(self, word):
        """
        获取 word 的词性.
        """
        return self.words.get(word, (0, 'UNK'))[1]

    def gen_DAG(self, text):
        """
        生成词图.
        """
        N = len(text)
        DAG = {}
        ptr = self.trie
        i, j = 0, 0

        while i < N:
            ch = text[j]
            if ch in ptr:
                ptr = ptr[ch]
                if '' in ptr:
                    if not i in DAG:
                        DAG[i] = []
                    DAG[i].append(j)
                j += 1
                if j >= N or j - i > self.__class__.MAX_WORD_LENGTH:
                    i += 1
                    j = i
                    ptr = self.trie
            else:
                ptr = self.trie
                i += 1
                j = i

        for i in xrange(len(text)):
            if not i in DAG:
                DAG[i] = [i]
        return DAG

