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

import pprint
import unittest

from vocabulary import Vocabulary

class VocabularyTest(unittest.TestCase):

    def setUp(self):
        self.vocabulary = Vocabulary()
        self.vocabulary.load('testdata/vocabulary.dat', 'testdata/custom_words')

        pprint.pprint(self.vocabulary.trie)
        pprint.pprint(self.vocabulary.word_prob)

    def test_vocabulary(self):
        self.assertIn(u'英雄三国', self.vocabulary.word_prob.keys())
        self.assertIn(u'魔鬼代言人', self.vocabulary.word_prob.keys())
        self.assertIn(u'黄河水利委员会', self.vocabulary.word_prob.keys())
        self.assertNotIn(u'十大伪歌手', self.vocabulary.word_prob.keys())
        self.assertNotIn(u'走路太牛', self.vocabulary.word_prob.keys())

        self.assertEqual('n', self.vocabulary.get_pos(u'英雄三国'))
        self.assertEqual('n', self.vocabulary.get_pos(u'魔鬼代言人'))
        self.assertEqual('nt', self.vocabulary.get_pos(u'黄河水利委员会'))
        self.assertEqual('UNK', self.vocabulary.get_pos(u'十大伪歌手'))
        self.assertEqual('UNK', self.vocabulary.get_pos(u'走路太牛'))

    def test_gen_DAG(self):
        pprint.pprint(self.vocabulary.gen_DAG(
            u'《英雄三国》是由网易历时四年自主研发运营的一款英雄对战竞技网游。'))

if __name__ == '__main__':
    unittest.main()

