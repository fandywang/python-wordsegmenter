#!/usr/bin/env python
#coding=utf-8

# Copyright(c) 2013 python-wordsegmenter project.
# Author: Lifeng Wang (ofandywang@gmail.com)
#

import unittest

from hmm_segmenter import HMMSegmenter
from max_prob_segmenter import MaxProbSegmenter
from vocabulary import Vocabulary

class MaxProbSegmenterTest(unittest.TestCase):

    def setUp(self):
        self.vocabulary = Vocabulary()
        self.vocabulary.load('../data/vocabulary.dat')
        self.hmm_segmenter = HMMSegmenter()
        self.hmm_segmenter.load('../data/hmm_model')
        self.max_prob_segmenter = MaxProbSegmenter(
                self.vocabulary, self.hmm_segmenter)

    def call_segment(self, text):
        for word in self.max_prob_segmenter.segment(text):
            print word + '/\t',
        print ''

    def test_segment(self):
        fp = open('testdata/document.dat', 'rb')
        for text in fp.readlines():
            self.call_segment(text.strip())
        fp.close()

if __name__ == '__main__':
    unittest.main()

