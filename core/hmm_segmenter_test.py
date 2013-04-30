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

import unittest

from hmm_segmenter import HMMSegmenter

class HMMSegmenterTest(unittest.TestCase):

    def setUp(self):
        self.hmm_segmenter = HMMSegmenter()
        self.hmm_segmenter.load('../data/hmm_segment_model')

    def call_segment(self, text):
        for word in self.hmm_segmenter.segment(text):
            print word + '/\t',
        print ''

    def test_segment(self):
        fp = open('testdata/document.dat', 'rb')
        for text in fp.readlines():
            self.call_segment(text.strip())
        fp.close()

if __name__ == '__main__':
    unittest.main()

