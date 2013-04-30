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

class HMMSegmenter(object):
    """
    采用 HMM 模型实现基于字标注的中文分词 (Character-based Tagging).

    以往的分词方法, 无论是基于规则的, 还是基于统计的, 一般都依赖一个事先编制
    的词表(词典). 自动分词过程就是通过词表和歧义消除来做出词语切分的决策.

    与此不同, 基于字标注的分词方法从构词角度入手, 把分词过程视为字在字串中的
    标注问题. 由于每个字在构造一个特定的词语时都占据着一个确定的构词位置(即词位),
    假如规定每个字最多只有四个构词位置：即B(词首), M(词中), E(词尾)和S(单独成词),
    那么下面句子(a)的分词结果就可以直接表示成如(b)所示的逐字标注形式:

    　　(a)分词结果：上海／ 计划／ N／ 本／ 世纪／ 末／ 实现／ 人均／ 国内／
                     生产／ 总值／ 五千美元／ 。
    　　(b)字标注形式：上／B 海／E 计／B 划／E N／S 本／S 世／B 纪／E 末／S
                       实／B 现／E 人／B 均／E 国／B 内／E生／B产／E总／B值／E
                       五／B千／M 美／M 元／E 。／S

    需要特别说明的是，这里说到的“字”不限于汉字. 考虑到中文真实文本中不可避免地
    包含一定数量的非汉字字符, “字”也包括外文字母、阿拉伯数字和标点符号等字符,
    所有这些字符都是构词的基本单元. 当然, 汉字依然是这个单元集合中数量最多的.

    把分词过程视为字的标注问题的一个重要优势在于, 它能够平衡地看待词表词和
    未登录词的识别问题. 在这种分词技术中, 文本中的词表词和未登录词都可以用统一
    的字标注过程来实现. 在学习架构上, 既可以不必专门强调词表词信息, 也不用
    专门设计特定的未登录词 (如人名、地名、机构名) 识别模块. 这使得分词系统的
    设计大大简化.

      (1) 首先, 在字标注过程中, 所有的字根据预定义的特征进行词位特性的学习,
          获得一个概率模型, 可以使用 HMM、MaxEnt、CRF 建模.
      (2) 然后, 在待分字串上, 根据字与字之间的结合紧密程度, 得到词位的标注结果.
      (3) 最后, 根据词位定义直接获得最终的分词结果.
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

    def segment(self, text):
        """
        对输入文本 text 做中文切词, 返回词序列.

        NOTE: 支持 text 编码自动识别.
        """
        if not (type(text) is unicode):
            try:
                text = text.decode('utf-8')
            except:
                text = text.decode('gbk', 'ignore')

        blocks = self.re_chinese.split(text)
        for block in blocks:
            if self.re_chinese.match(block):
                for word in self._tagging(block):
                    yield word
            else:
                words = self.re_skip.split(block)
                for word in words:
                    if len(word) > 0:
                        yield word

    def _tagging(self, text):
        """
        基于 HMM 模型切词.
        """
        log_prob, tag_list = self.hmm.viterbi(text)

        begin = 0
        for i, ch in enumerate(text):
            tag = tag_list[i]
            if tag == 'B':
                begin = i
            elif tag == 'E':
                yield text[begin : i + 1]
            elif tag == 'S':
                yield ch

