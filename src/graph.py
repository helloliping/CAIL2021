# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 图模型

if __name__ == '__main__':
	import sys
	sys.path.append('../')


import os
import time
import numpy
import torch
import jieba
import jieba.analyse	# 2021/12/31 18:47:09 知识盲区, 这个模块得单独import
import pandas
import networkx

from copy import deepcopy
from torch.nn import Module, Embedding, Linear, Sigmoid, CrossEntropyLoss, functional as F

from wordcloud import WordCloud
from matplotlib import pyplot as plt

from setting import *

from src.data_tools import load_stopwords, filter_stopwords
from src.qa_module import BaseLSTMEncoder, BaseAttention
from src.utils import load_args, timer


class Graph:
	
	def __init__(self):
		
		pass
	
	
	def build_reference_graph():
		reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
		stopwords = load_stopwords(stopword_names=None)
		for law, group_dataframe in reference_dataframe.groupby(['law']):
			group_dataframe = group_dataframe.reset_index(drop=True)
			contents = []
			for i in range(group_dataframe.shape[0]):
				content = filter_stopwords(tokens=eval(group_dataframe.loc[i, 'content']), stopwords=stopwords)
				contents.append(content)
			text = '\n'.join(contents)		
		
		
	def build_concept_graph(self, model, threshold=.9, allow_negative=False):
		"""
		构建概念网络
		"""
		# model = gensim.models.Word2Vec.load(REFERENCE_WORD2VEC_MODEL_PATH)
		# dictionary = Dictionary.load(REFERENCE_DICTIONARY_PATH)
		similarity_index = WordEmbeddingSimilarityIndex(model.wv)	
		similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)
		
		
		model.wv.similarity('盗窃', '抢劫')
