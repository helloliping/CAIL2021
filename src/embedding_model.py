# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 词嵌入模型

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import time
import jieba
import gensim
import pandas
import pickle

from copy import deepcopy
from gensim.corpora import MmCorpus, Dictionary
from gensim.models import Word2Vec, WordEmbeddingSimilarityIndex
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix


from setting import *

from src.data_tools import load_stopwords, filter_stopwords
from src.utils import timer

class GensimEmbeddingModel:
	"""gensim模块下的词嵌入模型"""
	def __init__(self, args):
		"""
		:param args	: EmbeddingModelConfig配置
		"""
		self.args = deepcopy(args)
		
		if self.args.filter_stopword:
			self.stopwords = load_stopwords(stopword_names=None)
	
	@timer
	def build_word2vec_model(self, 
							 corpus_import_path=REFERENCE_CORPUS_PATH,
							 document_import_path=REFERENCE_DOCUMENT_PATH,
							 model_export_path=REFERENCE_WORD2VEC_MODEL_PATH):
		kwargs = {
			'size'		: self.args.size_word2vec,
			'min_count'	: self.args.min_count_word2vec,
		}
		return GensimEmbeddingModel.easy_build_model(model_name='word2vec',
													 corpus_import_path=corpus_import_path,
													 document_import_path=document_import_path,
													 model_export_path=model_export_path,
													 **kwargs)
	@timer
	def build_fasttext_model(self, 
							 corpus_import_path=REFERENCE_CORPUS_PATH,
							 document_import_path=REFERENCE_DOCUMENT_PATH,
							 model_export_path=REFERENCE_WORD2VEC_MODEL_PATH):
		kwargs = {
			'size'		: self.args.size_fasttext,
			'min_count'	: self.args.min_count_fasttext,
		}
		return GensimEmbeddingModel.easy_build_model(model_name='fasttext',
													 corpus_import_path=corpus_import_path,
													 document_import_path=document_import_path,
													 model_export_path=model_export_path,
													 **kwargs)
	
	@classmethod
	def easy_build_model(cls,
						 model_name,
						 corpus_import_path,
						 document_import_path,
						 model_export_path,
						 **kwargs):
		"""
		20211218更新: 
		最近发现用corpus_file参数训练得到的模型词汇表全是索引而非分词
		而且观察下来跟dictionary的索引还对不上, 非常的恼火, 只能改用sentences参数的写法了
		"""
		
		# model = eval(GENSIM_EMBEDDING_MODEL_SUMMARY[model_name]['class'])(corpus_file=corpus_import_path, **kwargs)
		model = eval(GENSIM_EMBEDDING_MODEL_SUMMARY[model_name]['class'])(sentences=pickle.load(open(document_import_path, 'rb')), **kwargs)
		if model_export_path is not None:
			model.save(model_export_path)
		return model
	
	@timer
	def build_similarity(self, model_name):
		"""构建模型的gensim相似度(Similarity)"""
		model = eval(GENSIM_EMBEDDING_MODEL_SUMMARY[model_name]['class']).load(GENSIM_EMBEDDING_MODEL_SUMMARY[model_name]['model'])
		dictionary = Dictionary.load(REFERENCE_DICTIONARY_PATH)
		corpus = MmCorpus(REFERENCE_CORPUS_PATH)
		similarity_index = WordEmbeddingSimilarityIndex(model.wv)
		similarity_matrix = SparseTermSimilarityMatrix(similarity_index, dictionary)
		similarity = SoftCosineSimilarity(corpus, similarity_matrix, num_best=self.args.num_best)
		return similarity

	def query(self, query_tokens, dictionary, similarity):
		"""
		给定查询分词列表返回相似度匹配向量
		:param query_tokens	: 需要查询的关键词分词列表
		:param dictionary	: gensim字典
		:param similarity	: gensim相似度
		:param sequence		: 模型序列
		:return result		: 文档中每个段落的匹配分值
		"""
		if self.args.filter_stopword:
			filter_stopwords(tokens=query_tokens, stopwords=self.stopwords)
		query_corpus = dictionary.doc2bow(query_tokens)
		result = similarity[query_corpus]
		return result

	
class BertEmbeddingModel:
	
	pass
