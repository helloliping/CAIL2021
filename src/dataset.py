# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 数据预处理相关方法

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import os
import time
import jieba
import torch
import numpy
import pandas
import gensim
import logging

from copy import deepcopy
from collections import Counter
from gensim.corpora import Dictionary
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from setting import *
from config import RetrievalModelConfig, EmbeddingModelConfig
from src.data_tools import load_stopwords, encode_answer, decode_answer, chinese_to_number, filter_stopwords
from src.retrieval_model import GensimRetrieveModel
from src.utils import load_args, timer


# 生成数据加载器
# 把这个函数写在Dataset类里作为类方法会报错找不到__collate_id函数, 目前没有找到很好的解决方案
# 如果放到data_tools.py中的话会导致循环调用, 因此只能先放在这里了
def generate_dataloader(args, mode='train', do_export=False, pipeline='judgment'):
	dataset = Dataset(args=args, 
					  mode=mode, 
					  do_export=do_export, 
					  pipeline=pipeline)
	column = dataset.data.columns.tolist()
	if mode.startswith('train'):
		batch_size = args.train_batch_size
		shuffle = True
	if mode.startswith('valid'):
		batch_size = args.valid_batch_size
		shuffle = False
	if mode.startswith('test'):
		batch_size = args.test_batch_size
		shuffle = False
	
	def _collate_fn(_batch_data):
		
		def __collate_id():
			return [__data['id'] for __data in _batch_data]
		
		if args.word_embedding is None:
			
			def __collate_question():
				return torch.LongTensor([__data['question'] for __data in _batch_data])

			def __collate_reference():
				return torch.LongTensor([__data['reference'] for __data in _batch_data])
				
			def __collate_options():
				return torch.LongTensor([__data['options'] for __data in _batch_data])
				
			def __collate_option():
				return torch.LongTensor([__data['option'] for __data in _batch_data])
				
		else:
			def __collate_question():
				return torch.FloatTensor([__data['question'] for __data in _batch_data])

			def __collate_reference():
				return torch.FloatTensor([__data['reference'] for __data in _batch_data])
				
			def __collate_options():
				return torch.FloatTensor([__data['options'] for __data in _batch_data])
				
			def __collate_option():
				return torch.FloatTensor([__data['option'] for __data in _batch_data])			

		def __collate_type():
			return [__data['type'] for __data in _batch_data]
			
		def __collate_subject():
			return torch.LongTensor([__data['subject'] for __data in _batch_data])
		
		def __collate_label_choice():
			return torch.LongTensor([__data['label_choice'] for __data in _batch_data])		# 0-15的选择题答案编码值
			
		def __collate_label_judgment():
			return torch.LongTensor([__data['label_judgment'] for __data in _batch_data])	# 零一的判断题答案编码值

		def __collate_option_id():
			return [__data['option_id'] for __data in _batch_data]							# 20211216更新: 选项号需要记录进来

		_collate_data = {}
		for _column in column:
			_collate_data[_column] = eval(f'__collate_{_column}')()
		return _collate_data

	dataloader = DataLoader(dataset=dataset,
							batch_size=batch_size,
							num_workers=args.num_workers,
							collate_fn=_collate_fn,
							shuffle=shuffle)
	return dataloader


class Dataset(Dataset):
	"""模型输入数据集管道"""
	def __init__(self, args, mode='train', do_export=False, pipeline='judgment'):
		"""
		:param args			: DatasetConfig配置
		:param mode			: 数据集模式, 详见下面第一行的断言
		:param do_export	: 是否导出self.data
		"""
		self.pipelines = {
			'choice': self.choice_pipeline,
			'judgment': self.judgment_pipeline,
		}
		
		assert mode in ['train', 'train_kd', 'train_ca', 'valid', 'valid_kd', 'valid_ca', 'test', 'test_kd', 'test_ca']
		assert pipeline in self.pipelines
		
		# 构造变量转为成员变量
		self.args = deepcopy(args)
		self.mode = mode
		self.do_export = do_export
		self.pipeline = pipeline
		
		# 根据配置生成对应的成员变量
		if self.args.filter_stopword:
			self.stopwords = load_stopwords(stopword_names=None)
		if self.args.use_reference:
			self.grm = GensimRetrieveModel(args=load_args(Config=RetrievalModelConfig))		
		if self.args.word_embedding in GENSIM_RETRIEVAL_MODEL_SUMMARY:
			self.gem = GensimEmbeddingModel(args=load_args(Config=EmbeddingModelConfig))	
		
		# 生成数据表
		self.pipelines[pipeline]()
		
		# 导出数据表
		if self.do_export:
			logging.info('导出数据表...')
			self.data.to_csv(COMPLETE_REFERENCE_PATH, sep='\t', header=True, index=False)
	
	@timer
	def choice_pipeline(self):
		"""
		选择题形式的输入数据, 输出字段有:
		id			: 题目编号
		question	: 题目题干
		options		: 合并后的四个选项
		subject		: use_reference配置为True时生效, 包含num_top_subject个法律门类
		reference	: use_reference配置为True时生效, 包含相关的num_best个参考书目文档段落
		type		: 零一值表示概念题或情景题
		label_choice: train或valid模式时生效, 即题目答案
		"""	
		if self.mode.startswith('train'):
			filepaths = TRAINSET_PATHs[:]
		elif self.mode.startswith('valid'):  # 20211101新增验证集处理逻辑
			filepaths = VALIDSET_PATHs[:]
		elif self.mode.startswith('test'):
			filepaths = TESTSET_PATHs[:]
		else:
			assert False
		max_option_length = self.args.max_option_length
		max_statement_length = self.args.max_statement_length
		max_reference_length = self.args.max_reference_length

		# 数据集字段预处理
		logging.info('预处理题目题干与选项...')
		start_time = time.time()

		token2id_dataframe = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0)
		token2id = {token: _id for token, _id in zip(token2id_dataframe['token'], token2id_dataframe['id'])}			# token2id字典: 20211212后决定以参考书目文档的token2id为标准, 而非题库的token2id

		dataset_dataframe = pandas.concat([pandas.read_csv(filepath, sep='\t', header=0) for filepath in filepaths])	# 合并概念题和情景题后的题库
		if self.mode.endswith('_kd'):   
			dataset_dataframe = dataset_dataframe[dataset_dataframe['type'] == 0].reset_index(drop=True)		# 筛选概念题
		elif self.mode.endswith('_ca'): 
			dataset_dataframe = dataset_dataframe[dataset_dataframe['type'] == 1].reset_index(drop=True)		# 筛选情景分析题
		else:
			dataset_dataframe = dataset_dataframe.reset_index(drop=True)										# 无需筛选直接重索引
			
		dataset_dataframe['id'] = dataset_dataframe['id'].astype(str)				# 字段id转为字符串
		dataset_dataframe['type'] = dataset_dataframe['type'].astype(int)			# 字段type转为整数
		dataset_dataframe['statement'] = dataset_dataframe['statement'].map(eval)	# 字段statement用eval函数转为分词列表				
		dataset_dataframe['option_a'] = dataset_dataframe['option_a'].map(eval)		# 字段option_a用eval函数转为分词列表		
		dataset_dataframe['option_b'] = dataset_dataframe['option_b'].map(eval)		# 字段option_b用eval函数转为分词列表		
		dataset_dataframe['option_c'] = dataset_dataframe['option_c'].map(eval)		# 字段option_c用eval函数转为分词列表		
		dataset_dataframe['option_d'] = dataset_dataframe['option_d'].map(eval)		# 字段option_d用eval函数转为分词列表		
		
		if self.args.word_embedding is None:
			# 使用token2id的顺序编码值进行词嵌入
			dataset_dataframe['question'] = dataset_dataframe['statement'].map(self.token_to_id(max_length=max_statement_length, token2id=token2id))																# 题目题干的分词列表转为编号列表
			dataset_dataframe['options'] = dataset_dataframe[['option_a', 'option_b', 'option_c', 'option_d']].apply(self.combine_option(max_length=max_option_length, token2emb=token2id, encode_as='id'), axis=1)	# 题目选项的分词列表转为编号列表并合并
		elif self.args.word_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
			# 使用gensim词向量模型进行训练
			embedding_model_class = eval(GENSIM_EMBEDDING_MODEL_SUMMARY[self.args.word_embedding]['class'])
			embedding_model_path = GENSIM_EMBEDDING_MODEL_SUMMARY[self.args.word_embedding]['model']
			embedding_model = embedding_model_class.load(embedding_model_path)
			token2vector = embedding_model.wv
			self.vector_size = embedding_model.wv.vector_size
			del embedding_model

			dataset_dataframe['question'] = dataset_dataframe['statement'].map(self.token_to_vector(max_length=max_statement_length, token2vector=token2vector))															# 题目题干的分词列表转为编号列表
			dataset_dataframe['options'] = dataset_dataframe[['option_a', 'option_b', 'option_c', 'option_d']].apply(self.combine_option(max_length=max_option_length, token2emb=token2vector, encode_as='vector'), axis=1)	# 题目选项的分词列表转为编号列表并合并
		else:
			# 目前尚未完成其他词嵌入的使用
			raise NotImplementedError
		
		# 参考文献相关字段预处理
		if self.args.use_reference:
			# 加载文档检索模型相关内容
			dictionary_path = GENSIM_RETRIEVAL_MODEL_SUMMARY[self.args.retrieval_model_name]['dictionary']
			if dictionary_path is None:		# logentropy模型的dictionary字段是None
				dictionary_path = REFERENCE_DICTIONARY_PATH
			dictionary = Dictionary.load(dictionary_path)
			similarity = self.grm.build_similarity(model_name=self.args.retrieval_model_name)
			sequence = GensimRetrieveModel.load_sequence(model_name=self.args.retrieval_model_name)

			reference_dataframe = pandas.read_csv(REFERENCE_PATH, sep='\t', header=0)
			index2subject = {index: '法制史' if law == '目录和中国法律史' else law for index, law in enumerate(reference_dataframe['law'])}		# 记录reference_dataframe中每一行对应的法律门类
			
			# 新生成的几个字段说明:
			# query_result		: 形如[(4, 0.8), (7, 0.1), (1, 0.1)], 列表长度为args.num_best
			# reference_index	: 将[4, 7, 1]给抽取出来
			# reference			: 将[4, 7, 1]对应的参考书目文档的段落的分词列表给抽取出来并转为编号列表
			# subject			: 题目对应的args.num_top_subject个候选法律门类
			logging.info('生成查询得分向量...')
			dataset_dataframe['query_result'] = dataset_dataframe[['statement', 'option_a', 'option_b', 'option_c', 'option_d']].apply(self.generate_query_result(dictionary=dictionary, 
																																								  similarity=similarity, 
																																								  sequence=sequence), axis=1)
			dataset_dataframe['reference_index'] = dataset_dataframe['query_result'].map(lambda result: list(map(lambda x: x[0], result)))
			

			logging.info('检索参考书目文档段落...')
			
			if self.args.word_embedding is None:
				dataset_dataframe['reference'] = dataset_dataframe['reference_index'].map(self.find_reference_by_index(max_length=max_reference_length, 
																													   token2emb=token2id, 
																													   reference_dataframe=reference_dataframe,
																													   encode_as='id'))
			elif self.args.word_embedding in GENSIM_EMBEDDING_MODEL_SUMMARY:
				dataset_dataframe['reference'] = dataset_dataframe['reference_index'].map(self.find_reference_by_index(max_length=max_reference_length, 
																													   token2emb=token2vector, 
																													   reference_dataframe=reference_dataframe,
																													   encode_as='vector'))			
			logging.info('填充subject字段的缺失值...')
			dataset_dataframe['subject'] = dataset_dataframe[['reference_index', 'subject']].apply(self.fill_subject(index2subject), axis=1)

			if self.mode.startswith('train') or self.mode.startswith('valid'):
				dataset_dataframe['label_choice'] = dataset_dataframe['answer'].astype(int)
				self.data = dataset_dataframe[['id', 'question', 'options', 'subject', 'reference', 'type', 'label_choice']].reset_index(drop=True)
			elif self.mode.startswith('test'):
				self.data = dataset_dataframe[['id', 'question', 'options', 'subject', 'reference', 'type']].reset_index(drop=True)
		else:
			if self.mode.startswith('train') or self.mode.startswith('valid'):
				dataset_dataframe['label_choice'] = dataset_dataframe['answer'].astype(int)
				self.data = dataset_dataframe[['id', 'question', 'options', 'type', 'label_choice']].reset_index(drop=True)
			elif self.mode.startswith('test'):
				self.data = dataset_dataframe[['id', 'question', 'options', 'type']].reset_index(drop=True)
	
	@timer
	def judgment_pipeline(self):
		"""
		20211121更新: 判断题形式的输入数据, 输出字段有:
		id				: 题目编号
		question		: 题目题干
		option			: 每个选项
		subject			: use_reference配置为True时生效, 包含num_top_subject个法律门类
		reference		: use_reference配置为True时生效, 包含相关的num_best个参考书目文档段落
		type			: 零一值表示概念题或情景题
		label_judgment	: train或valid模式时生效, 零一值表示判断题的答案
		option_id		: 20211216更新, 记录判断题对应的原选择题编号(ABCD)
		"""		
		self.choice_pipeline()
		self.data = self.choice_to_judgment(choice_dataframe=self.data, id_column='id', choice_column='options', answer_column='label_choice')
		
	def token_to_id(self, max_length, token2id):
		"""题目题干分词列表转编号"""
		def _token_to_id(_tokens):
			_ids = list(map(lambda _token: token2id.get(_token, token2id['UNK']), _tokens))
			if len(_ids) >= max_length:
				return _ids[: max_length]
			else:
				return _ids + [token2id['PAD']] * (max_length - len(_ids))
		return _token_to_id
		
	def token_to_vector(self, max_length, token2vector):
		"""20211218更新: 引入词向量对分词进行编码"""
		unk_or_pad_vector = [0] * self.vector_size
		def _token_to_vector(_tokens):
			_vectors = list(map(lambda _token: token2vector[_token] if _token in token2vector else unk_or_pad_vector[:], _tokens))
			if len(_vectors) >= max_length:
				return _vectors[: max_length]
			else:
				return _vectors + [unk_or_pad_vector[:] for _ in range(max_length - len(_vectors))]	# 注意这里复制列表就别用乘号了, 老老实实用循环免得出问题
		return _token_to_vector
		
	def combine_option(self, max_length, token2emb, encode_as='id'):
		"""题目选项分词列表转编号并合并"""		
		if encode_as == 'id':
			def _combine_option(_dataframe):
				__token_to_id = self.token_to_id(max_length=max_length, token2id=token2emb)
				_option_a, _option_b, _option_c, _option_d = _dataframe
				return [__token_to_id(_option_a),
						__token_to_id(_option_b),
						__token_to_id(_option_c),
						__token_to_id(_option_d)]
		elif encode_as == 'vector':
			def _combine_option(_dataframe):
				__token_to_vector = self.token_to_vector(max_length=max_length, token2vector=token2emb)
				_option_a, _option_b, _option_c, _option_d = _dataframe
				return [__token_to_vector(_option_a),
						__token_to_vector(_option_b),
						__token_to_vector(_option_c),
						__token_to_vector(_option_d)]		
		else:
			raise NotImplementedError

		return _combine_option

	
	def generate_query_result(self, dictionary, similarity, sequence):
		"""生成查询得分向量"""
		def _generate_query_result(_dataframe):
			_statement, _option_a, _option_b, _option_c, _option_d = _dataframe						# 提取用于查询文档的关键词: 题干与四个选项
			_query_tokens = _statement + _option_a + _option_b + _option_c + _option_d				# 拼接题目和四个选项的分词
			if self.args.filter_stopword:											
				_query_tokens = filter_stopwords(tokens=_query_tokens, stopwords=self.stopwords)	# 筛除停用词
			return self.grm.query(query_tokens=_query_tokens, 
								  dictionary=dictionary, 
								  similarity=similarity, 
								  sequence=sequence)
		return _generate_query_result


	def find_reference_by_index(self, max_length, token2emb, reference_dataframe, encode_as='id'):
		"""根据新生成的reference_index寻找对应的参考段落, 并转为编号形式"""
		if encode_as == 'id':
			def _find_reference_by_index(_reference_index):
				_reference = []
				__token_to_id = self.token_to_id(max_length=max_length, token2id=token2emb)	
				for _index in _reference_index:
					_tokens = eval(reference_dataframe.loc[_index, 'content'])	# reference_index对应在reference_dataframe中的分词列表
					_reference.append(__token_to_id(_tokens))
				if len(_reference) < self.args.num_best:						# 2021/12/19 11:18:24 竟然Similarity可能返回的结果不足num_best, 也不是很能理解, 只能手动填补了
					for _ in range(self.args.num_best - len(_reference)):
						_reference.append([token2emb['UNK']] * max_length)		# 2021/12/19 11:25:16 填补为UNK
				return _reference
		elif encode_as == 'vector':
			def _find_reference_by_index(_reference_index):
				_reference = []
				__token_to_vector = self.token_to_vector(max_length=max_length, token2vector=token2emb)	
				for _index in _reference_index:
					_tokens = eval(reference_dataframe.loc[_index, 'content'])	# reference_index对应在reference_dataframe中的分词列表
					_reference.append(__token_to_vector(_tokens))
				if len(_reference) < self.args.num_best:						# 2021/12/19 11:18:24 竟然Similarity可能返回的结果不足num_best, 也不是很能理解, 只能手动填补了
					for _ in range(self.args.num_best - len(_reference)):
						_reference.append(numpy.zeros((self.vector_size, )))	# 2021/12/19 11:25:26 填补为零向量
				return _reference		
		else:
			raise NotImplementedError
		return _find_reference_by_index


	def fill_subject(self, index2subject):
		"""填充缺失的subject字段, 这里拟填充args.top_subject个候选subject"""
		def _fill_subject(_dataframe):
			_reference_index, subject = _dataframe
			if subject == subject:															# 不缺失的情况无需填充, 直接填充到长度为args.top_subject即可
				return [SUBJECT2INDEX[subject]] + [0] * (self.args.num_top_subject - 1)
			_candidate_subjects = [index2subject[_index] for _index in _reference_index]	# 根据reference_index生成候选的法律门类: index2subject记录了参考书目文档里每一行对应的法律门类
			_weighted_count = {}															# 记录每个候选的法律门类的加权累和数
			for _rank, _subject in enumerate(_candidate_subjects):
				if subject in _weighted_count:
					_weighted_count[_subject] += 1 / (_rank + 1)							# 这个加权的方式就是MRR
				else:
					_weighted_count[_subject] = 1 / (_rank + 1)
			_counter = Counter(_weighted_count).most_common(self.args.num_top_subject)		# 提取前args.top_subject个候选subject, 注意虽然是提取self.args.top_subject个, 但是可能会不足, 所以return的时候还需要填充
			_predicted_subjects = list(map(lambda x: x[0], _counter))						# 真实预测得到的至多args.top_subject给法律门类
			# 转为真实的法律门类字符串并填充到长度为args.top_subject
			return [SUBJECT2INDEX[_subject] for _subject in _predicted_subjects] + [0] * (self.args.num_top_subject - len(_predicted_subjects))	
		return _fill_subject


	def choice_to_judgment(self, choice_dataframe, id_column='id', choice_column='options', answer_column='label_choice'):
		"""
		20211124更新: 将选择题形式的dataframe转为判断题形式的dataframe
		此处使用了dataframe的apply高阶用法, 使用result_type='expand'的模式来简化代码格式
		:param choice_dataframe		: 选择题形式的dataframe
		:param id_column			: 题目编号所在的字段名, 用于表的连接
		:param choice_column		: 题目选项所在的字段名, 要求是一个长度为4的可迭代对象, 用于拆分
		:param answer_column		: 题目答案所在的字段名, 要求是0-15的编码值
		:return judgment_dataframe	: 判断题形式的dataframe
		"""
		# 左表是去除选项和答案的原数据表
		left_dataframe_columns = choice_dataframe.columns.tolist()
		left_dataframe_columns.remove(choice_column)
		if self.mode.startswith('train') or self.mode.startswith('valid'):
			left_dataframe_columns.remove(answer_column)
		left_dataframe = choice_dataframe[left_dataframe_columns]

		# 右表是由问题编号、单选项、判断真伪三个字段构成的数据表
		if self.mode.startswith('train') or self.mode.startswith('valid'):
			right_dataframe_columns = [id_column, 'option', 'label_judgment']
			right_dataframe = pandas.concat([choice_dataframe[[id_column]].apply(lambda x: [x[0]] * 4, axis=1, result_type='expand').stack().reset_index(drop=True),									# 通过expand将一道题的id扩充为四道题
											 choice_dataframe[[choice_column]].apply(lambda x: [x[0][0], x[0][1], x[0][2], x[0][3]], axis=1, result_type='expand').stack().reset_index(drop=True),		# 通过expand将一道题的四个选项扩充为四道题
											 choice_dataframe[[answer_column]].apply(lambda x: decode_answer(x[0], result_type=int), axis=1, result_type='expand').stack().reset_index(drop=True)], 	# 使用result_type为int的decode_answer方法获得每个选项的对错值, 并扩充成四道题
											 axis=1)
		else:
			right_dataframe_columns = [id_column, 'option']
			right_dataframe = pandas.concat([choice_dataframe[[id_column]].apply(lambda x: [x[0]] * 4, axis=1, result_type='expand').stack().reset_index(drop=True),									# 通过expand将一道题的id扩充为四道题
											 choice_dataframe[[choice_column]].apply(lambda x: [x[0][0], x[0][1], x[0][2], x[0][3]], axis=1, result_type='expand').stack().reset_index(drop=True)],		# 通过expand将一道题的四个选项扩充为四道题
											 axis=1)
		right_dataframe.columns = right_dataframe_columns
		right_dataframe['option_id'] = ['A', 'B', 'C', 'D'] * (right_dataframe.shape[0] // 4)	# 20211216更新: 选项号需要记录进来
		
		judgment_dataframe = left_dataframe.merge(right_dataframe, how='left', on=id_column).reset_index(drop=True)
		return judgment_dataframe

	def __getitem__(self, item):
		return self.data.loc[item, :]

	def __len__(self):
		return len(self.data)






