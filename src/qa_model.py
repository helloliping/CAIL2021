# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 用于解题的问答模型

if __name__ == '__main__':
	import sys
	sys.path.append('../')

import torch
import pandas

from copy import deepcopy
from torch.nn import Module, Embedding, Linear, Sigmoid, CrossEntropyLoss, functional as F

from setting import *

from src.data_tools import encode_answer, decode_answer
from src.qa_module import BaseLSTMEncoder, BaseAttention
from src.utils import load_args, timer


class BaseChoiceModel(Module):
	"""选择题Baseline模型: 不使用参考文献"""
	def __init__(self, args):
		super(BaseChoiceModel, self).__init__()
		self.d_hidden = 128
		self.n_tokens = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0).shape[0]
		self.confusion_matrix = []
		self.embedding = Embedding(self.n_tokens, self.d_hidden)
		self.options_encoder = BaseLSTMEncoder()
		self.question_encoder = BaseLSTMEncoder()
		self.attention = BaseAttention()
		self.rank_module = Linear(self.d_hidden * 2, 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data, mode='train'):
		assert mode in ['train', 'test']
		options = data['options']
		question = data['question']
		batch_size = question.size()[0]
		n_options = TOTAL_OPTIONS
		embedded_options = self.embedding(options.view(batch_size * n_options, -1))
		embedded_question = self.embedding(torch.cat([question.view(batch_size, -1) for _ in range(n_options)]))	# 扩展问题的维度与选项相同
		_, encoded_options = self.options_encoder(embedded_options)
		_, encoded_question = self.question_encoder(embedded_question)
		options_attention, question_attention, attention = self.attention(encoded_options, encoded_question)
		y = torch.cat([torch.max(options_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)
		y = self.rank_module(y.view(batch_size * n_options, -1))
		output = self.multi_module(y.view(batch_size, n_options))
		return output


class BaseJudgmentModel(Module):
	"""判断题Baseline模型: 不使用参考文献"""
	def __init__(self, args):
		super(BaseJudgmentModel, self).__init__()
		self.d_hidden = 128
		self.n_tokens = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0).shape[0]
		self.confusion_matrix = []
		self.embedding = Embedding(self.n_tokens, self.d_hidden)
		self.options_encoder = BaseLSTMEncoder()
		self.question_encoder = BaseLSTMEncoder()
		self.attention = BaseAttention()
		self.rank_module = Linear(self.d_hidden * 2, 1)
		self.activation_function = Sigmoid()
	
	def forward(self, data, mode='train'):
		assert mode in ['train', 'test']
		option = data['option']
		question = data['question']
		batch_size = question.size()[0]
		embedded_option = self.embedding(option.view(batch_size, -1))
		embedded_question = self.embedding(question.view(batch_size, -1))
		_, encoded_option = self.options_encoder(embedded_option)
		_, encoded_question = self.question_encoder(embedded_question)
		option_attention, question_attention, attention = self.attention(encoded_option, encoded_question)
		y = torch.cat([torch.max(option_attention, dim=1)[0], torch.max(question_attention, dim=1)[0]], dim=1)
		y = self.rank_module(y.view(batch_size, -1))
		output = self.activation_function(y).squeeze(-1)				# 如果不squeeze输出结果形如[[.9], [.8], [.5]], 希望得到形如[.9, .8, .5]的输出结果
		return output


class ReferenceChoiceModel(Module):
	"""选择题Baseline模型: 使用参考文献"""
	def __init__(self, args):
		super(ReferenceChoiceModel, self).__init__()
		self.d_hidden = 128
		self.n_tokens = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0).shape[0]
		self.confusion_matrix = []
		self.embedding = Embedding(self.n_tokens, self.d_hidden)
		self.options_encoder = BaseLSTMEncoder()
		self.question_encoder = BaseLSTMEncoder()
		self.reference_encoder = BaseLSTMEncoder()
		self.attention = BaseAttention()
		self.rank_module = Linear(64, 1)
		self.multi_module = Linear(4, 16)

	def forward(self, data, mode='train'):
		assert mode in ['train', 'test']
		options = data['options']
		question = data['question']
		reference = data['reference']
		batch_size = question.size()[0]
		n_options = TOTAL_OPTIONS
		
		
		# torch.Size([32, 4, 128])
		# torch.Size([32, 256])
		# torch.Size([32, 32, 512])
		# torch.Size([32, 512, 128])
		# torch.Size([32, 256, 128])
		# torch.Size([32, 16384, 128])
		
		print('options', options.shape)
		print('question', question.shape)
		print('reference', reference.shape)
		
		embedded_options = self.embedding(options.view(batch_size, -1))
		embedded_question = self.embedding(question.view(batch_size, -1))
		embedded_reference = self.embedding(reference.view(batch_size, -1))
		
		print('embedded_options', embedded_options.shape)
		print('embedded_question', embedded_question.shape)
		print('embedded_reference', embedded_reference.shape)
		
		embedded_options_and_question = torch.cat([embedded_options, embedded_question], axis=1)
		
		print('embedded_options_and_question', embedded_options_and_question.shape)
		
		_, encoded_options_and_question = self.options_encoder(embedded_options_and_question)

		print('encoded_options_and_question', encoded_options_and_question.shape)
		_, encoded_reference = self.reference_encoder(embedded_reference)
		
		print('encoded_reference', encoded_reference.shape)
		
		options_and_question_attention, reference_attention, attention = self.attention(encoded_options_and_question, encoded_reference)
		
		print('options_and_question_attention', options_and_question_attention.shape)
		print('reference_attention', reference_attention.shape)
		print('attention', attention.shape)
		
		y = torch.cat([torch.max(options_and_question_attention, dim=1)[0], torch.max(reference_attention, dim=1)[0]], dim=1)
		
		print('y', y.shape)
		
		y = self.rank_module(y.view(batch_size * n_options, -1))
		
		print('y', y.shape)
		
		# options torch.Size([2, 4, 128])
		# question torch.Size([2, 256])
		# reference torch.Size([2, 32, 512])
		# embedded_options torch.Size([2, 512, 128])
		# embedded_question torch.Size([2, 256, 128])
		# embedded_reference torch.Size([2, 16384, 128])
		# embedded_options_and_question torch.Size([2, 768, 128])
		# encoded_options_and_question torch.Size([2, 768, 128])
		# encoded_reference torch.Size([2, 16384, 128])
		# options_and_question_attention torch.Size([2, 768, 128])
		# reference_attention torch.Size([2, 16384, 128])
		# attention torch.Size([2, 768, 16384])
		# y torch.Size([2, 256])
		# y torch.Size([8, 1])

		output = self.multi_module(y.view(batch_size, n_options))
		return output


class ReferenceJudgmentModel(Module):
	"""选择题Baseline模型: 使用参考文献"""
	def __init__(self, args):
		super(ReferenceJudgmentModel, self).__init__()
		self.d_hidden = 128
		self.n_tokens = pandas.read_csv(REFERENCE_TOKEN2ID_PATH, sep='\t', header=0).shape[0]
		self.confusion_matrix = []
		self.embedding = Embedding(self.n_tokens, self.d_hidden)
		self.option_encoder = BaseLSTMEncoder()
		self.question_encoder = BaseLSTMEncoder()
		self.reference_encoder = BaseLSTMEncoder()
		self.attention = BaseAttention()
		self.rank_module = Linear(self.d_hidden * 2, 1)
		self.activation_function = Sigmoid()

	def forward(self, data, mode='train'):
		assert mode in ['train', 'test']
		option = data['option']
		question = data['question']
		reference = data['reference']
		batch_size = question.size()[0]
		n_option = TOTAL_OPTIONS
		
		# print('option', option.shape)
		# print('question', question.shape)
		# print('reference', reference.shape)
		
		embedded_option = self.embedding(option.view(batch_size, -1))
		embedded_question = self.embedding(question.view(batch_size, -1))
		embedded_reference = self.embedding(reference.view(batch_size, -1))
		
		# print('embedded_option', embedded_option.shape)
		# print('embedded_question', embedded_question.shape)
		# print('embedded_reference', embedded_reference.shape)
		
		embedded_option_and_question = torch.cat([embedded_option, embedded_question], axis=1)
		
		# print('embedded_option_and_question', embedded_option_and_question.shape)
		
		_, encoded_option_and_question = self.option_encoder(embedded_option_and_question)

		# print('encoded_option_and_question', encoded_option_and_question.shape)
		_, encoded_reference = self.reference_encoder(embedded_reference)
		
		# print('encoded_reference', encoded_reference.shape)
		
		option_and_question_attention, reference_attention, attention = self.attention(encoded_option_and_question, encoded_reference)
		
		# print('option_and_question_attention', option_and_question_attention.shape)
		# print('reference_attention', reference_attention.shape)
		# print('attention', attention.shape)
		
		y = torch.cat([torch.max(option_and_question_attention, dim=1)[0], torch.max(reference_attention, dim=1)[0]], dim=1)
		
		# print('y', y.shape)
		
		y = self.rank_module(y.view(batch_size, -1))
		
		# print('y', y.shape)
		
		output = self.activation_function(y).squeeze(-1)
		return output	

	
if __name__ == '__main__':
	
	pass
