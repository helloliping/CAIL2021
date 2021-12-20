# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 全局变量设定

import os
import torch
import platform

# Linux系统使用相对路径读取文件时需要添加前缀
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLATFORM = platform.system()
DIR_SUFFIX = '' if PLATFORM == 'Windows' else '/'
DIR_SUFFIX = ''

# data文件夹及其结构设定
DATA_DIR = DIR_SUFFIX + 'data'

RAWDATA_DIR		= os.path.join(DATA_DIR, 'JEC-QA')						# [JEC-QA.zip](https://jecqa.thunlp.org/readme)压缩包解压到当前文件夹可得RAWDATA_DIR中的若干内容
NEWDATA_DIR		= os.path.join(DATA_DIR, 'JEC-QA-preprocessed')			# 存放预处理的新数据

REFERENCE_DIR	= os.path.join(RAWDATA_DIR, 'reference_book')			# 存放JEC-QA数据集中的原始参考书目文档，包含在[JEC-QA.zip](https://jecqa.thunlp.org/readme)中
STOPWORDS_DIR	= os.path.join(DATA_DIR, 'stopwords-master')			# [stopwords-master.zip](https://github.com/goto456/stopwords)压缩包解压到当前文件夹可得

RAW_TRAINSET_PATHs	= [os.path.join(RAWDATA_DIR, '0_train.json'), os.path.join(RAWDATA_DIR, '1_train.json')]	# 原始训练集包含两个JSON文件: 第一个是概念题, 第二个是情境题
RAW_TESTSET_PATHs	= [os.path.join(RAWDATA_DIR, '0_test.json'), os.path.join(RAWDATA_DIR, '1_test.json')]		# 原始测试集包含两个JSON文件: 第一个是概念题, 第二个是情境题

TRAINSET_PATHs	= [os.path.join(NEWDATA_DIR, '0_train.csv'), os.path.join(NEWDATA_DIR, '1_train.csv')]		# 预处理后的训练集包含两个JSON文件: 第一个是概念题, 第二个是情境题
VALIDSET_PATHs	= [os.path.join(NEWDATA_DIR, '0_valid.csv'), os.path.join(NEWDATA_DIR, '1_valid.csv')]		# 预处理后的验证集包含两个JSON文件: 第一个是概念题, 第二个是情境题
TESTSET_PATHs	= [os.path.join(NEWDATA_DIR, '0_test.csv'), os.path.join(NEWDATA_DIR, '1_test.csv')]		# 预处理后的测试集包含两个JSON文件: 第一个是概念题, 第二个是情境题

TOKEN2ID_PATH					= os.path.join(NEWDATA_DIR, 'token2id.csv')						# 预处理得到的分词编号文件(题库)
TOKEN2FREQUENCY_PATH			= os.path.join(NEWDATA_DIR, 'token2frequency.csv')				# 预处理得到的分词词频文件(题库)
REFERENCE_PATH					= os.path.join(NEWDATA_DIR, 'reference_book.csv')				# 预处理得到的参考书目文件
REFERENCE_TOKEN2ID_PATH			= os.path.join(NEWDATA_DIR, 'reference_token2id.csv')			# 预处理得到的分词编号文件(参考书目)
REFERENCE_TOKEN2FREQUENCY_PATH	= os.path.join(NEWDATA_DIR, 'reference_token2frequency.csv')	# 预处理得到的分词词频文件(参考书目)

STOPWORD_PATHs = {
    'baidu'	: os.path.join(STOPWORDS_DIR, 'baidu_stopwords.txt'),	# 百度停用词表
    'cn'	: os.path.join(STOPWORDS_DIR, 'cn_stopwords.txt'),		# 中文停用词表
    'hit'	: os.path.join(STOPWORDS_DIR, 'hit_stopwords.txt'),		# 哈工大停用词表
    'scu'	: os.path.join(STOPWORDS_DIR, 'scu_stopwords.txt'),		# 四川大学机器智能实验室停用词库
}

# logging文件夹及其结构设定
LOGGING_DIR = DIR_SUFFIX + 'logging'

# temp文件夹及其结构设定
TEMP_DIR = DIR_SUFFIX + 'temp'

# checkpoint文件夹及其结构设定
CHECKPOINT_DIR = DIR_SUFFIX + 'checkpoint'

# image文件夹及其结构设定
IMAGE_DIR = DIR_SUFFIX + 'image'

# model文件夹及其结构设定
MODEL_DIR = DIR_SUFFIX + 'model'
RETRIEVAL_MODEL_DIR = os.path.join(MODEL_DIR, 'retrieval_model')
GENSIM_RETRIEVAL_MODEL_DIR = os.path.join(RETRIEVAL_MODEL_DIR, 'gensim')

REFERENCE_DOCUMENT_PATH				= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_document.pk')				# 参考书目字典
REFERENCE_DICTIONARY_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_dictionary.dtn')			# 参考书目字典
REFERENCE_CORPUS_PATH				= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus.cps')				# 参考书目分词权重(原始词频)
REFERENCE_CORPUS_TFIDF_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus_tfidf.cps')		# 参考书目分词权重(TFIDF处理后)
REFERENCE_CORPUS_LSI_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus_lsi.cps')			# 参考书目分词权重(LSI处理后)
REFERENCE_CORPUS_LDA_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus_lda.cps')			# 参考书目分词权重(LDA处理后)
REFERENCE_CORPUS_HDP_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus_hdp.cps')			# 参考书目分词权重(HDP处理后)
REFERENCE_CORPUS_LOGENTROPY_PATH	= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_corpus_logentropy.cps')	# 参考书目分词权重(LOGENTROPY处理后)
REFERENCE_TFIDF_MODEL_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_tfidf.m')					# 参考书目TFIDF模型
REFERENCE_LSI_MODEL_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_lsi.m')					# 参考书目LSI模型
REFERENCE_LDA_MODEL_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_lda.m')					# 参考书目LDA模型
REFERENCE_HDP_MODEL_PATH			= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_hdp.m')					# 参考书目HDP模型
REFERENCE_LOGENTROPY_MODEL_PATH		= os.path.join(GENSIM_RETRIEVAL_MODEL_DIR, 'reference_logentropy.m')			# 参考书目LogEntropy模型

# 类似注册表的字典, 便于相关代码简化
# build_function	: 在src.retrieval_model中对应的模型构建方法
# class				: 在gensim中对应的模型类
# sequence			: 该模型依次需要调用的模型序列, 如LSI模型需要先调用TFIDF生成词权矩阵后再进行奇异值分解
GENSIM_RETRIEVAL_MODEL_SUMMARY = {
	'tfidf': {
		'corpus'		: REFERENCE_CORPUS_TFIDF_PATH,
		'model'			: REFERENCE_TFIDF_MODEL_PATH,
		'dictionary'	: REFERENCE_DICTIONARY_PATH,
		'build_function': 'GensimRetrieveModel.build_tfidf_model',
		'class'			: 'gensim.models.TfidfModel',
		'sequence'		: ['tfidf'],					
	},
	'lsi': {
		'corpus'		: REFERENCE_CORPUS_LSI_PATH,
		'model'			: REFERENCE_LSI_MODEL_PATH,
		'dictionary'	: REFERENCE_DICTIONARY_PATH,
		'build_function': 'GensimRetrieveModel.build_lsi_model',		
		'class'			: 'gensim.models.LsiModel',
		'sequence'		: ['tfidf', 'lsi'],
	},
	'lda': {
		'corpus'		: REFERENCE_CORPUS_LDA_PATH,
		'model'			: REFERENCE_LDA_MODEL_PATH,
		'dictionary'	: REFERENCE_DICTIONARY_PATH,
		'build_function': 'GensimRetrieveModel.build_lda_model',
		'class'			: 'gensim.models.LdaModel',
		'sequence'		: ['tfidf', 'lda'],
	},
	
	# 20211210新增HDP模型
	'hdp': {
		'corpus'		: REFERENCE_CORPUS_HDP_PATH,
		'model'			: REFERENCE_HDP_MODEL_PATH,
		'dictionary'	: REFERENCE_DICTIONARY_PATH,
		'build_function': 'GensimRetrieveModel.build_hdp_model',
		'class'			: 'gensim.models.HdpModel',
		'sequence'		: ['hdp'],
	},	
	
	# 20211210新增LogEntropy模型
	'logentropy': {
		'corpus'		: REFERENCE_CORPUS_LOGENTROPY_PATH,
		'model'			: REFERENCE_LOGENTROPY_MODEL_PATH,
		'dictionary'	: None,											# 不知为何gensim.models.LogEntropyModel的构造参数里竟然没有id2word
		'build_function': 'GensimRetrieveModel.build_logentropy_model',
		'class'			: 'gensim.models.LogEntropyModel',
		'sequence'		: ['logentropy'],
	},
}


EMBEDDING_MODEL_DIR = os.path.join(MODEL_DIR, 'embedding_model')
GENSIM_EMBEDDING_MODEL_DIR = os.path.join(EMBEDDING_MODEL_DIR, 'gensim')

REFERENCE_WORD2VEC_MODEL_PATH = os.path.join(GENSIM_EMBEDDING_MODEL_DIR, 'reference_word2vec.m')
REFERENCE_FASTTEXT_MODEL_PATH = os.path.join(GENSIM_EMBEDDING_MODEL_DIR, 'reference_fasttext.m')
REFERENCE_DOC2VEC_MODEL_PATH = os.path.join(GENSIM_EMBEDDING_MODEL_DIR, 'reference_doc2vec.m')

GENSIM_EMBEDDING_MODEL_SUMMARY = {
	'word2vec': {
		'model': REFERENCE_WORD2VEC_MODEL_PATH,
		'class': 'gensim.models.Word2Vec',
	},
	'fasttext': {
		'model': REFERENCE_FASTTEXT_MODEL_PATH,
		'class': 'gensim.models.FastText',
	},
}

# 其他全局变量
OPTION2INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}							# 选项对应的索引
INDEX2OPTION = {index: option for option, index in OPTION2INDEX.items()}# 索引对应的选项
TOTAL_OPTIONS = len(OPTION2INDEX)										# 每道题固定的选项数
VALID_RATIO = .1														# 从训练数据中划分验证集的比例(用于本地测试)
TOKEN2ID = {'PAD': 0, 'UNK': 1}											# 预设的特殊分词符号
FREQUENCY_THRESHOLD = 1													# 统计分词的最小频次

SUBJECT2INDEX = {'法制史' if subject == '目录和中国法律史' else subject: index + 1 for index, subject in enumerate(os.listdir(REFERENCE_DIR))}	# 法律门类对应索引
INDEX2SUBJECT = {index: subject for subject, index in SUBJECT2INDEX.items()}																# 索引对应法律门类
