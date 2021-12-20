# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# 数据预处理

import os
import time
import gensim

from setting import *
from config import RetrievalModelConfig, EmbeddingModelConfig
from src.data_tools import json_to_csv, split_validset, token2frequency_to_csv, token2id_to_csv, reference_to_csv, load_stopwords, filter_stopwords
from src.retrieval_model import GensimRetrieveModel
from src.embedding_model import GensimEmbeddingModel
from src.utils import load_args, save_args, timer

# 新建所有文件夹
def makedirs():
	os.makedirs(NEWDATA_DIR, exist_ok=True)
	os.makedirs(LOGGING_DIR, exist_ok=True)
	os.makedirs(TEMP_DIR, exist_ok=True)
	os.makedirs(MODEL_DIR, exist_ok=True)
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
	os.makedirs(RETRIEVAL_MODEL_DIR, exist_ok=True)
	os.makedirs(GENSIM_RETRIEVAL_MODEL_DIR, exist_ok=True)
	os.makedirs(EMBEDDING_MODEL_DIR, exist_ok=True)
	os.makedirs(GENSIM_EMBEDDING_MODEL_DIR, exist_ok=True)


# 题库训练集与测试集的预处理
@timer
def preprocess_trainsets_and_testsets():
	token2frequency = {}
	for raw_trainset_path, trainset_path, validset_path in zip(RAW_TRAINSET_PATHs, TRAINSET_PATHs, VALIDSET_PATHs):
		dataframe, token2frequency = json_to_csv(json_path=raw_trainset_path,
												 csv_path=None,
												 token2frequency=token2frequency,
												 mode='train')
		split_validset(dataframe, train_export_path=trainset_path, valid_export_path=validset_path)

	for raw_testset_path, new_testset_path in zip(RAW_TESTSET_PATHs, TESTSET_PATHs):
		_, token2frequency = json_to_csv(json_path=raw_testset_path,
										 csv_path=new_testset_path,
										 token2frequency=token2frequency,
										 mode='test')

	token2frequency_to_csv(export_path=TOKEN2FREQUENCY_PATH, token2frequency=token2frequency)
	token2id_to_csv(export_path=TOKEN2ID_PATH, token2frequency=token2frequency)

# 参考书目的预处理
@timer
def preprocess_reference_book():
	# 参考书目的预处理
	_, token2frequency = reference_to_csv(export_path=REFERENCE_PATH)
	token2frequency_to_csv(export_path=REFERENCE_TOKEN2FREQUENCY_PATH, token2frequency=token2frequency)
	token2id_to_csv(export_path=REFERENCE_TOKEN2ID_PATH, token2frequency=token2frequency)

# gensim文档检索模型预构建
@timer
def build_gensim_retrieval_models(args=None, model_names=None, update_reference_corpus=False):
	if args is None:
		args = load_args(Config=RetrievalModelConfig)

	if model_names is None:
		model_names = list(GENSIM_RETRIEVAL_MODEL_SUMMARY.keys())
			
	grm = GensimRetrieveModel(args=args)
	
	if update_reference_corpus:
		grm.build_reference_corpus(reference_path=REFERENCE_PATH, 
								   dictionary_export_path=REFERENCE_DICTIONARY_PATH, 
								   corpus_export_path=REFERENCE_CORPUS_PATH)
	
	
	if 'tfidf' in model_names:
		# 20211214更新: 默认参数是(None, None, .25), 测试下来这一组参数的hit@3精确率有87.8%
		# ('atu', 1., .5)参数的hit@3能到91.6%
		# ('atu', .5, .5)参数的hit@3还是87.8%
		# ('ann', None, .25)参数的hit@3还是91.7%
		# 详细调参结果见文件夹temp/tfidf调参/下的结果
		args.smartirs = 'ann'
		args.pivot = None
		args.slope = .25
		grm.build_tfidf_model(corpus_import_path=REFERENCE_CORPUS_PATH, 
							  model_export_path=REFERENCE_TFIDF_MODEL_PATH,
							  corpus_export_path=REFERENCE_CORPUS_TFIDF_PATH)
	if 'lsi' in model_names:
		args.num_topics_lsi = 256
		args.power_iters_lsi = 3
		args.extra_samples_lsi = 256
		grm.build_lsi_model(corpus_import_path=REFERENCE_CORPUS_TFIDF_PATH, 
							model_export_path=REFERENCE_LSI_MODEL_PATH,
							corpus_export_path=REFERENCE_CORPUS_LSI_PATH)
	if 'lda' in model_names:
		args.num_topics_lsi = 256
		args.decay_lda = 1.
		args.iterations_lda = 500
		args.gamma_threshold_lda = .0001
		args.minimum_probability_lda = 0.
		grm.build_lda_model(corpus_import_path=REFERENCE_CORPUS_TFIDF_PATH, 
							model_export_path=REFERENCE_LDA_MODEL_PATH,
							corpus_export_path=REFERENCE_CORPUS_LDA_PATH)
	if 'hdp' in model_names:
		args.kappa_hdp = 0.8
		args.tau_hdp = 32.
		args.K_hdp = 16
		args.T_hdp = 256
		grm.build_hdp_model(corpus_import_path=REFERENCE_CORPUS_PATH,
							model_export_path=REFERENCE_HDP_MODEL_PATH,
							corpus_export_path=REFERENCE_CORPUS_HDP_PATH)
	if 'logentropy' in model_names:				
		grm.build_logentropy_model(corpus_import_path=REFERENCE_CORPUS_PATH, 
								   model_export_path=REFERENCE_LOGENTROPY_MODEL_PATH,
								   corpus_export_path=REFERENCE_CORPUS_LOGENTROPY_PATH)
	
	save_args(args=args, save_path=os.path.join(TEMP_DIR, 'RetrievalModelConfig.json'))

# gensim词嵌入模型预构建
@timer
def build_gensim_embedding_models(args=None, model_names=None):
	if args is None:
		args = load_args(Config=EmbeddingModelConfig)
		args.size_word2vec = 256
		args.min_count_word2vec = 1
		args.size_fasttext = 256			 
		args.min_count_fasttext = 1		
		
	if model_names is None:
		model_names = list(GENSIM_EMBEDDING_MODEL_SUMMARY.keys())

	gem = GensimEmbeddingModel(args=args)
	
	if 'word2vec' in model_names:
		gem.build_word2vec_model(corpus_import_path=REFERENCE_CORPUS_PATH, model_export_path=REFERENCE_WORD2VEC_MODEL_PATH)
	if 'fasttext' in model_names:
		gem.build_fasttext_model(corpus_import_path=REFERENCE_CORPUS_PATH, model_export_path=REFERENCE_FASTTEXT_MODEL_PATH)
	
	save_args(args=args, save_path=os.path.join(TEMP_DIR, 'EmbeddingModelConfig.json'))
	
if __name__ == '__main__':
	# makedirs()
	# preprocess_trainsets_and_testsets()
	# preprocess_reference_book()
	# build_gensim_retrieval_models(model_names=['tfidf', 'lsi', 'lda', 'hdp'], update_reference_corpus=True)
	build_gensim_embedding_models(model_names=['word2vec', 'fasttext'])
