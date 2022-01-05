# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 一些用于代码可用性测试的脚本

import json
import time
import torch
import numpy
import gensim
import pandas

from config import DatasetConfig, RetrievalModelConfig, EmbeddingModelConfig
from setting import *
from preprocess import build_gensim_retrieval_models
from gensim.corpora import Dictionary, MmCorpus

from src.dataset import Dataset, generate_dataloader
from src.retrieval_model import GensimRetrievalModel
from src.embedding_model import GensimEmbeddingModel
from src.evaluation_tools import evaluate_gensim_model_in_filling_subject
from src.utils import load_args

os.makedirs(NEWDATA_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RETRIEVAL_MODEL_DIR, exist_ok=True)
os.makedirs(GENSIM_RETRIEVAL_MODEL_DIR, exist_ok=True)
os.makedirs(EMBEDDING_MODEL_DIR, exist_ok=True)
os.makedirs(GENSIM_RETRIEVAL_MODEL_DIR, exist_ok=True)

# 测试dataset.py运行情况
def test_dataset():
	args = load_args(Config=DatasetConfig)
	args.use_reference = True
	args.retrieval_model_name = 'tfidf'
	args.train_batch_size = 2
	args.valid_batch_size = 2
	args.test_batch_size = 2
	
	# args.word_embedding = 'word2vec'
	# args.document_embedding = None
	# for pipeline in ['judgment']:
		# for mode in ['train', 'valid', 'test']:
			# print(pipeline, mode, args.word_embedding, args.document_embedding)
			# dataloader = generate_dataloader(args=args, mode=mode, do_export=False, pipeline=pipeline, for_test=True)
			# for i, data in enumerate(dataloader):
				# print(i)
	
	# 测试2
	args.word_embedding = None
	
	for document_embedding in ['doc2vec', 'bert-base-chinese']:
		args.document_embedding = document_embedding
		for pipeline in ['judgment']:
			for mode in ['train', 'valid', 'test']:
				print(pipeline, mode, args.word_embedding, args.document_embedding)
				dataloader = generate_dataloader(args=args, mode=mode, do_export=False, pipeline=pipeline, for_test=True)
				for i, data in enumerate(dataloader):
					print(i)
					
# tfidf调参
def test_tfidf():
	args = load_args(Config=RetrievalModelConfig)	
	summary = []
	count = 0
	for pivot in [None, 1.]:
		args.pivot_tfidf = pivot
		for slope in [.25, .5]:
			args.slope_tfidf = slope
			for a in ['b', 'n', 'a', 'l', 'd']:
				for b in ['n', 'f', 't', 'p']:
					for c in ['n', 'c', 'u', 'b']:
						try:
							count += 1
							args.smartirs_tfidf = a + b + c
							print(count, args.pivot_tfidf, args.slope_tfidf, args.smartirs_tfidf)
							build_gensim_retrieval_models(args=args, model_names=['tfidf'], update_reference_corpus=False)
							_summary = evaluate_gensim_model_in_filling_subject(gensim_retrieval_model_names=['tfidf'], 
																			    gensim_embedding_model_names=[],
																			    hits=[1, 3, 5, 10])													   
							temp_summary = {'args': {'smartirs': args.smartirs_tfidf, 'pivot': args.pivot_tfidf, 'slope': args.slope_tfidf}, 'result': _summary}
							summary.append(temp_summary)
						except Exception as e:
							with open('error.txt', 'a') as f:
								f.write(f'{args.pivot_tfidf} - {args.slope_tfidf} - {args.smartirs_tfidf}')
								f.write(str(e))
								f.write('\n')

	with open(os.path.join(TEMP_DIR, 'test_smartirs.json'), 'w', encoding='utf8') as f:
		json.dump(model_name(summary, f, indent=4))


# plot
def plot():
	# for mode in ['train_kd', 'train_ca']:
		# train_logging_dataframe = pandas.read_csv(os.path.join(TEMP_DIR, 'summary', 'BaseChoiceModel', f'BaseChoiceModel_{mode}.csv'), header=0, sep='\t')
		# valid_logging_dataframe = pandas.read_csv(os.path.join(TEMP_DIR, 'summary', 'BaseChoiceModel', f'BaseChoiceModel_{mode.replace("train", "valid")}.csv'), header=0, sep='\t')
		# train_plot_choice(model_name=f'BaseChoiceModel{mode.split("_")[-1].upper()}', 
						  # train_logging_dataframe=train_logging_dataframe, 
						  # valid_logging_dataframe=valid_logging_dataframe,
						  # train_plot_export_path=os.path.join(IMAGE_DIR, f'BaseChoiceModel_{mode}.png'),
						  # valid_plot_export_path=os.path.join(IMAGE_DIR, f'BaseChoiceModel_{mode.replace("train", "valid")}.png'))

	for mode in ['train_kd', 'train_ca']:
		train_logging_dataframe = pandas.read_csv(os.path.join(TEMP_DIR, 'summary', 'BaseJudgmentModel', f'BaseJudgmentModel_{mode}.csv'), header=0, sep='\t')
		valid_logging_dataframe = pandas.read_csv(os.path.join(TEMP_DIR, 'summary', 'BaseJudgmentModel', f'BaseJudgmentModel_{mode.replace("train", "valid")}.csv'), header=0, sep='\t')
		train_plot_judgment(model_name=f'BaseJudgmentModel{mode.split("_")[-1].upper()}', 
							train_logging_dataframe=train_logging_dataframe, 
							valid_logging_dataframe=valid_logging_dataframe,
							train_plot_export_path=os.path.join(IMAGE_DIR, f'BaseJudgmentModel_{mode}.png'),
							valid_plot_export_path=os.path.join(IMAGE_DIR, f'BaseJudgmentModel_{mode.replace("train", "valid")}.png'))



if __name__ == '__main__':
	
	# evaluate_gensim_model_in_filling_subject(gensim_retrieval_model_names=[], 
											 # gensim_embedding_model_names=['word2vec', 'fasttext'],
											 # hits=[1, 3, 5, 10])
	test_dataset()
	
	
	# plot()
