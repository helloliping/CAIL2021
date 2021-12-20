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
from src.retrieval_model import GensimRetrieveModel
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
	args.word_embedding = None
	for retrieval_model_name in ['tfidf']:
		args.retrieval_model_name = retrieval_model_name
		for pipeline in ['choice', 'judgment']:
			# for mode in ['test_ca', 'test', 'test_kd', 'train', 'train_ca', 'train_kd', 'valid', 'valid_ca', 'valid_kd']:
			# for mode in ['test', 'train', 'valid']:
			for mode in ['train_kd']:
				print(pipeline, mode)
				dataloader = generate_dataloader(args=args, mode=mode, do_export=False, pipeline=pipeline)
				for i, data in enumerate(dataloader):
					print(i, data['reference'].shape)
	
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

if __name__ == '__main__':
	
	# evaluate_gensim_model_in_filling_subject(gensim_retrieval_model_names=[], 
											 # gensim_embedding_model_names=['word2vec', 'fasttext'],
											 # hits=[1, 3, 5, 10])
	test_dataset()
