# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn

import os
import json
import time
import pandas

from config import DatasetConfig, RetrievalModelConfig, EmbeddingModelConfig
from setting import *
from preprocess import build_gensim_retrieval_models

from src.dataset import Dataset
from src.retrieval_model import GensimRetrieveModel
from src.embedding_model import GensimEmbeddingModel
from src.evaluation_tools import evaluate_gensim_model_in_filling_subject
from src.plot_tools import train_plot_choice, train_plot_judgment
from src.utils import load_args

		
os.makedirs(NEWDATA_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(RETRIEVAL_MODEL_DIR, exist_ok=True)
os.makedirs(GENSIM_RETRIEVAL_MODEL_DIR, exist_ok=True)
os.makedirs(EMBEDDING_MODEL_DIR, exist_ok=True)
os.makedirs(GENSIM_EMBEDDING_MODEL_DIR, exist_ok=True)

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
						



