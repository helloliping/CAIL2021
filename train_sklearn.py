# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
# 模型训练: 使用sklearn

import os
import gc
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

from sklearn import preprocessing

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import auc, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve

from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier

from matplotlib import pyplot as plt

from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, lr_scheduler

from setting import *
from config import QAModelConfig, DatasetConfig

from src.dataset import generate_dataloader, Dataset
from src.evaluation_tools import evaluate_qa_model_choice, evaluate_qa_model_judgment, evaluate_classifier
from src.plot_tools import plot_roc_curve, plot_pr_curve
from src.qa_model import BaseChoiceModel, BaseJudgmentModel, ReferenceChoiceModel, ReferenceJudgmentModel
from src.torch_tools import save_checkpoint
from src.utils import initialize_logger, terminate_logger, load_args, save_args
from src.easy_machine_learning import EasyClassifier


def generate_dataset_for_sklearn(args, mode):

	dataset_train = Dataset(args=args, mode=mode, do_export=False, pipeline='judgment').data
	dataset_valid = Dataset(args=args, mode=mode.replace('train', 'valid'), do_export=False, pipeline='judgment').data
	dataset_test = Dataset(args=args, mode=mode.replace('train', 'test'), do_export=False, pipeline='judgment').data
	
	dataset_train = pd.concat([dataset_train, dataset_valid])
	
	'''
	id				: 题目编号
	question		: 题目题干
	option			: 每个选项
	subject			: use_reference配置为True时生效, 包含num_top_subject个法律门类
	reference		: use_reference配置为True时生效, 包含相关的num_best个参考书目文档段落
	type			: 零一值表示概念题或情景题
	label_judgment	: train或valid模式时生效, 零一值表示判断题的答案
	option_id		: 20211216更新, 记录判断题对应的原选择题编号(ABCD)
	'''
	
	if args.use_reference:
		dataset_X = np.hstack([np.vstack(dataset_train['question'].map(np.array).values),
							   np.vstack(dataset_train['option'].map(np.array).values)])
		test_X = np.hstack([np.vstack(dataset_test['question'].map(np.array).values),
							np.vstack(dataset_test['option'].map(np.array).values)])
	else:
		dataset_X = np.hstack([np.vstack(dataset_train['question'].map(np.array).values),
							   np.vstack(dataset_train['option'].map(np.array).values),
							   np.vstack(dataset_train['reference']).map(lambda x: np.array(x).reshape((-1, )))])
		test_X = np.hstack([np.vstack(dataset_test['question'].map(np.array).values),
							np.vstack(dataset_test['option'].map(np.array).values),
							np.vstack(dataset_test['reference']).map(lambda x: np.array(x).reshape((-1, )))])

	dataset_y = dataset['label_judgment'].values	
	
	return dataset_X, dataset_y, test_X

def train_model(model_name, mode='train_kd', **kwargs):
	args = load_args(DatasetConfig)
	for key, value in kwargs.items():
		args.__setattr__(key, value)	
	
	dataset_X, dataset_y, test_X = generate_dataset_for_sklearn(args=args, mode=mode)
	
	if model_name == 'lgb':
		params = {
			'boosting_type': 'gbdt',
			'num_leaves': 256,
			'max_depth': 8,
			'learning_rate': .001,
			'n_estimators': 128,
			'subsample_for_bin': 200000,
			'objective': None,
			'class_weight': None,
			# 'min_split_gain': .0,
			# 'min_child_weight': 0,
			# 'min_child_samples': 100,
			# 'subsample': 1.,
			# 'subsample_freq': 0,
			# 'colsample_bytree': 1.,
			'reg_alpha': 0,
			'reg_lambda': 0,
			'random_state': None,
			'n_jobs': 1,
			'silent': True,
			'importance_type': 'split',
		}
		
		easyclf = EasyClassifier(dataset_X, dataset_y, LGBMClassifier, params)
		easyclf.kfold_train(n_splits=5,
							shuffle=True,
							pr_title_formatter='PR Curve of LGBM {}'.format,
							roc_title_formatter='ROC Curve of LGBM {}'.format,
							model_export_path_formatter='temp/lgbm_fold_{}.m'.format,
							pr_export_path_formatter='temp/pr_curve_fold_{}.png'.format,
							roc_export_path_formatter='temp/roc_curve_fold_{}.png'.format,
							evaluation_export_path='temp/lgbm_eval.json')
	if model_name == 'xgb':
		# XGBoost

		params = {
			'n_estimators': 128,
			'use_label_encoder': False,
			'max_depth': 8,
			'num_leaves': 256,
			'learning_rate': .01,
			'verbosity': 0,
			# 'objective': None,
			'booster': 'dart',
			# 'tree_method': None,
			'n_jobs': 1,
			'gamma': .001,
			# 'min_child_weight': .01,
			# 'max_delta_step': .01,
			# 'subsample': 1.,
			# 'colsample_bytree': 1.,
			# 'colsample_bylevel': 1.,
			# 'colsample_bynode': 1.,
			'reg_alpha': .01,
			'reg_lambda': .01,
			# 'scale_pos_weight': .01,
			# 'base_score': 1,
			'random_state': None,
			# 'missing': np.nan,
			'num_parallel_tree': 10,
			# 'monotone_constraints': None,
			# 'interaction_constraints': None,
			# 'importance_type': 'gain',
			# 'gpu_id': None,
			# 'validate_parameters': None,
		}

		easyclf = EasyClassifier(dataset_X, dataset_y, XGBClassifier, params)
		easyclf.kfold_train(n_splits=5,
							shuffle=True,
							pr_title_formatter='PR Curve of XGB {}'.format,
							roc_title_formatter='ROC Curve of XGB {}'.format,
							model_export_path_formatter='temp/xgb_fold_{}.m'.format,
							pr_export_path_formatter='temp/pr_curve_fold_{}.png'.format,
							roc_export_path_formatter='temp/roc_curve_fold_{}.png'.format,
							evaluation_export_path='temp/xgb_eval.json')
		
kwargs = {
	'use_reference'	: True,
	'num_best': 32,
}

train_model(model_name='lgb', mode='train_kd', **kwargs)
